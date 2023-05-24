import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from posdiff.modules.ops import point_to_node_partition, index_select
from posdiff.modules.registration import get_node_correspondences
from posdiff.modules.sinkhorn import LearnableLogOptimalTransport
from posdiff.modules.posdiff import (
    PosDiffTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN

from posdiff.modules.posdiff.posidff_module import PosDiffNet
from torchdiffeq import odeint
from posdiff.modules.posdiff.point_correspondence import get_point_correspondences
from posdiff.modules.ops import apply_transform

class PosDiffuNet_model(nn.Module):
    def __init__(self, cfg):
        super(PosDiffuNet_model, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.posdiff_layer_f = PosDiffNet(emb_dims = cfg.backbone.output_dim*2)
        self.posdiff_layer_c = PosDiffNet(emb_dims = cfg.posdiff.input_dim*2)

        self.transformer = PosDiffTransformer(
            cfg.posdiff.input_dim,
            cfg.posdiff.output_dim,
            cfg.posdiff.hidden_dim,
            cfg.posdiff.num_heads,
            cfg.posdiff.blocks,
            reduction_a=cfg.posdiff.reduction_a,
        )


        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching_c = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.coarse_matching_f = SuperPointMatching(
            cfg.coarse_matching.num_point_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.point_corresponding = get_point_correspondences()


        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c # [ref_length_c, 3]
        output_dict['src_points_c'] = src_points_c # [src_length_c, 3]
        output_dict['ref_points_f'] = ref_points_f # [ref_length_f, 3]
        output_dict['src_points_f'] = src_points_f # [src_length_f, 3]
        output_dict['ref_points'] = ref_points   # [ref_length, 3]
        output_dict['src_points'] = src_points   # [src_length, 3]

        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        ) #  _, [ref_length_c], [ref_length_c, k], [ref_length_c, k]


        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        ) #  _, [src_length_c], [src_length_c, k], [src_length_c, k]

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0) # [ref_length_c, k, 3]
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0) # [src_length_c, k, 3]

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        ) # [ ref_node_knn_points_overlap, src_node_knn_points_overlap], [ref_src_node_knn_points_overlap]

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps # rate of overlapped points

        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        ref_feats_c = feats_c[:ref_length_c] # [ref_length_c, 2048] --> [num, dim]
        src_feats_c = feats_c[ref_length_c:] # [src_length_c. 2048] --> [num, dim]
        ref_feats_f = feats_f[:ref_length_f] # [ref_length_f, 256] --> [num, dim]
        src_feats_f = feats_f[ref_length_f:] # [src_length_f, 256] --> [num, dim]

        device = feats_c.device
        ref_feats_c_pos = torch.zeros_like(ref_feats_c).to(device) # [ref_length_c, 2048] --> [num, dim]
        src_feats_c_pos = torch.zeros_like(src_feats_c).to(device) # [src_length_c, 2048] --> [num, dim]
        ref_feats_c_pos[:, 0:3] = ref_points_c # position 
        src_feats_c_pos[:, 0:3] = src_points_c # position
        ref_feats_c_pp = torch.cat([ref_feats_c, ref_feats_c_pos], dim=1) # [ref_length_c, 4096] --> [num, dim]
        src_feats_c_pp =  torch.cat([src_feats_c, src_feats_c_pos], dim=1) # [src_length_c, 4096] --> [num, dim]
        ref_feats_c_pp_out = self.posdiff_layer_c(ref_feats_c_pp.transpose(1,0).unsqueeze(0)) # posdiff # [batch, dim, num]
        src_feats_c_pp_out = self.posdiff_layer_c(src_feats_c_pp.transpose(1,0).unsqueeze(0)) # posdiff # [batch, dim, num]
        ref_feats_c_pp_out = ref_feats_c_pp_out.squeeze(0).transpose(1,0) # [ref_length_c, 4096] --> [num, dim]
        src_feats_c_pp_out = src_feats_c_pp_out.squeeze(0).transpose(1,0) # [src_length_c, 4096] --> [num, dim]
        ref_feats_c = ref_feats_c_pp_out[:, :ref_feats_c_pp_out.shape[1]//2]   # [ref_length_c, 2048] --> [num, dim]
        src_feats_c = src_feats_c_pp_out[:, :src_feats_c_pp_out.shape[1]//2]  # [src_length_c, 2048] --> [num, dim]
        ref_feats_c_pos = ref_feats_c_pp_out[:, ref_feats_c_pp_out.shape[1]//2:] # [ref_length_c, 2048] --> [num, dim]
        src_feats_c_pos = src_feats_c_pp_out[:, src_feats_c_pp_out.shape[1]//2:] # [ref_length_c, 2048] --> [num, dim]

        ref_feats_f_pos = torch.zeros_like(ref_feats_f).to(device) # [ref_length_f, 256] --> [num, dim]
        src_feats_f_pos = torch.zeros_like(src_feats_f).to(device) # [src_length_f, 256] --> [num, dim]
        ref_feats_f_pos[:, 0:3] = ref_points_f 
        src_feats_f_pos[:, 0:3] = src_points_f
        ref_feats_f_pp = torch.cat([ref_feats_f, ref_feats_f_pos], dim=1) # [ref_length_f, 512] --> [num, dim]
        src_feats_f_pp =  torch.cat([src_feats_f, src_feats_f_pos], dim=1) # [src_length_f, 512] --> [num, dim]
        ref_feats_f_pp_out = self.posdiff_layer_f(ref_feats_f_pp.transpose(1,0).unsqueeze(0)) 
        src_feats_f_pp_out = self.posdiff_layer_f(src_feats_f_pp.transpose(1,0).unsqueeze(0))
        ref_feats_f_pp_out = ref_feats_f_pp_out.squeeze(0).transpose(1,0) # [ref_length_f, 512] --> [num, dim]
        src_feats_f_pp_out = src_feats_f_pp_out.squeeze(0).transpose(1,0) # [src_length_f, 512] --> [num, dim]
        ref_feats_f = ref_feats_f_pp_out[:, :ref_feats_f_pp_out.shape[1]//2] # [ref_length_f, 256] --> [num, dim]
        src_feats_f = src_feats_f_pp_out[:, :src_feats_f_pp_out.shape[1]//2] # [src_length_f, 256] --> [num, dim]
        ref_feats_f_pos = ref_feats_f_pp_out[:, ref_feats_f_pp_out.shape[1]//2:] # [ref_length_f, 256] --> [num, dim]
        src_feats_f_pos = src_feats_f_pp_out[:, src_feats_f_pp_out.shape[1]//2:] # [src_length_f, 256] --> [num, dim]


        ref_feats_c, src_feats_c = self.transformer(
            ref_feats_c_pos.unsqueeze(0),
            src_feats_c_pos.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )

        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching_c(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores


        ref_point_corr_indices, src_point_corr_indices, point_corr_scores = self.coarse_matching_f(ref_feats_f, src_feats_f) # [num_point_correspondences], [num_point_correspondences], [num_point_correspondences]    
        ref_feats_f_norm_corr = ref_feats_f[ref_point_corr_indices] #[num_point_correspondences, dim=256]
        src_feats_f_norm_corr = src_feats_f[src_point_corr_indices] #[num_point_correspondences, dim=256]
        ref_points_f_corr = ref_points_f[ref_point_corr_indices]     #[num_point_correspondences, 3]
        src_points_f_corr = src_points_f[src_point_corr_indices]     #[num_point_correspondences, 3]
        ref_points_f_corr_src = ref_points_f_corr
        
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform, transfered_src_points_f_corr= self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
                src_points_f_corr,
                ref_points_f_corr_src,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform
            output_dict['transfered_src_points_f_corr'] = transfered_src_points_f_corr
            output_dict['ref_points_f_corr_src'] = ref_points_f_corr_src


        return output_dict


def create_model(cfg):
    model = PosDiffuNet_model(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
