import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from posdiff.utils.common import ensure_dir

_C = edict()


_C.seed = 1234


_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = '.../' 
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join('.../','output') 
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'log')
_C.event_dir = osp.join(_C.output_dir, 'event')
_C.feature_dir = osp.join(_C.output_dir, 'prediction')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)

_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir)

_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 768
_C.train.use_augmentation = False
_C.train.augmentation_noise = 0.01
_C.train.augmentation_min_scale = 0.8
_C.train.augmentation_max_scale = 1.2
_C.train.augmentation_shift = 2.0
_C.train.augmentation_rotation = 1.0

_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.point_limit = None

_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 1.0
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 5.0
_C.eval.rte_threshold = 2.0

_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 4
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 50
_C.optim.grad_acc_steps = 1

_C.backbone = edict()
_C.backbone.num_stages = 5
_C.backbone.init_voxel_size = 0.3
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 4.25
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

_C.model = edict()
_C.model.ground_truth_matching_radius = 0.1
_C.model.num_points_in_patch = 20
_C.model.num_sinkhorn_iterations = 100

_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 786
_C.coarse_matching.dual_normalization = True
_C.coarse_matching.num_point_correspondences = 786

_C.posdiff = edict()
_C.posdiff.input_dim = 2048
_C.posdiff.hidden_dim = 128
_C.posdiff.output_dim = 256
_C.posdiff.num_heads = 4
_C.posdiff.blocks = ['self', 'cross']
_C.posdiff.sigma_d = 4.8
_C.posdiff.sigma_a = 15
_C.posdiff.angle_k = 3
_C.posdiff.reduction_a = 'max'

_C.fine_matching = edict()
_C.fine_matching.topk = 1
_C.fine_matching.acceptance_radius = 3.0
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1


_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.1


_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0


_C.num_points = 1024
_C.gaussian_noise=False
_C.partial=False
_C.dataset = 'kitti'
_C.reserve=1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
