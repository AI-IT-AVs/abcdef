import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class get_point_correspondences(nn.Module):
    def __init__(self,):
        super(get_point_correspondences, self).__init__()

    def forward(self, src, tgt, src_emb, tgt_emb):

        batch_size, n_dims, num_points = src.size()
        # Calculate the distance matrix
        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        scores = torch.softmax(pairwise_distance, dim=2)  # [b,num,num]
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        
   
        return src_corr
