import numpy as np
import torch
import torch.nn as nn

from posdiff.modules.ops import pairwise_distance
from posdiff.modules.transformer import SinusoidalPositionalEmbedding
from posdiff.modules.transformer.conditional_transformer import PPConditionalTransformer
from torchdiffeq import odeint


class PosDiffTransformer_module(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):

        super(PosDiffTransformer_module, self).__init__()

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_proj_pos = nn.Linear(input_dim, hidden_dim)

        self.transformer = PPConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_feats_pos, 
        src_feats_pos, 
        ref_feats, 
        src_feats, 
        ref_masks, 
        src_masks, 
    ):

        ref_embeddings = self.in_proj_pos(ref_feats_pos)
        src_embeddings = self.in_proj_pos(src_feats_pos)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats



class PosDiffTransformer_func(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):

        super(PosDiffTransformer_func, self).__init__()

        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.in_proj_pos = nn.Linear(hidden_dim, hidden_dim)

        self.transformer = PPConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        t, input
    ):

        ref_feats_pos = input[0]
        src_feats_pos = input[1]
        ref_feats = input[2]
        src_feats = input[3]
        ref_masks = None
        src_masks = None

        ref_embeddings = self.in_proj_pos(ref_feats_pos)
        src_embeddings = self.in_proj_pos(src_feats_pos)

        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        output = (ref_feats_pos, src_feats_pos, ref_feats, src_feats)
        return output

    

class PosDiffTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):

        super(PosDiffTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.blocks = blocks
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.reduction_a = reduction_a

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_proj_pos = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.odefunc = PosDiffTransformer_func( self.input_dim, 
                                                self.output_dim,
                                                self.hidden_dim,
                                                self.num_heads,
                                                self.blocks,
                                                self.dropout,
                                                self.activation_fn,
                                                self.reduction_a )

        self.odeint = odeint

        tol_scale = torch.tensor([1.0, 1.0, 1.0, 1.0], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.atol = tol_scale * 1e-2
        self.rtol = tol_scale * 1e-2 

        self.method = 'euler'
        self.step_size = 1.0
        self.t = torch.tensor([0, 2.0], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(
        self,
        ref_feats_pos,
        src_feats_pos,
        ref_feats,
        src_feats,
        ref_masks=None,
        src_masks=None,
    ):

        ref_feats_pos = self.in_proj_pos(ref_feats_pos)
        src_feats_pos = self.in_proj_pos(src_feats_pos)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        input = (ref_feats_pos, src_feats_pos, ref_feats, src_feats)

        t = self.t
        integrator = self.odeint
        func = self.odefunc
        state = input
        state_dt = integrator(
            func, state, t,
            method=self.method,
            options={'step_size': self.step_size},
            atol=self.atol,
            rtol=self.rtol)
        z_out = state_dt

        ref_feats_pos = z_out[0][1]
        src_feats_pos = z_out[1][1]
        ref_feats = z_out[2][1]
        src_feats = z_out[3][1]

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)


        return ref_feats, src_feats


