
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from IPython import embed

from posdiff.modules.layers import build_dropout_layer
from posdiff.modules.transformer.output_layer import AttentionOutput


class PPCrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(PPCrossMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_r = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(torch.Tensor([0.0]), requires_grad=False).to(device)
        self.w = nn.Parameter(torch.Tensor([0.0]), requires_grad=False).to(device)


    def forward(self, input_q, input_k, input_v, embed_q, embed_k, key_weights=None, key_masks=None, attention_factors=None):

        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_q), 'b n (h c) -> b h n c', h=self.num_heads)
        r = rearrange(self.proj_p(embed_k), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhmc->bhnm', p, r)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = ( (torch.exp(-self.w)*attention_scores_e+self.w) + (torch.exp(-self.weight)*attention_scores_p+self.weight) ) / self.d_model_per_head ** 0.5

        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class PPCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(PPCrossAttentionLayer, self).__init__()
        self.attention = PPCrossMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states_input,
        position_states_memory,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states_input,
            position_states_memory,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class PPCrossTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(PPCrossTransformerLayer, self).__init__()
        self.attention = PPCrossAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states_input,
        position_states_memory,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states_input,
            position_states_memory,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
