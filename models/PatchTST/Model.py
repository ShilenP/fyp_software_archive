# PatchTST model adpated from the original implementation in the paper "A TIME SERIES IS WORTH 64 WORDS: LONG-TERM FORECASTING WITH TRANSFORMERS" by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/models/PatchTST.py


# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
# Cell

import math

from utils.config import Config
from fastbound.bound_propagation import ibp_propagation
# code from https://github.com/ts-kim/RevIN, with minor modifications


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    
    def forward_bounds(self, x_lower, x_upper, mode:str):
        if mode == 'norm':
            self._get_statistics((x_lower + x_upper) / 2)
            x_lower = self._normalize(x_lower)
            x_upper = self._normalize(x_upper)
        elif mode == 'denorm':
            x_lower = self._denormalize(x_lower)
            x_upper = self._denormalize(x_upper)
        else: raise NotImplementedError
        return x_lower, x_upper

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

    
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable') 
    
    
# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
    
# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(q_len, d_model):
    W_pos = torch.empty((q_len, d_model))
    nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=True)


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, attn_dropout:float=0., dropout:float=0., act:str="gelu",
                 padding_var:Optional[int]=None,
                 head_dropout = 0, padding_patch = None,
                 head_type = 'flatten',
                 verbose:bool=False, **kwargs):
        
        super().__init__()

        # RevIn
        self.revin_layer = RevIN(c_in)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)

        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        patch_num += 1
    
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, padding_var=padding_var, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type

        self.head = Flatten_Head(self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        z = z.permute(0,2,1)
        z = self.revin_layer(z, 'norm')
        z = z.permute(0,2,1)
            
        # do patching
        z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        z = z.permute(0,2,1)
        z = self.revin_layer(z, 'denorm')
        z = z.permute(0,2,1)
        return z
    
    def forward_bounds(self, z_lower, z_upper):
        z_lower, z_upper = z_lower.permute(0,2,1), z_upper.permute(0,2,1)
        z_lower, z_upper = self.revin_layer.forward_bounds(z_lower, z_upper, 'norm')
        z_lower, z_upper = z_lower.permute(0,2,1), z_upper.permute(0,2,1)

        z_lower, z_upper = ibp_propagation(z_lower, z_upper, self.padding_patch_layer)
        z_lower, z_upper = z_lower.unfold(dimension=-1, size=self.patch_len, step=self.stride), z_upper.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_lower, z_upper = z_lower.permute(0,1,3,2), z_upper.permute(0,1,3,2)

        z_lower, z_upper = self.backbone.forward_bounds(z_lower, z_upper)
        z_lower, z_upper = self.head.forward_bounds(z_lower, z_upper)

        z_lower, z_upper = z_lower.permute(0,2,1), z_upper.permute(0,2,1)
        z_lower, z_upper = self.revin_layer.forward_bounds(z_lower, z_upper, 'denorm')
        z_lower, z_upper = z_lower.permute(0,2,1), z_upper.permute(0,2,1)
        return z_lower, z_upper

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
    def forward_bounds(self, x_lower, x_upper):
        x_lower, x_upper = ibp_propagation(x_lower, x_upper, self.flatten)
        x_lower, x_upper = ibp_propagation(x_lower, x_upper, self.linear)
        x_lower, x_upper = ibp_propagation(x_lower, x_upper, self.dropout)
        return x_lower, x_upper
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, attn_dropout=0., dropout=0., act="gelu", padding_var=None,
                 verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                                   activation=act, n_layers=n_layers)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
    
    def forward_bounds(self, x_lower, x_upper):
        n_vars = x_lower.shape[1]
        x_lower, x_upper = x_lower.permute(0,1,3,2), x_upper.permute(0,1,3,2)
        x_lower, x_upper = ibp_propagation(x_lower, x_upper, self.W_P)
        u_lower, u_upper = torch.reshape(x_lower, (x_lower.shape[0]*x_lower.shape[1],x_lower.shape[2],x_lower.shape[3])), torch.reshape(x_upper, (x_upper.shape[0]*x_upper.shape[1],x_upper.shape[2],x_upper.shape[3]))
        u_lower, u_upper = u_lower + self.W_pos, u_upper + self.W_pos
        u_lower, u_upper = ibp_propagation(u_lower, u_upper, self.dropout)

        z_lower, z_upper = self.encoder.forward_bounds(u_lower, u_upper)
        z_lower, z_upper = torch.reshape(z_lower, (-1,n_vars,z_lower.shape[-2],z_lower.shape[-1])), torch.reshape(z_upper, (-1,n_vars,z_upper.shape[-2],z_upper.shape[-1]))
        z_lower, z_upper = z_lower.permute(0,1,3,2), z_upper.permute(0,1,3,2)
        return z_lower, z_upper
        

    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        attn_dropout=0., dropout=0., activation='gelu',
                        n_layers=1, pre_norm=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation) for _ in range(n_layers)])

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        for mod in self.layers: output, scores = mod(output, prev=scores)
        return output
    
    def forward_bounds(self, src_lower, src_upper, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output_lower, output_upper = src_lower, src_upper
        scores_lower, scores_upper = None, None
        for mod in self.layers: 
            output_lower, output_upper, scores_lower, scores_upper = mod.forward_bounds(output_lower, output_upper, scores_lower, scores_upper)
        return output_lower, output_upper



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256,
                attn_dropout=0, dropout=0., bias=True, activation="gelu"):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))


    def forward(self, src:Tensor, prev:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn, scores = self.self_attn(src, src, src, prev)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.norm_attn(src)

        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.norm_ffn(src)

        return src, scores
    
    def forward_bounds(self, src_lower, src_upper, prev_lower, prev_upper):
        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2_lower, src2_upper, _, _, scores_lower, scores_upper = self.self_attn.forward_bounds(src_lower, src_upper, prev_lower, prev_upper)
        ## Add & Norm
        src_lower = src_lower + src2_lower
        src_upper = src_upper + src2_upper

        # TODO: This is not correct
        src_lower, src_upper = self.norm_attn(src_lower), self.norm_attn(src_upper)

        src2_lower, src2_upper = ibp_propagation(src_lower, src_upper, self.ff)

        src_lower = src_lower + src2_lower
        src_upper = src_upper + src2_upper
        # TODO: This is not correct
        src_lower, src_upper = self.norm_attn(src_lower), self.norm_attn(src_upper)
        return src_lower, src_upper, scores_lower, scores_upper







class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, lsa=lsa)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights, attn_scores

    def forward_bounds(self, Q_lower, Q_upper, prev_lower=None, prev_upper=None):
        bs = Q_lower.size(0)
        K_lower, K_upper, V_lower, V_upper = Q_lower, Q_upper, Q_lower, Q_upper
        
        q_s_lower, q_s_upper = ibp_propagation(Q_lower, Q_upper, self.W_Q)
        q_s_lower, q_s_upper = q_s_lower.view(bs, -1, self.n_heads, self.d_k).transpose(1,2), q_s_upper.view(bs, -1, self.n_heads, self.d_k).transpose(1,2)

        k_s_lower, k_s_upper = ibp_propagation(K_lower, K_upper, self.W_K)
        k_s_lower, k_s_upper = k_s_lower.view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1), k_s_upper.view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)

        v_s_lower, v_s_upper = ibp_propagation(V_lower, V_upper, self.W_V)
        v_s_lower, v_s_upper = v_s_lower.view(bs, -1, self.n_heads, self.d_v).transpose(1,2), v_s_upper.view(bs, -1, self.n_heads, self.d_v).transpose(1,2)

        output_lower, output_upper, attn_weights_lower, attn_weights_upper, attn_scores_lower, attn_scores_upper = self.sdp_attn.forward_bounds(q_s_lower, q_s_upper, k_s_lower, k_s_upper, v_s_lower, v_s_upper, prev_lower=prev_lower, prev_upper=prev_upper)

        output_lower, output_upper = output_lower.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v), output_upper.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output_lower, output_upper = ibp_propagation(output_lower, output_upper, self.to_out)

        return output_lower, output_upper, attn_weights_lower, attn_weights_upper, attn_scores_lower, attn_scores_upper






class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev


        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights, attn_scores
    
    def forward_bounds(self, q_lower, q_upper, k_lower, k_upper, v_lower, v_upper, prev_lower=None, prev_upper=None):
        # TODO: This is not correct
        attn_scores_lower = torch.matmul(q_lower, k_lower)
        attn_scores_upper = torch.matmul(q_upper, k_upper)
        if self.scale >= 0:
            attn_scores_lower, attn_scores_upper = attn_scores_lower * self.scale, attn_scores_upper * self.scale
        else:
            attn_scores_lower, attn_scores_upper = attn_scores_upper * self.scale, attn_scores_lower * self.scale
        if prev_lower is not None and prev_upper is not None:
            attn_scores_lower, attn_scores_upper = attn_scores_lower + prev_lower, attn_scores_upper + prev_upper
        
        # TODO: This is not correct
        attn_weights_lower = F.softmax(attn_scores_lower, dim=-1)
        attn_weights_upper = F.softmax(attn_scores_upper, dim=-1)

        attn_weights_lower, attn_weights_upper = ibp_propagation(attn_weights_lower, attn_weights_upper, self.attn_dropout)

        # TODO: This is not correct
        output_lower = torch.matmul(attn_weights_lower, v_lower)
        output_upper = torch.matmul(attn_weights_upper, v_upper)
        return output_lower, output_upper, attn_weights_lower, attn_weights_upper, attn_scores_lower, attn_scores_upper


class Model(nn.Module):
    def __init__(self, config: Config, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, 
                 head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = config.enc_in
        context_window = config.seq_len
        target_window = config.pred_len
        
        n_layers = config.e_layers
        n_heads = config.n_heads
        d_model = config.d_model
        d_ff = config.d_ff
        dropout = config.dropout
        fc_dropout = config.fc_dropout
        head_dropout = config.head_dropout
        
    
        patch_len = config.patch_len
        stride = config.stride
        
        
        self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, fc_dropout=fc_dropout, head_dropout=head_dropout, 
                                head_type=head_type, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    
    def forward_bounds(self, x_lower, x_upper):
        x_lower, x_upper = x_lower.permute(0,2,1), x_upper.permute(0,2,1)
        x_lower, x_upper = self.model.forward_bounds(x_lower, x_upper)
        x_lower, x_upper = x_lower.permute(0,2,1), x_upper.permute(0,2,1)
        return x_lower, x_upper
    
    def __str__(self):
        return 'PatchTST'