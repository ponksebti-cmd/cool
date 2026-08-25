import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 32000
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    hybrid_pattern: str = "3:1"  # e.g., 3 Mamba-2 blocks for every 1 Attention block

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:, :seq_len, :, :]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class Mamba2Block(nn.Module):
    """
    Pure PyTorch implementation of Mamba-2 State Space Duality (SSD) core.
    By using PyTorch's native associative operations (einsum, cumsum), this avoids 
    the need for custom CUDA kernels (mamba_ssm) which would fail to compile 
    or run on an AMD RX 6650 XT GPU in Windows. 
    It computes the SSD natively through dual masked causal attention!
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.d_inner = args.dim * 2
        self.d_state = 64
        self.nheads = args.n_heads
        self.head_dim = self.d_inner // self.nheads
        
        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=True,
            kernel_size=4, groups=self.d_inner, padding=3
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.nheads + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.nheads, self.nheads, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=False)
        
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        
    def forward(self, x, freqs_cis=None, mask=None):
        # freqs_cis is ignored for Mamba blocks, as they don't use RoPE
        bsz, seqlen, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        # Causal Conv1d
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.nheads, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        x_in_reshaped = x_in.view(bsz, seqlen, self.nheads, self.head_dim)
        
        A = -torch.exp(self.A_log.float())
        dtA = dt * A
        cumsum_dtA = torch.cumsum(dtA, dim=1)
        
        # Dual formulation of SSD (State Space as Attention)
        S = cumsum_dtA.unsqueeze(2) - cumsum_dtA.unsqueeze(1)
        S = S.permute(0, 3, 1, 2) # (bsz, nheads, seqlen, seqlen)
        
        # Apply the same block-diagonal causal mask used by the Attention layer
        if mask is not None:
            S = S.masked_fill(~mask, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool))
            S = S.masked_fill(~causal_mask, float('-inf'))
            
        S_exp = torch.exp(S)
        
        CB = torch.einsum('bld, bmd -> blm', C, B).unsqueeze(1)
        attn = S_exp * CB
        
        X = x_in_reshaped.transpose(1, 2)
        dt_expanded = dt.transpose(1, 2).unsqueeze(-1)
        X_scaled = X * dt_expanded # scale input by discrete step size
        
        out = torch.einsum('bhlm, bhmd -> bhld', attn, X_scaled)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.d_inner)
        
        # Residual branch
        D_expanded = self.D.view(1, 1, self.nheads, 1).expand(-1, -1, -1, self.head_dim).reshape(1, 1, self.d_inner)
        out = out + x_in * D_expanded
        
        # Gating
        out = out * F.silu(z)
        out = self.out_proj(out)
        return out

class SwiGLUMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        
        # Parse hybrid pattern
        m_count, a_count = map(int, args.hybrid_pattern.split(":"))
        period = m_count + a_count
        is_attn = (layer_id % period) >= m_count
        
        if is_attn:
            self.attention = Attention(args)
        else:
            self.attention = Mamba2Block(args)
            
        self.feed_forward = SwiGLUMLP(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(l, params) for l in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        
        self.tok_embeddings.weight = self.output.weight

        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        
        for layer in self.layers:
            if isinstance(layer.attention, Attention):
                layer.attention.wo.SCALE_INIT = True
            else:
                layer.attention.out_proj.SCALE_INIT = True
            layer.feed_forward.w2.SCALE_INIT = True
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std = (2.0 * self.params.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, mask=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        
        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h)
        return output
