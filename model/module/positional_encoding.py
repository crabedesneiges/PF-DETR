import torch
from torch import nn

def build_pos_embed_from_eta_phi(deta, dphi, num_feats=64):
    import math
    """
    Create 2D sine-cosine positional embeddings from η and φ.
    
    Args:
        deta (Tensor): [B, N]
        dphi (Tensor): [B, N]
        num_feats (int): Half of the final dimension size. Default 64 → output dim = 128
    Returns:
        pos (Tensor): [B, N, 2 * num_feats]
    """
    scale = 2 * math.pi

    # [B, N, 1]
    eta = deta.unsqueeze(-1) * scale
    phi = dphi.unsqueeze(-1) * scale

    # [num_feats]
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=deta.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_feats)

    # [B, N, num_feats]
    pos_eta = eta / dim_t
    pos_phi = phi / dim_t

    # Apply sin to even indices, cos to odd
    pos_eta = torch.stack((pos_eta.sin(), pos_eta.cos()), dim=3).flatten(2)  # [B, N, 2*num_feats]
    pos_phi = torch.stack((pos_phi.sin(), pos_phi.cos()), dim=3).flatten(2)

    # Concatenate eta and phi encodings
    pos = torch.cat((pos_eta, pos_phi), dim=2)  # [B, N, 4*num_feats]

    return pos