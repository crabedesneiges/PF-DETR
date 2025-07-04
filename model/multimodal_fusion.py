import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block that allows one modality to attend to another
    """
    def __init__(self, hidden_dim: int, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Cross-attention: query attends to key_value
        attn_out, _ = self.cross_attention(
            query, key_value, key_value, 
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


class CoAttentionLayer(nn.Module):
    """
    Co-attention layer inspired by ViLBERT - alternating attention between modalities
    """
    def __init__(self, hidden_dim: int, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Self-attention for each modality
        self.self_attn_a = nn.MultiheadAttention(hidden_dim, nheads, dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(hidden_dim, nheads, dropout, batch_first=True)
        
        # Cross-attention between modalities
        self.cross_attn_a2b = nn.MultiheadAttention(hidden_dim, nheads, dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(hidden_dim, nheads, dropout, batch_first=True)
        
        # Layer norms
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.norm_a3 = nn.LayerNorm(hidden_dim)
        
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)
        self.norm_b3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor,
                mask_a: Optional[torch.Tensor] = None,
                mask_b: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Self-attention for modality A
        attn_a, _ = self.self_attn_a(feat_a, feat_a, feat_a, key_padding_mask=mask_a)
        feat_a = self.norm_a1(feat_a + attn_a)
        
        # Self-attention for modality B
        attn_b, _ = self.self_attn_b(feat_b, feat_b, feat_b, key_padding_mask=mask_b)
        feat_b = self.norm_b1(feat_b + attn_b)
        
        # Cross-attention: A attends to B
        cross_a2b, _ = self.cross_attn_a2b(feat_a, feat_b, feat_b, key_padding_mask=mask_b)
        feat_a = self.norm_a2(feat_a + cross_a2b)
        
        # Cross-attention: B attends to A
        cross_b2a, _ = self.cross_attn_b2a(feat_b, feat_a, feat_a, key_padding_mask=mask_a)
        feat_b = self.norm_b2(feat_b + cross_b2a)
        
        # Feed-forward networks
        ffn_a_out = self.ffn_a(feat_a)
        feat_a = self.norm_a3(feat_a + ffn_a_out)
        
        ffn_b_out = self.ffn_b(feat_b)
        feat_b = self.norm_b3(feat_b + ffn_b_out)
        
        return feat_a, feat_b


class BilinearAttentionFusion(nn.Module):
    """
    Bilinear attention mechanism for fusing two modalities
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        # feat_a: [bs, len_a, hidden_dim]
        # feat_b: [bs, len_b, hidden_dim]
        
        bs, len_a, hidden_dim = feat_a.shape
        len_b = feat_b.shape[1]
        
        # Expand for pairwise computation
        feat_a_exp = feat_a.unsqueeze(2).expand(-1, -1, len_b, -1)  # [bs, len_a, len_b, hidden_dim]
        feat_b_exp = feat_b.unsqueeze(1).expand(-1, len_a, -1, -1)  # [bs, len_a, len_b, hidden_dim]
        
        # Bilinear interaction
        interaction = self.bilinear(feat_a_exp, feat_b_exp)  # [bs, len_a, len_b, hidden_dim]
        
        # Pool over one dimension (you can also use attention pooling)
        fused = torch.mean(interaction, dim=2)  # [bs, len_a, hidden_dim]
        
        fused = self.dropout(fused)
        fused = self.norm(fused + feat_a)  # Residual connection
        
        return fused


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with separate encoders + cross-attention fusion
    """
    def __init__(self, hidden_dim: int, nheads: int = 8, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Separate encoders for each modality
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            batch_first=True,
            dropout=dropout
        )
        
        self.topo_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.track_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, nheads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, topo_feats: torch.Tensor, track_feats: torch.Tensor,
                topo_mask: Optional[torch.Tensor] = None,
                track_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Separate encoding
        topo_encoded = self.topo_encoder(topo_feats, src_key_padding_mask=topo_mask)
        track_encoded = self.track_encoder(track_feats, src_key_padding_mask=track_mask)
        
        # Cross-attention between modalities
        topo_cross = topo_encoded
        track_cross = track_encoded
        
        for cross_layer in self.cross_attention_layers:
            # Topo attends to track
            topo_cross = cross_layer(topo_cross, track_encoded, track_mask)
            # Track attends to topo
            track_cross = cross_layer(track_cross, topo_encoded, topo_mask)
        
        # Concatenate and fuse
        # Global pooling for each modality
        topo_pooled = torch.mean(topo_cross, dim=1)  # [bs, hidden_dim]
        track_pooled = torch.mean(track_cross, dim=1)  # [bs, hidden_dim]
        
        # Concatenate and project
        combined = torch.cat([topo_pooled, track_pooled], dim=-1)  # [bs, 2*hidden_dim]
        fused = self.fusion_layer(combined)  # [bs, hidden_dim]
        
        return fused.unsqueeze(1).expand(-1, max(topo_feats.size(1), track_feats.size(1)), -1)


class GatedMultiModalFusion(nn.Module):
    """
    Gated fusion mechanism that learns how to combine modalities
    """
    def __init__(self, hidden_dim: int, num_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Modality-specific transformations
        self.modality_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_modalities)
        ])
        
        self.final_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, *modality_features) -> torch.Tensor:
        """
        Args:
            *modality_features: Variable number of tensors [bs, seq_len, hidden_dim]
        """
        assert len(modality_features) == self.num_modalities
        
        bs, seq_len, hidden_dim = modality_features[0].shape
        
        # Transform each modality
        transformed_features = []
        for i, feat in enumerate(modality_features):
            transformed = self.modality_transforms[i](feat)
            transformed_features.append(transformed)
        
        # Compute gates
        concat_features = torch.cat(modality_features, dim=-1)  # [bs, seq_len, hidden_dim * num_mod]
        gates = self.gate_network(concat_features)  # [bs, seq_len, num_modalities]
        
        # Weighted combination
        fused = torch.zeros_like(modality_features[0])
        for i, feat in enumerate(transformed_features):
            weight = gates[:, :, i:i+1]  # [bs, seq_len, 1]
            fused += weight * feat
        
        fused = self.final_transform(fused)
        return fused