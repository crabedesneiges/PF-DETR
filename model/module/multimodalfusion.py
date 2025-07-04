import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        # CORRECTION: Utilisation du mask dans l'attention
        attn_output, _ = self.multihead_attn(
            query, key_value, key_value, 
            key_padding_mask=key_padding_mask
        )
        return self.norm(query + self.dropout(attn_output))
    

class AttentionModalityFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities=3):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, num_modalities)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_features, masks=None):
        min_len = min([f.size(1) for f in modality_features])
        truncated_feats = [f[:, :min_len, :] for f in modality_features]
        stacked = torch.stack(truncated_feats, dim=2)
        bs, seq_len, n_modalities, hidden_dim = stacked.shape
        stacked_flat = stacked.reshape(-1, hidden_dim)
        attention_scores = self.attention_weights(stacked_flat)
        attention_scores = attention_scores.view(bs, seq_len, n_modalities, n_modalities)
        attention_scores = attention_scores.diagonal(dim1=2, dim2=3)
        
        # CORRECTION: Application des masks si fournis
        if masks is not None and len(masks) >= 3:  # masks pour [cell, topo, track]
            track_mask = masks[2]  # mask pour les tracks
            cell_mask = masks[0]  # mask pour les cells
            topo_mask = masks[1]  # mask pour les topologies
            if track_mask is not None:
                # Étendre le mask pour correspondre aux dimensions
                extended_mask = track_mask.unsqueeze(-1).expand(-1, -1, n_modalities)
                attention_scores = attention_scores.masked_fill(~extended_mask, -1e4)

            if cell_mask is not None:
                # Étendre le mask pour correspondre aux dimensions
                extended_mask = cell_mask.unsqueeze(-1).expand(-1, -1, n_modalities)
                attention_scores = attention_scores.masked_fill(~extended_mask, -1e4)

            if topo_mask is not None:
                # Étendre le mask pour correspondre aux dimensions
                extended_mask = topo_mask.unsqueeze(-1).expand(-1, -1, n_modalities)
                attention_scores = attention_scores.masked_fill(~extended_mask, -1e4)
        
        attention_weights = F.softmax(attention_scores, dim=2)
        fused = (stacked * attention_weights.unsqueeze(-1)).sum(dim=2)
        return self.layer_norm(fused)
    
class MultiModalInteraction(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cell_to_track = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.cell_to_topo = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.track_to_cell = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.track_to_topo = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.topo_to_cell = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.topo_to_track = CrossModalAttention(hidden_dim, num_heads, dropout)

    def forward(self, cells, topos, tracks, cell_mask=None, topo_mask=None, track_mask=None):
        # CORRECTION: Utilisation du track_mask dans les interactions cross-modales
        # Inverser le mask pour key_padding_mask (True = padded, False = valide)
        track_padding_mask = ~track_mask if track_mask is not None else None
        cell_padding_mask = ~cell_mask if cell_mask is not None else None
        topo_padding_mask = ~topo_mask if topo_mask is not None else None
        
        cells = self.cell_to_track(cells, tracks, key_padding_mask=track_padding_mask)
        cells = self.cell_to_topo(cells, topos, key_padding_mask=topo_padding_mask)
        tracks = self.track_to_cell(tracks, cells, key_padding_mask=cell_padding_mask)
        tracks = self.track_to_topo(tracks, topos, key_padding_mask=topo_padding_mask)
        topos = self.topo_to_cell(topos, cells, key_padding_mask=cell_padding_mask)
        topos = self.topo_to_track(topos, tracks, key_padding_mask=track_padding_mask)
        return cells, topos, tracks