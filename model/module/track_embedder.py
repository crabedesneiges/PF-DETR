import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from .positional_encoding import build_pos_embed_from_eta_phi

class TrackEmbedder(nn.Module):
    """
    Un module avancé pour créer des embeddings de traces en utilisant 
    à la fois les paramètres globaux et la trajectoire séquentielle.
    """
    def __init__(self, num_track_features, hidden_dim, mlp_dropout=0.0, lstm_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.logd0_idx = 0
        self.logz0_idx = 1
        self.logpt_idx = 2
        self.deta_idx = 3
        self.dphi_idx = 4

        # Les features de layer vont de l'indice 6 à 23 (6 layers * 3 coords = 18 features)
        self.layer_feats_start_idx = 6
        self.num_layers = 6
        self.num_coords = 2

        # Les features globales que nous allons utiliser directement
        # d0, z0, p_t
        num_global_features = 3
        
        # LSTM pour encoder la trajectoire séquentielle (x,y,z par couche)
        self.lstm_hidden_dim = hidden_dim // 2  # On alloue une partie du budget de dimension
        self.trajectory_encoder = nn.LSTM(
            input_size=self.num_coords,  # (x, y, z)
            hidden_size=self.lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,  # Très important pour la manipulation des tenseurs
            dropout=mlp_dropout if lstm_layers > 1 else 0.0
        )

        # MLP final pour combiner les features globales et l'embedding de trajectoire
        self.final_projection = MLP(
            input_dim=num_global_features + self.lstm_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=mlp_dropout
        )

    def forward(self, track_feats):
        """
        Prend en entrée le tenseur de features de trace et retourne les embeddings et pos_encodings.
        
        Args:
            track_feats (Tensor): Shape [bs, num_track_features, N_track]
        
        Returns:
            tuple(Tensor, Tensor): 
                - embeddings (Tensor): Shape [bs, N_track, hidden_dim]
                - pos_encoding (Tensor): Shape [bs, N_track, hidden_dim/2]
        """
        # Transposer pour avoir (bs, N_track, num_track_features) pour un accès facile
        track_feats = track_feats.transpose(1, 2)
        bs, n_track, _ = track_feats.shape

        # 1. Extraire les paramètres cinématiques pour la régression et le pos_encoding
        original_phi = track_feats[..., self.dphi_idx]
        original_eta = track_feats[..., self.deta_idx]
        original_log_pt = track_feats[..., self.logpt_idx]

        # 2. Préparer les données pour l'encodeur de trajectoire (LSTM)
        layer_feats_end = self.layer_feats_start_idx + (self.num_layers * self.num_coords)
        # Isoler les coordonnées des couches : [bs, N_track, 18]
        trajectory_data = track_feats[..., self.layer_feats_start_idx:layer_feats_end]
        
        # Reshape pour le LSTM : [bs, N_track, num_layers, num_coords]
        trajectory_sequence = trajectory_data.reshape(bs, n_track, self.num_layers, self.num_coords)
        
        # Le LSTM attend une séquence, on "aplatit" les dimensions batch et track
        lstm_input = trajectory_sequence.reshape(bs * n_track, self.num_layers, self.num_coords)
        
        # Passer dans le LSTM. On ne récupère que l'état caché final.
        # `h_n` a pour shape [num_lstm_layers, bs * n_track, lstm_hidden_dim]
        _, (h_n, _) = self.trajectory_encoder(lstm_input)
        
        # On prend la sortie de la dernière couche du LSTM
        # Shape: [bs * n_track, lstm_hidden_dim]
        trajectory_embedding = h_n[-1]
        
        # Remettre en forme : [bs, N_track, lstm_hidden_dim]
        trajectory_embedding = trajectory_embedding.view(bs, n_track, self.lstm_hidden_dim)

        # 3. Extraire les features globales
        logd0 = track_feats[..., self.logd0_idx].unsqueeze(-1)
        logz0 = track_feats[..., self.logz0_idx].unsqueeze(-1)
        logpt = track_feats[..., self.logpt_idx].unsqueeze(-1)

        global_features = torch.cat([logd0, logz0, logpt], dim=-1)

        # 4. Combiner les features et projeter
        combined_features = torch.cat([global_features, trajectory_embedding], dim=-1)
        
        final_embedding = self.final_projection(combined_features) # Shape: [bs, N_track, hidden_dim]

        # 5. Construire l'encodage positionnel à partir de eta/phi initiaux
        pos_encoding = build_pos_embed_from_eta_phi(
            deta=original_eta, dphi=original_phi, num_feats=self.hidden_dim // 4
        )

        return final_embedding, pos_encoding, original_log_pt, original_eta, original_phi