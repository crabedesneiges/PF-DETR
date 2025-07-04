# DETR model modifié avec raffinement des paramètres de trace pour les particules chargées
# CORRECTION: Utilisation appropriée du track_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr.transformer import build_transformer
from .module.mlp import MLP
from .module.multimodalfusion import MultiModalInteraction, AttentionModalityFusion
from .module.positional_encoding import build_pos_embed_from_eta_phi
from .module.track_embedder import TrackEmbedder
from .module.gnn_topo_embedder import CalorimeterGNNEmbedder

class DualHeadDETR(nn.Module):
    def __init__(self, 
                 hidden_dim=256, 
                 nheads=8,
                 num_queries_notrack=15,
                 num_classes_charged=3, 
                 num_classes_neutral=2, 
                 class_no_particle=1,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 transformer_dropout=0.1,
                 mlp_dropout=0.0,
                 output_p4_dim=3,
                 indence_matrix_prediction=False):
        super().__init__()
        self.PADDED_LEN_CELL = 1250
        self.PADDED_LEN_TOPO = 70
        self.PADDED_LEN_TRACK = 15
        self.hidden_dim = hidden_dim
        # Queries apprenables pour les particules neutres (sans trace)
        self.query_embed_notrack = nn.Embedding(num_queries_notrack, hidden_dim)
        self.num_queries_notrack = num_queries_notrack

        self.cell_input_proj = nn.Conv1d(8, hidden_dim, kernel_size=1)
        self.topo_input_proj = nn.Conv1d(8, hidden_dim, kernel_size=1)
        self.track_input_proj = nn.Conv1d(18, hidden_dim, kernel_size=1)

        self.track_embedder = TrackEmbedder(
            num_track_features=18, 
            hidden_dim=hidden_dim, 
            mlp_dropout=mlp_dropout
        )
        self.calo_embedder = CalorimeterGNNEmbedder(
            num_cell_features=8, num_topo_features=8, hidden_dim=hidden_dim
        )   
        self.cell_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.topo_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.track_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.cross_modal = MultiModalInteraction(hidden_dim)
        self.fusion = AttentionModalityFusion(hidden_dim, num_modalities=3)


        """self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=transformer_dropout
        )"""
        self.transformer = build_transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,  # Typiquement 4x la dimension cachée
            dropout=transformer_dropout
        )

        # Têtes de prédiction pour les particules chargées
        self.class_embed_track = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, num_classes_charged)
        )

        # Tête de raffinement pour les paramètres de trace
        self.track_refinement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, 3)  # delta_log_pt, delta_eta, delta_phi
        )
        
        # Scales apprenables pour contrôler l'amplitude des corrections
        self.refinement_scales = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        
        # Scale adaptatif basé sur les features
        self.adaptive_scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        # Têtes de prédiction pour les particules neutres
        self.class_embed_notrack = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, num_classes_neutral + class_no_particle)
        )

        self.bbox_embed_notrack = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, output_p4_dim)
        )

        # Tête de prédiction pour la matrice d'incidence (nouvelle fonctionnalité)
        if indence_matrix_prediction:
            total_input_elements = self.PADDED_LEN_TOPO + self.PADDED_LEN_TRACK
            self.indence_matrix_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                # La couche de sortie prédit une valeur pour chaque élément d'entrée
                nn.Linear(hidden_dim, total_input_elements) 
            )
        else:
            self.indence_matrix_head = None # Initialiser à None si non utilisé
        self.init_weights()

    def init_weights(self):
        # Initialiser les MLP (déjà fait dans MLP)
        # Initialiser les Linear dans les séquentiels autres que MLP
        modules_to_init = [
            self.class_embed_track,
            self.track_refinement_head,
            self.adaptive_scale_head,
            self.class_embed_notrack,
            self.bbox_embed_notrack,
            self.fusion.attention_weights  # AttentionModalityFusion
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Initialisation des paramètres type embeddings
        nn.init.normal_(self.cell_type_embed, mean=0, std=0.02)
        nn.init.normal_(self.topo_type_embed, mean=0, std=0.02)
        nn.init.normal_(self.track_type_embed, mean=0, std=0.02)


    def forward(self, x, track_mask=None, cell_mask=None, topo_mask=None, ground_truth=None):
        
        """Forward pass with optional boolean masks (True = real entry).

        Incoming tensors:
            cell_feats  : (bs, feat_dim_c, N_cell_in)
            topo_feats  : (bs, feat_dim_t, N_topo_in)
            track_feats : (bs, feat_dim_k, N_track_in)

        Each mask has shape (bs, N_{modality}_in) if provided. We pad *both* the
        feature tensors and the masks to the fixed lengths expected by the
        network so that downstream modules can rely on constant sequence
        lengths while still distinguishing real vs padded positions.
        """
        cell_feats, topo_feats, track_feats = x
        bs = cell_feats.size(0)

        # Paramètres de padding fixes
        PADDED_LEN_CELL = self.PADDED_LEN_CELL
        PADDED_LEN_TOPO = self.PADDED_LEN_TOPO
        PADDED_LEN_TRACK = self.PADDED_LEN_TRACK

        # ------------------------------------------------------------------
        # Helper to pad tensors and masks to a fixed length
        # ------------------------------------------------------------------
        def pad_feats(feats, max_len):
            feat_len = feats.size(2)
            pad = max_len - feat_len
            if pad > 0:
                feats = F.pad(feats, (0, pad), value=0)
            elif pad < 0:
                feats = feats[:, :, :max_len]
            return feats

        def pad_mask(mask, max_len):
            if mask is None:
                # If no mask provided, consider everything valid up to original length.
                return torch.ones(bs, max_len, dtype=torch.bool, device=track_feats.device)
            pad = max_len - mask.size(1)
            if pad > 0:
                mask = F.pad(mask, (0, pad), value=False)
            elif pad < 0:
                mask = mask[:, :max_len]
            return mask

        # Pad features
        cell_feats = pad_feats(cell_feats, PADDED_LEN_CELL)
        topo_feats = pad_feats(topo_feats, PADDED_LEN_TOPO)
        track_feats = pad_feats(track_feats, PADDED_LEN_TRACK)
        
        # Pad / build masks
        cell_mask = pad_mask(cell_mask, PADDED_LEN_CELL)
        topo_mask = pad_mask(topo_mask, PADDED_LEN_TOPO)
        track_mask = pad_mask(track_mask, PADDED_LEN_TRACK)

        # Embedding des features
        tr, pos_tr, original_log_pt, original_eta, original_phi = self.track_embedder(track_feats)
        c,t, pos_c, pos_t = self.calo_embedder(cell_feats, topo_feats, cell_mask=cell_mask, topo_mask=topo_mask)

        tr += self.track_type_embed
        c += self.cell_type_embed
        t += self.topo_type_embed


        # Passage du track_mask aux interactions cross-modales 
        c, t, tr = self.cross_modal(c, t, tr, cell_mask=cell_mask, topo_mask=topo_mask, track_mask=track_mask)
        if torch.isnan(c).any() or torch.isnan(t).any() or torch.isnan(tr).any():
            raise ValueError("NaN detected in cross-modal features")

        encoder_input = torch.cat([c, t, tr], dim=1)
        
        encoder_mask = torch.cat([cell_mask, topo_mask, track_mask], dim=1)

        # Inverser le mask pour key_padding_mask (True = padded, False = valide)
        encoder_padding_mask = ~encoder_mask
        pos_enc = torch.cat([pos_c, pos_t, pos_tr], dim=1)
        
        # Encoder avec le mask
        encoder_input_t = encoder_input.transpose(0, 1)
        pos_enc_t = pos_enc.transpose(0, 1)

        memory = self.transformer.encoder(encoder_input_t, src_key_padding_mask=encoder_padding_mask, pos=pos_enc_t)
        memory = memory.transpose(0, 1)  # Revenir à (bs, seq_len, hidden_dim)

        # Extraction des queries pour les traces depuis l'encodeur
        start_idx = PADDED_LEN_CELL + PADDED_LEN_TOPO
        end_idx = start_idx + PADDED_LEN_TRACK
        track_queries = memory[:, start_idx:end_idx, :]

        # Queries pour les particules neutres
        notrack_queries = self.query_embed_notrack.weight.unsqueeze(0).repeat(bs, 1, 1)

        # Combinaison des queries
        combined_queries = torch.cat([track_queries, notrack_queries], dim=1)
        
        combined_queries_t = combined_queries.transpose(0, 1)  # [Lq, B, D]

        notrack_mask_t = torch.zeros(
            (bs, self.num_queries_notrack), 
            dtype=torch.bool, 
            device=track_mask.device
        )
        

        tgt_key_padding_mask = torch.cat([track_mask, notrack_mask_t], dim=1)
        # Décodage avec le transformer
        decoded_output = self.transformer.decoder(
            combined_queries_t, 
            memory.transpose(0, 1), 
            memory_key_padding_mask=encoder_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            pos=pos_enc_t
        )

        decoded_output = decoded_output[0, :, :, :].transpose(0, 1)  # [B, Lq, D]
        # Séparation des sorties
        decoded_track = decoded_output[:, :PADDED_LEN_TRACK, :]
        decoded_notrack = decoded_output[:, PADDED_LEN_TRACK:, :]
        # Prédictions pour les particules chargées

        pred_logits_track = self.class_embed_track(decoded_track)
        
        # Masquer les prédictions des tracks non valides
        pred_logits_track = pred_logits_track.masked_fill(~track_mask.unsqueeze(-1), -1e4)
        
        # Raffinement des paramètres de trace
        track_refinements = self.track_refinement_head(decoded_track)
        scaled_refinements = track_refinements * self.refinement_scales.unsqueeze(0).unsqueeze(0)
        
        delta_log_pt = scaled_refinements[:, :, 0]
        delta_eta = scaled_refinements[:, :, 1]
        delta_phi = scaled_refinements[:, :, 2]

        # CORRECTION: Masquer les raffinements des tracks non valides
        delta_eta = delta_eta.masked_fill(~track_mask, 0.0)
        delta_phi = delta_phi.masked_fill(~track_mask, 0.0)
        delta_log_pt = delta_log_pt.masked_fill(~track_mask, 0.0)
        
        # Application des raffinements aux paramètres originaux
        refined_eta = original_eta + delta_eta
        refined_phi = original_phi + delta_phi
        refined_log_pt = original_log_pt + delta_log_pt
        
        # Construction de la sortie raffinée
        pred_boxes_track = torch.stack([refined_log_pt, refined_eta, refined_phi], dim=-1)

        # Prédictions pour les particules neutres (inchangées)
        pred_logits_notrack = self.class_embed_notrack(decoded_notrack)
        pred_boxes_notrack = self.bbox_embed_notrack(decoded_notrack)

        output = {
            "pred_logits_track": pred_logits_track,
            "pred_boxes_track": pred_boxes_track,
            "pred_logits_notrack": pred_logits_notrack,
            "pred_boxes_notrack": pred_boxes_notrack,
            "track_mask": track_mask,
            "ground_truth": ground_truth,
            "original_track_params": {
                "eta": original_eta,
                "phi": original_phi,
                "log_pt": original_log_pt
            },
            "track_refinements": {
                "delta_eta": delta_eta,
                "delta_phi": delta_phi,
                "delta_log_pt": delta_log_pt,
                "refinement_scales": self.refinement_scales.detach().cpu()
            }
        }
        # Calcul et ajout de la matrice d'incidence si activée
        if self.indence_matrix_head is not None:
            # Appliquer la tête de matrice d'incidence à la sortie combinée du décodeur
            # (bs, L_decoder, hidden_dim) -> (bs, L_decoder, total_input_elements)
            indence_matrix_logits = self.indence_matrix_head(decoded_output)
            indence_matrix_mask = torch.cat([ topo_mask, track_mask], dim=1)
            # Masquer les logits pour les éléments d'entrée paddés avant le softmax
            # indence_matrix_mask a la forme (bs, total_input_elements)
            # Unsqueeze pour correspondre à la dimension du L_decoder
            # ~indence_matrix_mask inverse le masque (True pour padding, False pour réel)
            indence_matrix_logits = indence_matrix_logits.masked_fill(
                ~indence_matrix_mask.unsqueeze(1),  # Masque à appliquer sur la dernière dimension
                -1e4 # Valeur très petite pour que le softmax donne ~0
            )
            
            # Appliquer softmax sur la dernière dimension pour obtenir une distribution de probabilité
            # pour chaque particule sur les éléments d'entrée.
            #indence_matrix = F.softmax(indence_matrix_logits, dim=-1)
            output["indence_matrix"] = indence_matrix_logits
            output["indence_matrix_mask"] = indence_matrix_mask

        return output