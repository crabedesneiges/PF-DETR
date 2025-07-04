"""
PyTorch LightningModule wrapping the DETR transformer model.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import mlflow
import os
import json
import time
import platform
import subprocess
import psutil

from model.model_DETR_tracker_with_embedder import DualHeadDETR as DETRModelTracker
from model.hungarian_matcher import HungarianMatcher


class DETRLightningModule(pl.LightningModule):
    """
    LightningModule for DETR model with classification and bbox losses. 
    """
    def load_state_dict(self, state_dict, strict: bool = False):
        # Allow loading checkpoints with missing keys (for backward compatibility)
        return super().load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint):
        # Only print a warning if missing; registration happens in __init__
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing = []
        if "logpt_mean" not in state_dict:
            missing.append("logpt_mean")
        if "logpt_std" not in state_dict:
            missing.append("logpt_std")
        if missing:
            print(f"[INFO] The following buffers were missing in checkpoint and the model's default values will be used: {missing}")

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Always register buffers at init so PL can fallback to these if missing in checkpoint
        norm_params_path = config.get("data", {}).get("norm_params_path", "data/normalization/params.json")

        self.register_buffer("particle_logpt_mean", torch.tensor(8.50248866536399))
        self.register_buffer("particle_logpt_std", torch.tensor(1.1002988461441665))
        # Enregistrer les hyperparamètres dans MLflow
        # Permet de filtrer/trier les expériences facilement
        self.save_hyperparameters(config)
        
        # Stockage de métriques additionnelles pour MLflow
        self.validation_metrics = {}
        # instantiate DETR model from config
        mcfg = self.config["model"]
        # Option to use deformable DETR
        self.dualDETR = mcfg.get("use_DETR_V3", False)
        self.track_and_charged_cardinality = mcfg.get("track_and_charged_cardinality", False)
        self.use_indence_matrix_prediction = mcfg.get("indence_matrix_prediction", False)
        if mcfg.get("use_DETR_V3", False):
            self.model = DETRModelTracker(
                nheads=mcfg.get("nheads", 8),
                hidden_dim=mcfg.get("hidden_dim", 256), 
                num_queries_notrack=mcfg.get("num_queries_notrack", 15),
                num_encoder_layers=mcfg.get("num_encoder_layers", 6),
                num_decoder_layers=mcfg.get("num_decoder_layers", 6),
                transformer_dropout=mcfg.get("transformer_dropout", 0.1),
                mlp_dropout=mcfg.get("mlp_dropout", 0.0),
                indence_matrix_prediction=mcfg.get("indence_matrix_prediction", False),
                )
        else:
            raise ValueError("Invalid model configuration")
        # default loss functions
        self.padding_idx = 5  # index for padding (fixed: 6 classes)
        # --- Ajout coefficients de pondération ---
        training_cfg = self.config.get("training", {})
        self.loss_coeffs = {
            "cls": training_cfg.get("cls_loss_coeff", 1.0),
            "bbox": training_cfg.get("bbox_loss_coeff", 1.0),
            "conservation": training_cfg.get("conservation_loss_coeff", 0.0), # New conservation loss coefficient
            "cardinality": training_cfg.get("cardinality_loss_coeff", 0.0), # New cardinality loss coefficient
            "indence_matrix": training_cfg.get("indence_matrix_loss_coeff", 0.0), # New indence matrix loss coefficient
        }

        # Load normalization parameters for pT un-normalization
        # Assuming params.json is in a fixed relative path or configured elsewhere if needed.
        # For simplicity, using a common path. Adjust if your setup differs.
        
        norm_params_path = config.get("data", {}).get("norm_params_path", "data/normalization/params.json")
        if self.loss_coeffs["conservation"] > 0.0:
            try:
                with open(norm_params_path, 'r') as f:
                    norm_params = json.load(f)
                particle_logpt_mean_val = norm_params["logpt"]["mean_"][0]
                particle_logpt_std_val = norm_params["logpt"]["scale_"][0]
            except FileNotFoundError:
                print(f"WARNING: Normalization parameters file not found at {norm_params_path}. Using default values for pT un-normalization (mean=0, std=1).")
                particle_logpt_mean_val = 8.50248866536399
                particle_logpt_std_val = 1.1002988461441665
            except KeyError:
                print(f"WARNING: 'logpt' not found in {norm_params_path}. Using default values for pT un-normalization (mean=0, std=1).")
                particle_logpt_mean_val = 8.50248866536399
                particle_logpt_std_val = 1.1002988461441665

            self.particle_logpt_mean = torch.tensor(particle_logpt_mean_val, dtype=torch.float32)
            self.particle_logpt_std = torch.tensor(particle_logpt_std_val, dtype=torch.float32)
        self.n_epoch_warmup = training_cfg.get("n_epoch_warmup", 0)
        # --- NEW: Optional phase mode ---
        self.loss_phase_mode = training_cfg.get("loss_phase_mode", False)
        self.n_epoch_phase1 = training_cfg.get("n_epoch_phase1", 0)
        self.n_epoch_phase2 = training_cfg.get("n_epoch_phase2", 0)
        # Phase 3 is the remainder
        self.matcher = HungarianMatcher(
            cls_cost=self.loss_coeffs["cls"],
            bbox_cost=self.loss_coeffs["bbox"],
            padding_idx=self.padding_idx,
        )

        # --- Torchmetrics: per-class accuracy (spécificité/FPR supprimées) ---
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=6, average=None)
        self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=6, average=None)
        # Garder ces attributs pour compatibilité avec le code existant mais ne plus les utiliser
        self.train_specificity = torchmetrics.classification.MulticlassSpecificity(num_classes=6, average=None)
        self.val_specificity = torchmetrics.classification.MulticlassSpecificity(num_classes=6, average=None)


    def forward(self, x):
        # During training/validation/test, x is a dict; during prediction, it may be a tuple or dict
        if isinstance(x, dict) and 'input' in x:

            cell_mask = x.get('cell_mask', None)
            topo_mask = x.get('topo_mask', None)
            track_mask = x.get('track_mask', None)
            return self.model(x['input'], ground_truth=x["target"], cell_mask=cell_mask, topo_mask=topo_mask, track_mask=track_mask)
        else:
            print("WARNING : BAD FORMAT no Mask pass to the model --> BAD RESULT")
            return self.model(x, ground_truth=x["target"])

    

    def all_gather_total(self, value: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        """Gather scalar values from all processes and sum."""
        if not dist.is_available() or not dist.is_initialized():
            return value
        gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        return sum(gathered)

    def _step_with_matching(self, batch, log_name, prog_bar=False):
        
        # --- 1. Inférence du modèle ---
        x = batch['input']
        
        cell_mask = batch.get('cell_mask', None)
        topo_mask = batch.get('topo_mask', None)
        track_mask = batch.get('track_mask', None)
        outputs = self.model(x, cell_mask=cell_mask, topo_mask=topo_mask, track_mask=track_mask)

        pred_logits_track = outputs["pred_logits_track"]
        pred_boxes_track = outputs["pred_boxes_track"]
        pred_logits_notrack = outputs["pred_logits_notrack"]
        pred_boxes_notrack = outputs["pred_boxes_notrack"]
        if self.use_indence_matrix_prediction:
            pred_incidence_matrix = outputs.get("indence_matrix", None)
            incidence_matrix_mask = outputs.get("indence_matrix_mask", None)
            
        tgt_incidence_matrix = batch["target"].get("incidence_matrix", None)
        ghost_particle_idx = batch["target"].get("ghost_particle_idx", None)
        missing_particle_idx = batch["target"].get("missing_particle_idx", None)


        batch_size, num_queries, _ = pred_logits_notrack.shape

        # --- 2. Préparation des cibles (Ground Truth) ---
        targets = batch["target"]
        tgt_labels = targets["labels"]
        tgt_is_track = targets["is_track"]
        tgt_boxes = targets["boxes"]

        # Traitement conditionnel des labels si nécessaire
        if hasattr(self, 'track_and_charged_cardinality') and self.track_and_charged_cardinality:
            condition1_mask = (tgt_labels == 0) & (tgt_is_track == 0)
            condition2_mask = (tgt_labels == 1) & (tgt_is_track == 0)
            tgt_labels[condition1_mask] = 3  # charged_hadron -> neutral_hadron
            tgt_labels[condition2_mask] = 4  # electron -> photons

        # Initialisation du matcher unique
        matcher = HungarianMatcher(
            cls_cost=self.loss_coeffs["cls"], 
            bbox_cost=self.loss_coeffs["bbox"],
            
        )

        # --- 3. Séparation et matching par batch ---
        matched_track_pred_boxes = []
        matched_track_pred_logit = []
        matched_track_tgt_boxes = []
        matched_track_tgt_label = []
        matched_notrack_pred_boxes = []
        matched_notrack_pred_logit = []
        matched_notrack_tgt_boxes = []
        matched_notrack_tgt_label = []

        loss_indence = torch.zeros(batch_size).to(pred_logits_notrack.device)
        

        for b in range(batch_size):
            # Séparation des cibles pour ce batch
            is_track_mask = tgt_is_track[b] == 1
            is_notrack_mask = tgt_is_track[b] == 0
            
            # Cibles chargées (avec tracks)
            tgt_labels_charged = tgt_labels[b][is_track_mask]
            tgt_boxes_charged = tgt_boxes[b][is_track_mask]
            
            # Cibles non-chargées (sans tracks)
            tgt_labels_neutral = tgt_labels[b][is_notrack_mask]
            tgt_boxes_neutral = tgt_boxes[b][is_notrack_mask]
            
            # Filtrage des vrais objets (non-padding)
            
            real_neutral_mask = (tgt_labels_neutral != self.padding_idx)
            
            
            
            tgt_labels_real_neutral = tgt_labels_neutral[real_neutral_mask]
            tgt_boxes_real_neutral = tgt_boxes_neutral[real_neutral_mask]

            # --- Matching pour les particules chargées ---
            # Pas de padding pour les prédictions chargées (padding_idx = None)
            matcher.padding_idx = None  # Pas de classe "no object" pour les chargées
            
            
            num_tgt = tgt_boxes_charged.shape[0]
            truncated_pred_boxes_track = pred_boxes_track[b][:num_tgt, :]
            truncated_pred_logits_track = pred_logits_track[b][:num_tgt, :]
            
            
            indices_batch_track = matcher(
                truncated_pred_logits_track.unsqueeze(0), 
                truncated_pred_boxes_track.unsqueeze(0),
                [tgt_labels_charged], 
                [tgt_boxes_charged]
            )
            src_idx, tgt_idx = indices_batch_track[0]
            aligned_preds_boxes_track = torch.zeros_like(truncated_pred_boxes_track)
            aligned_preds_logit_track = torch.zeros_like(truncated_pred_logits_track)
            aligned_preds_boxes_track[tgt_idx] = truncated_pred_boxes_track[src_idx]
            aligned_preds_logit_track[tgt_idx] = truncated_pred_logits_track[src_idx]


            matched_track_pred_boxes.append(aligned_preds_boxes_track)
            matched_track_pred_logit.append(aligned_preds_logit_track)
            matched_track_tgt_boxes.append(tgt_boxes_charged)
            matched_track_tgt_label.append(tgt_labels_charged)

            # --- NOUVEAU: Alignement pour la matrice d'incidence (particules chargées) ---
            if pred_incidence_matrix is not None and tgt_incidence_matrix is not None:
                num_track_queries = 15  # Nombre de queries track
                # Prédictions d'incidence pour les queries chargées appariées
                event_incidence_matrix_charged = pred_incidence_matrix[b][:,incidence_matrix_mask[b]][:num_track_queries]
                pred_inc_charged = event_incidence_matrix_charged[src_idx]
                gt_inc_charged = tgt_incidence_matrix[b][is_track_mask][tgt_idx]

            # --- Matching pour les particules non-chargées ---
            # Classe 2 = "None Particle" pour le padding
            matcher.padding_idx = 2
            
            # Ajustement des labels neutres si nécessaire (soustraire 2 comme dans votre code original)
            tgt_labels_real_neutral_adjusted = tgt_labels_real_neutral.clone()
            if len(tgt_labels_real_neutral_adjusted) > 0:
                tgt_labels_real_neutral_adjusted -= 3  # Ajustement selon votre logique
                # S'assurer que les labels ajustés restent valides (>= 0)
                tgt_labels_real_neutral_adjusted = torch.clamp(tgt_labels_real_neutral_adjusted, min=0)
            

            indices_batch_notrack = matcher(
                pred_logits_notrack[b].unsqueeze(0), 
                pred_boxes_notrack[b].unsqueeze(0),
                [tgt_labels_real_neutral_adjusted], 
                [tgt_boxes_real_neutral]
            )

            # Stockage des indices
            
            src_idx, tgt_idx = indices_batch_notrack[0]

            aligned_tgt_boxes_notrack = torch.zeros_like(pred_boxes_notrack[b])
            aligned_tgt_label_notrack = torch.full_like(pred_logits_notrack[b, :, 0], fill_value=2, dtype=torch.int64)

            aligned_tgt_boxes_notrack[src_idx] = tgt_boxes_real_neutral[tgt_idx].to(aligned_tgt_boxes_notrack.dtype)
            aligned_tgt_label_notrack[src_idx] = tgt_labels_real_neutral_adjusted[tgt_idx]

            matched_notrack_pred_boxes.append(pred_boxes_notrack[b])
            matched_notrack_pred_logit.append(pred_logits_notrack[b])
            matched_notrack_tgt_boxes.append(aligned_tgt_boxes_notrack)
            matched_notrack_tgt_label.append(aligned_tgt_label_notrack) 
                      
            
            if pred_incidence_matrix is not None and tgt_incidence_matrix is not None:
                num_track_queries = 15  # Nombre de queries track
                # Prédictions d'incidence pour les queries chargées appariées
                event_incidence_matrix_neutral = pred_incidence_matrix[b][:,incidence_matrix_mask[b]][num_track_queries:]
                pred_inc_neutral = event_incidence_matrix_neutral[src_idx]
                gt_inc_neutral = tgt_incidence_matrix[b][is_notrack_mask][tgt_idx]
    
                #CAT
                pred_inc_true = torch.cat([pred_inc_charged, pred_inc_neutral], dim=0)
                gt_inc_true = torch.cat([gt_inc_charged, gt_inc_neutral], dim=0)

                # Vérification des formes
                #print(f"pred_inc_true shape: {pred_inc_true.shape}, gt_inc_true shape: {gt_inc_true.shape}")
                
                #Calcul de la loss
                gt_inc_T = gt_inc_true.T
                pred_inc_T = pred_inc_true.T
                loss_fn_kl = nn.KLDivLoss(reduction='sum')
                pred_log_probs = F.log_softmax(pred_inc_T, dim=0)
                if pred_log_probs.shape == gt_inc_T.shape:
                    loss_indence[b] = loss_fn_kl(pred_log_probs, gt_inc_T.float())
                else:
                    print(f"Shape mismatch: pred_log_probs {pred_log_probs.shape}, gt_inc_T {gt_inc_T.shape}")
                    loss_indence[b] = torch.tensor(0.0, device=pred_logits_notrack.device)

                #print(f"loss_indence[{b}]: {loss_indence[b].item()}")

        #Concat Res
        loss_cls_charged = torch.tensor(0.0, device=pred_logits_track.device)
        pred_boxes_track_cat = torch.cat(matched_track_pred_boxes, dim=0)
        pred_logits_track_cat = torch.cat(matched_track_pred_logit, dim=0)
        tgt_boxes_track_cat = torch.cat(matched_track_tgt_boxes, dim=0)
        tgt_labels_track_cat = torch.cat(matched_track_tgt_label, dim=0)

        loss_cls_notrack = torch.tensor(0.0, device=pred_logits_notrack.device)
        pred_boxes_notrack_cat = torch.cat(matched_notrack_pred_boxes, dim=0)
        pred_logits_notrack_cat = torch.cat(matched_notrack_pred_logit, dim=0)
        tgt_boxes_notrack_cat = torch.cat(matched_notrack_tgt_boxes, dim=0)
        tgt_labels_notrack_cat = torch.cat(matched_notrack_tgt_label, dim=0)

        loss_ind = torch.sum(loss_indence, dim=0)

        #Loss cls pour les particules chargées
        charged_class_weights = torch.tensor(self.config['model'].get('charged_class_weights', [1.0]*3), device=pred_logits_track.device)

        #Loss cls pour les particules non-chargées

        # Pondération pour réduire l'impact de la classe "None Particle"
        neutral_class_weights = torch.tensor(self.config['model'].get('neutral_class_weights', [1.0]*3), device=pred_logits_notrack.device)
        neutral_class_weights[2] = self.config['model'].get('no_object_weight', 0.1)  # Classe "None Particle"

        

        #Loss bbox  pour les particules non-chargées
        mask_true_particle = tgt_labels_notrack_cat != 2
        loss_bbox_neutral = torch.tensor(0.0, device=pred_logits_notrack.device)
        


        # Count objects
        num_objects_charged = sum([len(t) for t in matched_track_tgt_label])
        num_objects_neutral = (tgt_labels_notrack_cat != 2).sum()
        num_objects = (num_objects_charged + num_objects_neutral).to(dtype=torch.float32, device=pred_logits_track.device)

        # --- Sync object counts across processes ---
        total_num_objects = self.all_gather_total(num_objects).clamp(min=1.0)  # prevent divide-by-zero

        # Compute summed losses
        class_loss_type = self.config['training'].get('cls_loss_type', 'cross_entropy')  # 'cross_entropy' or 'focal'
        focal_gamma = self.config['training'].get('focal_gamma', 2.0)
        focal_alpha_charged = self.config['training'].get('focal_alpha_charged', None)
        focal_alpha_neutral = self.config['training'].get('focal_alpha_neutral', None)

        def focal_loss(logits, targets, gamma=2.0, alpha=None):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            if alpha is not None:
                if isinstance(alpha, (list, tuple)):
                    alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
                    at = alpha[targets]
                elif isinstance(alpha, torch.Tensor):
                    at = alpha[targets]
                elif isinstance(alpha, float):
                    at = torch.full_like(targets, fill_value=alpha, dtype=logits.dtype, device=logits.device)
                else:
                    raise TypeError("alpha must be a float, list, tuple, or torch.Tensor")
                ce_loss = at * ce_loss
            loss = ((1 - pt) ** gamma) * ce_loss
            return loss.sum()

        if class_loss_type == 'focal':
            loss_cls_charged = focal_loss(
            pred_logits_track_cat, tgt_labels_track_cat,
            gamma=focal_gamma,
            alpha=focal_alpha_charged
            ) if pred_logits_track_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_track.device)

            loss_cls_neutral = focal_loss(
            pred_logits_notrack_cat, tgt_labels_notrack_cat,
            gamma=focal_gamma,
            alpha=focal_alpha_neutral
            ) if pred_logits_notrack_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_notrack.device)
        else:
            loss_cls_charged = F.cross_entropy(
            pred_logits_track_cat, tgt_labels_track_cat,
            reduction='sum', weight=charged_class_weights
            ) if pred_logits_track_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_track.device)

            loss_cls_neutral = F.cross_entropy(
            pred_logits_notrack_cat, tgt_labels_notrack_cat,
            reduction='sum', weight=neutral_class_weights
            ) if pred_logits_notrack_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_notrack.device)

        # --- BBox loss: support for Huber (Smooth L1) or L1 ---
        bbox_loss_type = self.config['training'].get('bbox_loss_type', 'l1')  # 'l1' or 'huber'
        huber_beta = self.config['training'].get('huber_beta', 1.0)  # default beta for Huber

        if bbox_loss_type == 'huber':
            loss_bbox_charged = F.smooth_l1_loss(pred_boxes_track_cat, tgt_boxes_track_cat, reduction='sum', beta=huber_beta) \
            if pred_boxes_track_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_track.device)
            loss_bbox_neutral = F.smooth_l1_loss(pred_boxes_notrack_cat[mask_true_particle], tgt_boxes_notrack_cat[mask_true_particle], reduction='sum', beta=huber_beta) \
            if mask_true_particle.sum() > 0 else torch.tensor(0.0, device=pred_logits_notrack.device)
        else:
            loss_bbox_charged = F.l1_loss(pred_boxes_track_cat, tgt_boxes_track_cat, reduction='sum') \
            if pred_boxes_track_cat.shape[0] > 0 else torch.tensor(0.0, device=pred_logits_track.device)
            loss_bbox_neutral = F.l1_loss(pred_boxes_notrack_cat[mask_true_particle], tgt_boxes_notrack_cat[mask_true_particle], reduction='sum') \
            if mask_true_particle.sum() > 0 else torch.tensor(0.0, device=pred_logits_notrack.device)

        if self.loss_coeffs["conservation"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            
            pred_pt_track = pred_boxes_track[..., 0]  # pT is the first element
            pred_pt_notrack = pred_boxes_notrack[..., 0]  # pT is the first element
            truth_pt = tgt_boxes[..., 0]  # pT is the first element in the target boxes
            # === PARTIE TRACK ===
            # Toutes les prédictions track sont bonnes
            pred_log_pt_norm_track = pred_pt_track  # pT est déjà le premier élément
            pred_log_pt_unnorm_track = pred_log_pt_norm_track * self.particle_logpt_std + self.particle_logpt_mean
            pred_pt_linear_track = torch.exp(pred_log_pt_unnorm_track)
            
            # === PARTIE NOTRACK ===
            # Pour notrack, exclure la classe 2 (NoneParticle)
            pred_log_pt_norm_notrack = pred_pt_notrack  # pT est déjà le premier élément
            pred_log_pt_unnorm_notrack = pred_log_pt_norm_notrack * self.particle_logpt_std + self.particle_logpt_mean
            pred_pt_linear_notrack = torch.exp(pred_log_pt_unnorm_notrack)
            
            # Masque pour exclure le padding ET la classe 2 (NoneParticle) dans les notrack
            pred_probs_notrack = F.softmax(pred_logits_notrack, -1)
            pred_labels_notrack = pred_probs_notrack.argmax(-1)
            pred_non_none_particle_mask_notrack = (pred_labels_notrack != 2)  # Exclure classe 2
            pred_pt_linear_masked_notrack = pred_pt_linear_notrack * pred_non_none_particle_mask_notrack
            
            # === CONCATENATION ===
            # Concaténer les pT masqués des track et notrack
            pred_pt_linear_masked = torch.cat([pred_pt_linear_track, pred_pt_linear_masked_notrack], dim=1)
            
            # === PARTIE TRUTH ===
            # Traitement de la vérité terrain (inchangé)
            truth_log_pt_norm = truth_pt  # pT est le premier élément dans tgt_boxes
            truth_log_pt_unnorm = truth_log_pt_norm  * self.particle_logpt_std + self.particle_logpt_mean
            truth_pt_linear = torch.exp(truth_log_pt_unnorm)
            
            truth_non_padding_mask = (tgt_labels != 5)
            truth_pt_linear_masked = truth_pt_linear * truth_non_padding_mask
            
            # === CALCUL DE LA CONSERVATION ===
            # Somme des pT par événement
            sum_pt_pred_event = torch.sum(pred_pt_linear_masked, dim=1)
            sum_pt_truth_event = torch.sum(truth_pt_linear_masked, dim=1)
            
            # Calcul de la perte relative
            epsilon = 1e-8
            rel_diff = (sum_pt_pred_event - sum_pt_truth_event) / (sum_pt_truth_event + epsilon)
            loss_conservation = torch.mean(torch.abs(rel_diff))
        else:
            loss_conservation = torch.tensor(0.0, device=pred_logits_track.device)

        if self.loss_coeffs["cardinality"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            # Cardinalité neutre : compter les classes 3 et 4 dans les labels
            

            pred_probs_notrack = F.softmax(pred_logits_notrack, -1)
            neutral_cardinality = ((tgt_labels != 5) & (tgt_labels > 2)).sum(dim=-1)  # Exclure padding et classe 2 (NoneParticle)

            # Sommer les probabilités des classes réelles (0,1) - exclure padding (2)
            pred_cardinality = pred_probs_notrack[:, :, :2].sum(dim=(1, 2))
            
            loss_cardinality_neutral = F.l1_loss(pred_cardinality, neutral_cardinality.float())
        
        else:
            loss_cardinality_neutral = torch.tensor(0.0, device=pred_logits_notrack.device)

        # Combine and normalize manually
        loss_cls = loss_cls_charged + loss_cls_neutral
        loss_bbox = loss_bbox_charged + loss_bbox_neutral

        if self.loss_coeffs["conservation"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            loss_conservation = self.loss_coeffs["conservation"] * loss_conservation
            total_loss = (self.loss_coeffs["cls"] * loss_cls + self.loss_coeffs["bbox"] * loss_bbox) / total_num_objects + loss_conservation
        elif self.loss_coeffs["cardinality"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            loss_cardinality_neutral = self.loss_coeffs["cardinality"] * loss_cardinality_neutral
            total_loss = (self.loss_coeffs["cls"] * loss_cls + self.loss_coeffs["bbox"] * loss_bbox + loss_cardinality_neutral) / total_num_objects + loss_cardinality_neutral
        elif self.loss_coeffs["indence_matrix"] > 0.0 and self.use_indence_matrix_prediction:
            total_loss = (self.loss_coeffs["cls"] * loss_cls + self.loss_coeffs["bbox"] * loss_bbox + self.loss_coeffs["indence_matrix"] * loss_ind) / total_num_objects
        else:
            total_loss = (self.loss_coeffs["cls"] * loss_cls + self.loss_coeffs["bbox"] * loss_bbox) / total_num_objects


        # --- 7. Logging ---
        log_values = {
            log_name: total_loss.item(),
            f"{log_name}_bbox_charged": loss_bbox_charged.item()/ total_num_objects.item(),
            f"{log_name}_bbox_neutral": loss_bbox_neutral.item() / total_num_objects.item(),
            f"{log_name}_cls_charged": loss_cls_charged.item() / total_num_objects.item(),
            f"{log_name}_cls_neutral": loss_cls_neutral.item() / total_num_objects.item(),
            f"{log_name}_cls": loss_cls.item() / total_num_objects.item(),
            f"{log_name}_bbox": loss_bbox.item() / total_num_objects.item(),
        }
        if self.loss_coeffs["conservation"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            log_values[f"{log_name}_conservation"] = loss_conservation.item()
        if self.loss_coeffs["cardinality"] > 0.0 and self.current_epoch >= self.n_epoch_warmup:
            log_values[f"{log_name}_cardinality_neutral"] = loss_cardinality_neutral.item()
        if self.loss_coeffs["indence_matrix"] > 0.0:
            log_values[f"{log_name}_indence"] = loss_ind.item()/ total_num_objects.item()
        for name, value in log_values.items():
            self.log(name, value, prog_bar=(name == log_name and prog_bar), sync_dist=True)

        # Stockage pour utilisation ultérieure
        setattr(self, log_name, total_loss)

        # --- 8. Métriques d'accuracy ---
        # Combinaison des prédictions pour les métriques globales
        pred_classes_track = pred_logits_track_cat.argmax(-1)
        pred_classes_notrack = pred_logits_notrack_cat.argmax(-1)

        
        # Métriques par type
        class_names = ["Charged hadron", "Electron", "Muon", "Neutral hadron", "Photon", "Padding"]
        # Charged head: classes 0,1,2 (Charged hadron, Electron, Muon)
        # Neutral head: classes 0,1,2 (Neutral hadron, Photon, Padding) after adjustment

        # Charged per-class accuracy
        
        charged_preds = pred_classes_track
        charged_targets = tgt_labels_track_cat
        charged_acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3, average=None).to(charged_preds.device)
        charged_per_class_acc = charged_acc_metric(charged_preds, charged_targets)
        for i, acc in enumerate(charged_per_class_acc):
            self.log(f"{log_name}_acc_{class_names[i]}", acc, prog_bar=False, sync_dist=True)

        # Neutral per-class accuracy (labels are 0: Neutral hadron, 1: Photon, 2: Padding)
        neutral_preds = pred_classes_notrack
        neutral_targets = tgt_labels_notrack_cat
        neutral_acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3, average=None).to(neutral_preds.device)
        neutral_per_class_acc = neutral_acc_metric(neutral_preds, neutral_targets)
        for i, name in enumerate(["Neutral hadron", "Photon", "Padding"]):
            self.log(f"{log_name}_acc_{name}", neutral_per_class_acc[i], prog_bar=False, sync_dist=True)


        # --- Log refinement scales and upgrades if present (tracker model only) ---
        if "track_refinements" in outputs and "original_track_params" in outputs:
            refines = outputs["track_refinements"]
            originals = outputs["original_track_params"]
            # Log refinement scales (global)
            refinement_scales = refines.get("refinement_scales")
            if refinement_scales is not None:
                for i, name in enumerate(["log_pt"]):
                    if hasattr(refinement_scales, "detach"):
                        scale_val = refinement_scales[i].detach().cpu().item() if refinement_scales.numel() > i else float('nan')
                    else:
                        scale_val = float(refinement_scales[i]) if len(refinement_scales) > i else float('nan')
                    self.log(f"{log_name}_refinement_scale_{name}", scale_val, prog_bar=False, sync_dist=True)



        return total_loss
    
    def on_after_backward(self):
        """Log the global L2 norm of gradients after each backward pass."""
        parameters = [p for p in self.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return
        total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2)
        self.log("grad_norm_L2", total_norm, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

    def on_train_epoch_end(self):
        #TODO ipgrade metric logging
        """
        # Calculer les moyennes des métriques système
        avg_metrics = {}
        for key, values in self.system_metrics.items():
            if values and isinstance(values, list):  # Vérifier si la liste n'est pas vide
                if len(values) > 0:  # Vérifier si la liste contient des éléments
                    avg_metrics[f'system_{key}_avg'] = sum(values) / len(values)
                else:
                    continue
            else:
                avg_metrics[f'system_{key}_avg'] = values
        # Ajouter les métriques système aux logs
        print(avg_metrics)
        if avg_metrics:
            self.log_dict(avg_metrics, prog_bar=False, sync_dist=True)
        """
        # Ajouter les métriques système à MLflow
        if self.config.get("training", {}).get("single_mlflow_run", False):
            try:
                mlflow.log_metrics(avg_metrics, step=self.current_epoch)
                
                # Ajouter des tags système
                mlflow.set_tags({
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                    'os_version': platform.platform(),
                    'python_version': platform.python_version(),
                    'nvidia_driver_version': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).decode().strip()
                })
            except:
                pass
        try:
            # Enregistrer des artefacts personnalisés à certaines étapes clés
            if self.current_epoch % 25 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
                # Sauvegarder la configuration actuelle comme artefact
                config_path = f"config_epoch_{self.current_epoch}.json"
                with open(config_path, 'w') as f:
                    # Conversion des tenseurs et autres objets non-sérialisables
                    clean_config = {}
                    for section, params in self.config.items():
                        if isinstance(params, dict):
                            clean_config[section] = {}
                            for k, v in params.items():
                                if hasattr(v, 'tolist') and callable(getattr(v, 'tolist')):
                                    clean_config[section][k] = v.tolist()
                                elif isinstance(v, (int, float, str, bool, list, tuple)):
                                    clean_config[section][k] = v
                                else:
                                    clean_config[section][k] = str(v)
                        else:
                            clean_config[section] = str(params)
                            
                    json.dump(clean_config, f, indent=2)
                try:
                    mlflow.log_artifact(config_path)
                    os.remove(config_path)  # Nettoyer les fichiers temporaires
                except Exception as e:
                    print(f"Error logging config at epoch {self.current_epoch}: {e}")
                    
        except Exception as e:
            print(f"Error in training epoch end: {e}")
            
    def on_train_epoch_start(self):
        # Initialiser les métriques système
        self.system_metrics = {
            'gpu_memory_usage': [],
            'gpu_utilization': [],
            'cpu_memory_usage': [],
            'cpu_utilization': [],
            'batch_load_time': [],
            'batch_process_time': [],
            'epoch_start_time': time.time()
        }

    def on_train_batch_start(self, batch, batch_idx):
        # Mesurer le temps de chargement du batch
        self.batch_load_start = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Mesurer le temps de traitement du batch
        self.batch_process_time = time.time() - self.batch_load_start
        self.system_metrics['batch_process_time'].append(self.batch_process_time)

        # Collecter les métriques GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # en Mo
            self.system_metrics['gpu_memory_usage'].append(gpu_memory)
            
            # Utilisation GPU (nécessite nvidia-smi)
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'],
                                    capture_output=True, text=True)
                gpu_util = float(result.stdout.strip())
                self.system_metrics['gpu_utilization'].append(gpu_util)
            except:
                pass

        # Collecter les métriques CPU
        cpu_memory = psutil.Process().memory_info().rss / 1024**2  # en Mo
        cpu_util = psutil.cpu_percent()
        self.system_metrics['cpu_memory_usage'].append(cpu_memory)
        self.system_metrics['cpu_utilization'].append(cpu_util)

    def training_step(self, batch, batch_idx):

        res = self._step_with_matching(batch, log_name="train_loss", prog_bar=True)
        if batch_idx == 0:
            print(f"[VALIDATION] GPU memory after: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
        return res
    def validation_step(self, batch, batch_idx):

        res = self._step_with_matching(batch, log_name="val_loss", prog_bar=True)
        if batch_idx == 0:
            print(f"[VALIDATION] GPU memory after: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

        return res

    def test_step(self, batch, batch_idx):
        # Identical to validation step
        return self._step_with_matching(batch, log_name="test_loss")
    
    def on_test_end(self):
        """Enregistrer les artefacts et métriques finaux dans MLflow à la fin du test"""
        try:
            # Stocker le chemin de sauvegarde du modèle si disponible
            if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback.best_model_path:
                best_model_path = self.trainer.checkpoint_callback.best_model_path
                # Enregistrer le meilleur modèle dans MLflow
                mlflow.log_artifact(best_model_path)
                
            # Enregistrer des informations supplémentaires sur l'architecture du modèle
            model_info = {
                "hidden_dim": self.config["model"].get("hidden_dim", 256),
                "num_encoder_layers": self.config["model"].get("num_encoder_layers", 6),
                "num_decoder_layers": self.config["model"].get("num_decoder_layers", 6),
                "model_version": "DETR_V2" if self.config["model"].get("use_DETR_V2", False) else "DETR_V1",
                "class_weights": str(self.config["model"].get("class_weights", [1.0]*6)),
            }
            
            # Log des paramètres additionnels
            mlflow.log_params(model_info)
            
        except Exception as e:
            print(f"Error in test end: {e}")
            
    def on_fit_end(self):
        """Actions à effectuer à la fin de l'entraînement"""
        # S'assurer que toutes les métriques finales sont bien enregistrées
        if self.validation_metrics:
            # Les métriques sont maintenant déjà enregistrées à chaque epoch
            # mais on les renvoie une dernière fois pour être sûr
            mlflow.log_metrics(self.validation_metrics, step=self.current_epoch)
            
            # Enregistrer un résumé des métriques finales
            summary_metrics = {}
            for key, value in self.validation_metrics.items():
                if key.startswith("val_"):
                    # Créer une version "final_" de chaque métrique
                    summary_metrics[f"final_{key[4:]}"] = value
            
            # Enregistrer le résumé des métriques
            if summary_metrics:
                mlflow.log_metrics(summary_metrics, step=self.current_epoch)
            
        # Sauvegarder la configuration finale comme artefact
        config_path = "final_config.json"
        with open(config_path, 'w') as f:
            # Conversion des tenseurs et autres objets non-sérialisables
            clean_config = {}
            for section, params in self.config.items():
                if isinstance(params, dict):
                    clean_config[section] = {}
                    for k, v in params.items():
                        if hasattr(v, 'tolist') and callable(getattr(v, 'tolist')):
                            clean_config[section][k] = v.tolist()
                        elif isinstance(v, (int, float, str, bool, list, tuple)):
                            clean_config[section][k] = v
                        else:
                            clean_config[section][k] = str(v)
                else:
                    clean_config[section] = str(params)
                    
            json.dump(clean_config, f, indent=2)
        try:
            mlflow.log_artifact(config_path)
            os.remove(config_path)  # Nettoyer le fichier temporaire
        except Exception as e:
            print(f"Error logging final config: {e}")

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler using built-in PyTorch schedulers.
        Supports optional linear warmup followed by CosineAnnealingLR, or ReduceLROnPlateau.
        """
        tcfg = self.config["training"]
        # Base parameters
        base_lr = tcfg["learningrate"]
        weight_decay = tcfg.get("weight_decay", 0)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=base_lr, weight_decay=weight_decay
        )

        # Scheduler config
        sched_cfg = tcfg.get("lr_scheduler")
        if not sched_cfg:
            return optimizer

        name = sched_cfg.get("name")
        # CosineAnnealingWarmRestarts with optional multiplicative decay
        if name == "CosineAnnealingWarmRestarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, MultiplicativeLR

            T_0 = sched_cfg.get("T_0", 10)
            T_mult = sched_cfg.get("T_mult", 1)
            eta_min = sched_cfg.get("eta_min", 1e-6)
            decay_factor = sched_cfg.get("decay_factor", None)

            # Optionally add a warmup phase before the first restart
            warmup_steps = tcfg.get("warmup_steps", 0)
            warmup_start_lr = tcfg.get("warmup_start_lr", base_lr)
            schedulers = []
            milestones = []
            if warmup_steps > 0:
                from torch.optim.lr_scheduler import LinearLR
                start_factor = warmup_start_lr / base_lr
                warmup_sched = LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    total_iters=warmup_steps,
                )
                schedulers.append(warmup_sched)
                milestones.append(warmup_steps)
            cosine_restart_sched = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
            )
            if decay_factor is not None:
                # Apply multiplicative decay after each restart
                def decay_lambda(epoch):
                    # Count how many restarts have occurred
                    import math
                    if epoch < warmup_steps:
                        return 1.0
                    # epoch offset for warmup
                    effective_epoch = epoch - warmup_steps
                    # Compute number of completed restarts
                    n = 0
                    t = T_0
                    total = 0
                    while total + t <= effective_epoch:
                        total += t
                        t *= T_mult
                        n += 1
                    return decay_factor ** n
                multiplicative_sched = MultiplicativeLR(optimizer, lr_lambda=decay_lambda)
                schedulers.append(multiplicative_sched)
            schedulers.append(cosine_restart_sched)
            if warmup_steps > 0:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=schedulers,
                    milestones=milestones,
                )
            else:
                if len(schedulers) > 1:
                    # Combine multiplicative and cosine
                    scheduler = SequentialLR(
                        optimizer,
                        schedulers=schedulers,
                        milestones=[0],
                    )
                else:
                    scheduler = cosine_restart_sched

        # CosineAnnealingLR with optional warmup
        elif name == "CosineAnnealingLR":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup_steps = sched_cfg.get("warmup_steps", 0)
            warmup_start_lr = sched_cfg.get("warmup_start_lr", base_lr)

            # No warmup: use pure CosineAnnealingLR
            if warmup_steps <= 0:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=sched_cfg.get("T_max", 50),
                    eta_min=sched_cfg.get("eta_min", 1e-6),
                )
            else:
                # Linear warmup scheduler: linearly increase LR from warmup_start_lr to base_lr
                start_factor = warmup_start_lr / base_lr
                warmup_sched = LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    total_iters=warmup_steps,
                )
                # Cosine annealing after warmup
                cosine_sched = CosineAnnealingLR(
                    optimizer,
                    T_max=sched_cfg.get("T_max", 50),
                    eta_min=sched_cfg.get("eta_min", 1e-6),
                )
                # Chain warmup and cosine schedulers
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_steps],
                )

        # StepLR: classic step learning rate decay
        elif name == "StepLR":
            from torch.optim.lr_scheduler import StepLR
            step_size = sched_cfg.get("step_size", 40)
            gamma = sched_cfg.get("gamma", 0.1)
            scheduler = StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )

        # ReduceLROnPlateau
        elif name == "ReduceLROnPlateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(
                optimizer,
                **{k: v for k, v in sched_cfg.items() if k != "name"},
            )
        elif name =="ScheduleFree":
            raise NotImplementedError(f"ScheduleFree scheduler is not implemented")
        else:
            raise NotImplementedError(f"Unknown scheduler: {name}")

        # Return Lightning-compatible dictionary
        lr_dict = {"scheduler": scheduler}
        if name == "ReduceLROnPlateau":
            lr_dict["monitor"] = "val_loss"

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @staticmethod
    def load_from_checkpoint_and_config(ckpt_path, config, map_location=None):
        """Charge un modèle DETRLightningModule depuis un checkpoint et une config."""
        model = DETRLightningModule(config)
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        model.eval()
        
        return model
