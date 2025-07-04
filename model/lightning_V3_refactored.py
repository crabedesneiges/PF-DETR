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
    def _get_src_permutation_idx(self, indices):
        """
        Prend les indices du matcher et les transforme pour un gather sur le batch.
        
        Args:
            indices (list[tuple]): Une liste de tuples (src_idx, tgt_idx) pour chaque élément du batch.

        Returns:
            tuple[torch.Tensor]: Un tuple de (batch_idx, src_idx) prêt pour l'indexation.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _step_with_matching(self, batch, log_name, prog_bar=False):
        # --- 1. Inférence du modèle (inchangé) ---
        x = batch['input']
        cell_mask = batch.get('cell_mask', None)
        topo_mask = batch.get('topo_mask', None)
        track_mask = batch.get('track_mask', None)
        outputs = self.model(x, cell_mask=cell_mask, topo_mask=topo_mask, track_mask=track_mask)

        pred_logits_track = outputs["pred_logits_track"]
        pred_boxes_track = outputs["pred_boxes_track"]
        pred_logits_notrack = outputs["pred_logits_notrack"]
        pred_boxes_notrack = outputs["pred_boxes_notrack"]

        # --- 2. Préparation des cibles (Ground Truth) en listes ---
        targets = batch["target"]
        batch_size = pred_logits_notrack.shape[0]

        # C'est plus efficace de préparer les listes de cibles une fois pour le matcher
        tgt_labels_track_list = []
        tgt_boxes_track_list = []
        tgt_labels_notrack_list = []
        tgt_boxes_notrack_list = []

        for b in range(batch_size):
            is_track_mask = targets["is_track"][b] == 1
            is_notrack_mask = targets["is_track"][b] == 0

            # Cibles pour la tête "track"
            tgt_labels_track_list.append(targets["labels"][b][is_track_mask])
            tgt_boxes_track_list.append(targets["boxes"][b][is_track_mask])
            
            # Cibles pour la tête "notrack"
            lbl_notrack = targets["labels"][b][is_notrack_mask]
            box_notrack = targets["boxes"][b][is_notrack_mask]
            
            # Filtrer les objets réels (non-padding)
            real_neutral_mask = (lbl_notrack != self.padding_idx)
            lbl_notrack_real = lbl_notrack[real_neutral_mask]
            
            # Ajustement des labels pour la tête "notrack"
            if len(lbl_notrack_real) > 0:
                lbl_notrack_real = torch.clamp(lbl_notrack_real - 3, min=0)

            tgt_labels_notrack_list.append(lbl_notrack_real)
            tgt_boxes_notrack_list.append(box_notrack[real_neutral_mask])
        
        # --- 3. Matching pour les deux têtes (appels vectoriels) ---
        self.matcher.padding_idx = None # Pas de classe "no object" pour les particules chargées
        indices_track = self.matcher(pred_logits_track, pred_boxes_track, tgt_labels_track_list, tgt_boxes_track_list)

        self.matcher.padding_idx = 2 # Classe "None Particle" pour les neutres
        indices_notrack = self.matcher(pred_logits_notrack, pred_boxes_notrack, tgt_labels_notrack_list, tgt_boxes_notrack_list)

        # --- 4. Alignement des prédictions et cibles via les indices ---
        
        # Pour la tête "track"
        idx_track = self._get_src_permutation_idx(indices_track)
        pred_logits_track_cat = pred_logits_track[idx_track]
        pred_boxes_track_cat = pred_boxes_track[idx_track]
        
        tgt_labels_track_cat = torch.cat([t[J] for t, (_, J) in zip(tgt_labels_track_list, indices_track)])
        tgt_boxes_track_cat = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes_track_list, indices_track)])

        # Pour la tête "notrack"
        idx_notrack = self._get_src_permutation_idx(indices_notrack)
        
        # La loss de classification pour "notrack" se calcule sur toutes les queries
        # car il faut pénaliser les fausses détections (classe "None Particle").
        # On crée la cible alignée.
        tgt_labels_notrack_cat = torch.full(pred_logits_notrack.shape[:2], 2, dtype=torch.int64, device=self.device)
        tgt_labels_notrack_cat[idx_notrack] = torch.cat([t[J] for t, (_, J) in zip(tgt_labels_notrack_list, indices_notrack)])

        # La loss bbox ne s'applique qu'aux objets appariés.
        pred_boxes_notrack_matched = pred_boxes_notrack[idx_notrack]
        tgt_boxes_notrack_matched = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes_notrack_list, indices_notrack)])

        # --- 5. Calcul des pertes (Losses) ---
        
        # Nombre d'objets pour la normalisation
        num_objects_charged = len(tgt_labels_track_cat)
        num_objects_neutral = len(tgt_boxes_notrack_matched)
        num_objects = torch.as_tensor(num_objects_charged + num_objects_neutral, dtype=torch.float, device=self.device)
        total_num_objects = self.all_gather_total(num_objects).clamp(min=1.0)

        # Loss de classification
        charged_class_weights = torch.tensor(self.config['model'].get('charged_class_weights', [1.0]*3), device=self.device)
        loss_cls_charged = F.cross_entropy(pred_logits_track_cat, tgt_labels_track_cat, weight=charged_class_weights, reduction='sum')

        neutral_class_weights = torch.tensor(self.config['model'].get('neutral_class_weights', [1.0]*3), device=self.device)
        neutral_class_weights[2] = self.config['model'].get('no_object_weight', 0.1)
        loss_cls_neutral = F.cross_entropy(pred_logits_notrack.permute(0, 2, 1), tgt_labels_notrack_cat, weight=neutral_class_weights, reduction='sum')

        # Loss de BBox
        loss_bbox_charged = F.l1_loss(pred_boxes_track_cat, tgt_boxes_track_cat, reduction='sum')
        loss_bbox_neutral = F.l1_loss(pred_boxes_notrack_matched, tgt_boxes_notrack_matched, reduction='sum')
        
        loss_cls = loss_cls_charged + loss_cls_neutral
        loss_bbox = loss_bbox_charged + loss_bbox_neutral
        
        # --- NOTE: Les autres pertes (conservation, cardinality, incidence) nécessiteraient une refactorisation similaire. ---
        # Pour l'instant, nous nous concentrons sur les pertes principales.
        # TODO: Refactoriser les autres types de pertes si elles sont utilisées.
        loss_conservation = torch.tensor(0.0, device=self.device)
        loss_cardinality_neutral = torch.tensor(0.0, device=self.device)
        loss_ind = torch.tensor(0.0, device=self.device)

        total_loss = (self.loss_coeffs["cls"] * loss_cls + self.loss_coeffs["bbox"] * loss_bbox) / total_num_objects

        # --- 6. Logging (inchangé, mais plus propre car les valeurs sont déjà calculées) ---
        log_values = {
            log_name: total_loss.item(),
            f"{log_name}_bbox_charged": (loss_bbox_charged.item() / total_num_objects.item()) if num_objects_charged > 0 else 0.0,
            f"{log_name}_bbox_neutral": (loss_bbox_neutral.item() / total_num_objects.item()) if num_objects_neutral > 0 else 0.0,
            f"{log_name}_cls_charged": (loss_cls_charged.item() / total_num_objects.item()) if num_objects_charged > 0 else 0.0,
            f"{log_name}_cls_neutral": (loss_cls_neutral.item() / total_num_objects.item()),
        }
        self.log_dict(log_values, prog_bar=(log_name=="train_loss"), sync_dist=True)

        # --- 7. Métriques d'accuracy ---
        # TODO: Déplacer l'initialisation des métriques dans __init__ !
        # Le calcul ci-dessous est conservé pour la logique, mais il est inefficace.
        
        # Accuracy pour les particules chargées
        charged_acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3, average=None).to(self.device)
        charged_acc_metric.update(pred_logits_track_cat.argmax(-1), tgt_labels_track_cat)
        charged_per_class_acc = charged_acc_metric.compute()
        class_names = ["Charged hadron", "Electron", "Muon", "Neutral hadron", "Photon", "Padding"]
        for i, acc in enumerate(charged_per_class_acc):
            self.log(f"{log_name}_acc_{class_names[i]}", acc, prog_bar=False, sync_dist=True)

        # Accuracy pour les particules neutres
        neutral_acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=3, average=None).to(self.device)
        neutral_preds_matched = pred_logits_notrack.argmax(-1)[idx_notrack]
        neutral_targets_matched = tgt_labels_notrack_cat[idx_notrack]
        neutral_acc_metric.update(neutral_preds_matched, neutral_targets_matched)
        for i, acc in enumerate(neutral_acc_metric.compute()):
            self.log(f"{log_name}_neutral_acc_{class_names[i+3]}", acc, prog_bar=False, sync_dist=True)

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

        if batch_idx == 0:
            print(f"Model device: {next(self.parameters()).device}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"Batch[{key}] device: {value.device}, shape: {value.shape}")
            print(f"[VALIDATION] Model device: {next(self.parameters()).device}")

        res = self._step_with_matching(batch, log_name="train_loss", prog_bar=True)
        if batch_idx == 0:
            print(f"[VALIDATION] GPU memory after: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
        return res
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"Model device: {next(self.parameters()).device}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"Batch[{key}] device: {value.device}, shape: {value.shape}")
            print(f"[VALIDATION] Model device: {next(self.parameters()).device}")

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
