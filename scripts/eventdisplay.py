import os
import click
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import yaml
import torch
from pytorch_lightning import Trainer
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import get_latest_checkpoints, setup_environment
from data.pflow_datamodule import PFlowDataModule
from model.lightning_V3 import DETRLightningModule
from data.normalizer import Normalizer
from model.hungarian_matcher import HungarianMatcher
normalizer = Normalizer("data/normalization/params.json")


def process_predictions(predictions):
    """
    Transforme les sorties batchées en une liste d'événements (dicts) avec :
    - pred_logits : (n_queries, n_classes)
    - pred_boxes : (n_queries, n_box_features)
    - truth_labels : (n_truth, )
    - truth_boxes : (n_truth, n_box_features)
    Les prédictions sont réordonnées par Hungarian matching pour correspondre à l'ordre du ground_truth.
    """
    matcher = HungarianMatcher(cls_cost=1.0, bbox_cost=1.0, padding_idx=5)
    results = []
    for batch in predictions:
        
        for b in range(batch['pred_logits_track'].shape[0]):


            pred_logits_track = batch["pred_logits_track"]
            pred_boxes_track = batch["pred_boxes_track"]
            pred_logits_notrack = batch["pred_logits_notrack"]
            pred_boxes_notrack = batch["pred_boxes_notrack"]
            truth_labels = batch["ground_truth"]["labels"]
            truth_boxes = batch["ground_truth"]["boxes"]
            truth_is_track = batch["ground_truth"]["is_track"]

            condition1_mask = (truth_labels == 0) & (truth_is_track == 0)
            condition2_mask = (truth_labels == 1) & (truth_is_track == 0)
            # Clone to allow in-place modifications outside inference mode
            truth_labels = truth_labels.clone()
            truth_labels[condition1_mask] = 3  # charged_hadron -> neutral_hadron
            truth_labels[condition2_mask] = 4  # electron -> photons

            is_track_mask = truth_is_track[b] == 1
            is_notrack_mask = truth_is_track[b] == 0

            tgt_labels_charged = truth_labels[b][is_track_mask]
            tgt_boxes_charged = truth_boxes[b][is_track_mask]

            tgt_labels_neutral = truth_labels[b][is_notrack_mask]
            tgt_boxes_neutral = truth_boxes[b][is_notrack_mask]

            real_neutral_mask = (tgt_labels_neutral != 5)
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
            #print("aligned_preds_boxes_track", aligned_preds_boxes_track.shape)
            #print("pred_boxes_notrack", pred_boxes_notrack[b].shape)

            #print("aligned_preds_logit_track", aligned_preds_boxes_track.shape)
            #print("pred_boxes_track", pred_logits_notrack[b].shape)

            #print("tgt_boxes_charged", tgt_boxes_charged.shape)
            #print("aligned_tgt_boxes_notrack", aligned_tgt_boxes_notrack.shape)

            #print("tgt_labels_charged", tgt_labels_charged.shape)
            #print("aligned_tgt_label_notrack", aligned_tgt_label_notrack.shape)
            MAX_TRACK_PARTICLE = 15
            num_track = aligned_preds_logit_track.shape[0]
            pad_track = MAX_TRACK_PARTICLE - num_track

            # 1. Pad logits (track only)
            track_logits = torch.full((MAX_TRACK_PARTICLE, 6), -1e4, device=aligned_preds_logit_track.device)
            track_logits[:num_track, :3] = aligned_preds_logit_track
            track_logits[num_track:, 5] = 0.0  # Set class 5 (padding class) to 0 logit

            # 2. Pad boxes (track only) with zeros (or sentinel)
            track_boxes_pred = torch.zeros((MAX_TRACK_PARTICLE, 3), device=aligned_preds_boxes_track.device)
            track_boxes_pred[:num_track] = aligned_preds_boxes_track

            # 3. Pad labels (track only) with label 5
            track_labels = torch.full((MAX_TRACK_PARTICLE,), 5, device=tgt_labels_charged.device)
            track_labels[:num_track] = tgt_labels_charged

            # 4. Pad truth boxes (track only) with zeros (or sentinel)
            # 3. Pad labels (track only) with label 5
            track_truth_boxes = torch.full((MAX_TRACK_PARTICLE, 3), 0.0, device=tgt_boxes_charged.device)
            track_truth_boxes[:num_track] = tgt_boxes_charged        
                    

            # 4. Format notrack logits
            notrack_logits = torch.full((pred_logits_notrack[b].shape[0], 6), -1e4, device=pred_logits_notrack[b].device)
            notrack_logits[:, 3:] = pred_logits_notrack[b]

            # 5. Final concatenation
            matched_pred_logit = torch.cat([track_logits, notrack_logits], dim=0)
            matched_pred_boxes = torch.cat([track_boxes_pred, pred_boxes_notrack[b]], dim=0)

            aligned_tgt_label_notrack_offset = aligned_tgt_label_notrack + 3
            matched_truth_label = torch.cat([track_labels, aligned_tgt_label_notrack_offset], dim=0)
            matched_truth_boxes = torch.cat([track_truth_boxes, aligned_tgt_boxes_notrack], dim=0)

            # 6. Add to results
            event = {
                "pred_logits": matched_pred_logit.detach().cpu().numpy(),
                "pred_boxes": matched_pred_boxes.detach().cpu().numpy(),
                "truth_labels": matched_truth_label.detach().cpu().numpy(),
                "truth_boxes": matched_truth_boxes.detach().cpu().numpy(),
            }
            results.append(event)

    return results

def denormalize_data(results, datamodule):
    """
    Applique la dénormalisation sur les champs pertinents (ex: pred_boxes, truth_boxes).
    Utilise le normalizer du datamodule (test_dataset ou val_dataset).
    """
    # On tente d'utiliser le normalizer du test_dataset, sinon val_dataset
    dataset = getattr(datamodule, "test_dataset", None)
    if dataset is None:
        dataset = getattr(datamodule, "val_dataset", None)
    if dataset is None or not hasattr(dataset, "normalizer"):
        raise ValueError("Impossible de trouver un normalizer dans le datamodule.")
    normalizer = dataset.normalizer
    
    # Récupération des noms des variables p4 depuis input_variables.yaml
    p4_names = dataset.input_variables["graph"]["truths"]["p4"]
    
    # Vérifier quelles variables sont utilisées et noms complets
    
    
    # Vérifier dimensions des tenseurs de boxes
    if results and "pred_boxes" in results[0]:
        print(f"pred_boxes shape: {results[0]['pred_boxes'].shape}")
    if results and "truth_boxes" in results[0]:
        print(f"truth_boxes shape: {results[0]['truth_boxes'].shape}")
    
    # Enlever le suffix :normed pour la dénormalisation
    clean_p4_names = [name.replace(":normed", "") for name in p4_names]
    
    
    # Dénormalisation des boxes (p4) prédites et ground truth
    for event_idx, event in enumerate(results):
        # Debugger le premier événement en détail
        debug = event_idx == 0
        #debug = False
        if debug:
            print("\nDebug denormalization for first event:")
        
        for i, (name, clean_name) in enumerate(zip(p4_names, clean_p4_names)):
            #print("clean_name:", clean_name)
            if "pred_boxes" in event and i < event["pred_boxes"].shape[1]:
                if debug:
                    print(f"\nVariable {i}: {clean_name}")
                    sample = event["pred_boxes"][:5, i]  # Show first 5 values
                    print(f"  Before denorm: {sample}")
                
                # Dénormaliser en utilisant la même clé de groupe que lors de la normalisation
                grp_key = dataset.var2grp.get(clean_name, clean_name)
                #print("grp_key pred:", grp_key)
                
                event["pred_boxes"][:, i] = normalizer.denormalize(event["pred_boxes"][:, i], name=grp_key)
                
                if debug:
                    sample = event["pred_boxes"][:5, i]  # Show first 5 values after
                    print(f"  After denorm: {sample}")
                    
                    # Appliquer exp() si c'est logpt pour obtenir pt
                    if clean_name == "particle_logpt":
                        exp_sample = np.exp(sample)
                        print(f"  After exp(): {exp_sample} (if needed for logpt->pt)")
            
            if "truth_boxes" in event and i < event["truth_boxes"].shape[1]:
                if debug and i == 0:  # Only for first dimension
                    sample = event["truth_boxes"][:5, i]  # Show first 5 values
                    print(f"\n  Truth before denorm: {sample}")
                    
                grp_key = dataset.var2grp.get(clean_name, clean_name)
                #print("grp_key truth:", grp_key)
                
                event["truth_boxes"][:, i] = normalizer.denormalize(event["truth_boxes"][:, i], name=grp_key)
                
                if debug and i == 0:  # Only for first dimension
                    sample = event["truth_boxes"][:5, i]  # Show first 5 values after
                    print(f"  Truth after denorm: {sample}")

    
            global_eta = dataset[event_idx]["global_eta"]
            global_phi = dataset[event_idx]["global_phi"]
            eta_offset = global_eta.item() if hasattr(global_eta, 'item') else global_eta
            phi_offset = global_phi.item() if hasattr(global_phi, 'item') else global_phi
        event["pred_boxes"][:, 1] = event["pred_boxes"][:, 1] + eta_offset
        event["pred_boxes"][:, 2] = event["pred_boxes"][:, 2] + phi_offset

        event["truth_boxes"][:, 1] = event["truth_boxes"][:, 1] + eta_offset
        event["truth_boxes"][:, 2] = event["truth_boxes"][:, 2] + phi_offset

    # Ajouter une dernière étape pour convertir logpt en pt si nécessaire
    if "particle_logpt" in clean_p4_names:
        logpt_idx = clean_p4_names.index("particle_logpt")
        print(f"Converting logpt to pt at index {logpt_idx}")
        
        for event in results:
            if "pred_boxes" in event and logpt_idx < event["pred_boxes"].shape[1]:
                # Convertir logpt -> pt en appliquant exp()
                event["pred_boxes"][:, logpt_idx] = np.exp(event["pred_boxes"][:, logpt_idx])
            if "truth_boxes" in event and logpt_idx < event["truth_boxes"].shape[1]:
                event["truth_boxes"][:, logpt_idx] = np.exp(event["truth_boxes"][:, logpt_idx])
    
    return results

class EventPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        self.labels = ['muon_charged', 'electron_charged', 'muon', 'neutral_hadron', 'photon']
        self.markers_pred = ['o'] * 5
        self.markers_truth = ['x'] * 5
        
        # Create colormap for topoclusters
        import matplotlib.cm as cm
        self.cmap = cm.get_cmap('tab20', 20)  # Use tab20 colormap with 20 distinct colors

    def plot_events(self, results, input_data, datamodule, event_indices=None):
        # Retrieve p4_names from the dataset input_variables
        p4_names = input_data[0]["target"]["p4"]["names"]
        p4_names = [name.replace(":normed", "") for name in p4_names]

        print(input_data[0]["global_eta"])
        print(input_data[0]["global_phi"])

        #print(results[0]["pred_boxes"][0])
        print(results[0]["truth_boxes"][0])
        
        print(input_data[0]["target"]["p4"]["names"])
        print(input_data[0]["target"]["p4"]["values"][0])

        for idx, event in enumerate(results):
            # Use the actual event index if provided, otherwise use loop index
            event_idx = event_indices[idx] if event_indices is not None else idx
            centre_eta = input_data[idx]["global_eta"]
            centre_phi = input_data[idx]["global_phi"]
            tracks = input_data[idx]['tracks']
            cells = input_data[idx]['cells']  # Get cells information
            topoclusters = input_data[idx]['topoclusters']  # Get topoclusters information
            pred_class = event["pred_class"]  # (n_queries,)
            pred_boxes = event["pred_boxes"]  # (n_queries, 4 or more) - Already denormalized
            truth_labels = event["truth_labels"]  # Using the denormalized results
            truth_boxes = event["truth_boxes"]  # Using the denormalized results
            
            # Save pt comparison between prediction and truth
            self.save_pt_comparison(event, p4_names, event_idx)

            # Only keep classes 0-4 (ignore 5=fake)
            pred_mask = (pred_class >= 0) & (pred_class <= 4)

            eta_offset = centre_eta.item() if hasattr(centre_eta, 'item') else centre_eta
            phi_offset = centre_phi.item() if hasattr(centre_phi, 'item') else centre_phi

            pred_eta = pred_boxes[pred_mask, 1]
            pred_phi = pred_boxes[pred_mask, 2]
            pred_cls = pred_class[pred_mask]

            truth_mask = (truth_labels >= 0) & (truth_labels <= 4)

            truth_eta = np.asarray(truth_boxes[truth_mask, 1])
            truth_phi = np.asarray(truth_boxes[truth_mask, 2])
            truth_cls = truth_labels[truth_mask]

            # Combine all eta/phi for this event
            all_eta = np.concatenate([pred_eta, truth_eta])
            all_phi = np.concatenate([pred_phi, truth_phi])
            if all_eta.size == 0 or all_phi.size == 0:
                # Skip empty event
                continue

            delta_eta = 0.4
            delta_phi = 0.4
            delta_max = max(delta_eta, delta_phi)
            margin = 0.0
            eta_min = centre_eta - delta_max - margin
            eta_max = centre_eta + delta_max + margin
            phi_min = centre_phi - delta_max - margin
            phi_max = centre_phi + delta_max + margin

            plt.figure(figsize=(8, 8))
            for i in range(5):
                # Plot predicted
                mask = pred_cls == i
                plt.scatter(pred_eta[mask], pred_phi[mask], c=self.colors[i], marker=self.markers_pred[i], label=f'Pred {self.labels[i]}', alpha=0.7)
                # Plot truth
                mask_t = truth_cls == i
                plt.scatter(truth_eta[mask_t], truth_phi[mask_t], c=self.colors[i], marker=self.markers_truth[i], label=f'True {self.labels[i]}', alpha=0.7)

            

            # --- Plot tracks as arrows in (eta, phi) ---
            # Get track features for this event
            feat_names = tracks['names']
            features = tracks['features']  # shape: (n_tracks, n_features)

            # --- Unnormalize track features and boxes (p4) ---
            # On tente d'utiliser le normalizer du test_dataset, sinon val_dataset
            dataset = getattr(datamodule, "test_dataset", None)
            if dataset is None:
                dataset = getattr(datamodule, "val_dataset", None)
            if dataset is None or not hasattr(dataset, "normalizer"):
                raise ValueError("Impossible de trouver un normalizer dans le datamodule.")            
            # Convert to numpy if tensor
            if hasattr(features, 'detach'):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.array(features)
            features_unnorm = features_np.copy()
            for i, name in enumerate(feat_names):
                grp_key = dataset.var2grp.get(name, name)
                features_unnorm[:, i] = normalizer.denormalize(features_np[:, i], name=grp_key)
            features = features_unnorm  # Use unnormalized features for plotting

            # Indices for the initial and layer features
            eta0_idx = feat_names.index('track_deltaeta')
            phi0_idx = feat_names.index('track_deltaphi')
            eta_layer_idx = [feat_names.index(f'track_deltaeta_layer_{i}') for i in range(6)]
            phi_layer_idx = [feat_names.index(f'track_deltaphi_layer_{i}') for i in range(6)]

            handles_pred = []
            handles_truth = []
            handles_track = []

            for tr in range(features.shape[0]):
                # Start at (track_deltaeta, track_deltaphi)
                eta_start = centre_eta + features[tr, eta0_idx]
                phi_start = centre_phi + features[tr, phi0_idx]

                for i in range(1):
                    delta_eta_end = features[tr, eta_layer_idx[i]]
                    delta_phi_end = features[tr, phi_layer_idx[i]]
                    eta_end = eta_start + delta_eta_end
                    phi_end = phi_start + delta_phi_end

                    alpha = 0.2
                    ar = plt.arrow(
                        eta_start,
                        phi_start,
                        delta_eta_end,
                        delta_phi_end,
                        color="coral",
                        alpha=alpha,
                        head_width=0.01,
                        label="Tracks (PV to Calo layer)" if i == 0 and tr == 0 else "",
                    )
                    if i == 0 and tr == 0:
                        handles_track += [ar]

                    # Print arrow details for event 0
                    if idx == 0 and tr == 0:
                        print(f"Arrow {tr+1} Layer {i}: start=(eta={eta_start}, phi={phi_start}), end=(eta={eta_end}, phi={phi_end})")

            # 1) Build your handle groups as before
            pred_handles  = [Line2D([0], [0], marker=self.markers_pred[i], color='w',
                                    markerfacecolor=self.colors[i], markeredgecolor=self.colors[i],
                                    markersize=8, label=f'Pred {self.labels[i]}')
                            for i in range(5)]
            true_handles  = [Line2D([0], [0], marker=self.markers_truth[i], color='w',
                                    markerfacecolor=self.colors[i], markeredgecolor=self.colors[i],
                                    markersize=8, label=f'True {self.labels[i]}')
                            for i in range(5)]
            track_handles = [Line2D([0], [0], color='coral', lw=2, label='Track')]

            # 2) Plot everything first
            #    (your scatter/arrows code goes here)

            # 3) Draw the three legends:
            ax = plt.gca()

            # Pred column at x=0.25
            leg1 = ax.legend(
                handles=pred_handles,
                ncol=1,
                loc='upper center',
                bbox_to_anchor=(0.25, 1.25),
                frameon=False,
                handletextpad=0.5
            )
            ax.add_artist(leg1)

            # True column at x=0.50
            leg2 = ax.legend(
                handles=true_handles,
                ncol=1,
                loc='upper center',
                bbox_to_anchor=(0.50, 1.25),
                frameon=False,
                handletextpad=0.5
            )
            ax.add_artist(leg2)

            # Track column at x=0.75
            leg3 = ax.legend(
                handles=track_handles,
                ncol=1,
                loc='upper center',
                bbox_to_anchor=(0.75, 1.25),
                frameon=False,
                handletextpad=0.5
            )
            plt.xlim(eta_min, eta_max)
            plt.ylim(phi_min, phi_max)
            plt.grid(True)
            plt.tight_layout()

            # --- Draw matching lines between predictions and truths ---
            try:
                if pred_eta.size > 0 and truth_eta.size > 0:
                    d_eta = truth_eta[:, None] - pred_eta[None, :]
                    d_phi = truth_phi[:, None] - pred_phi[None, :]
                    cost = np.sqrt(d_eta**2 + d_phi**2)
                    row_ind, col_ind = linear_sum_assignment(cost)

                    for t_idx, p_idx in zip(row_ind, col_ind):
                        plt.plot([truth_eta[t_idx], pred_eta[p_idx]], [truth_phi[t_idx], pred_phi[p_idx]],
                                 color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
                        mid_eta = (truth_eta[t_idx] + pred_eta[p_idx]) / 2
                        mid_phi = (truth_phi[t_idx] + pred_phi[p_idx]) / 2
                        plt.text(mid_eta, mid_phi, str(t_idx), fontsize=6, color='grey')
            except Exception as e:
                print(f"Warning: failed to draw matching lines: {e}")

            plt.savefig(os.path.join(self.output_dir, f"event_{event_idx}_etaphi.png"))
            plt.close()
            
            # Plot cells by layer with topocluster coloring
            self.plot_cells_by_layer(datamodule, cells, topoclusters, eta_offset, phi_offset, event_idx, pred_eta, pred_phi, truth_eta, truth_phi, pred_cls, truth_cls)

    def plot_cells_by_layer(self, datamodule,cells, topoclusters, centre_eta, centre_phi, event_idx, pred_eta, pred_phi, truth_eta, truth_phi,pred_cls,truth_cls):
        """Plot cells by layer with colors representing their topocluster assignment"""
        # Extract the features and names from cells
        cell_features = cells['features']
        cell_feat_names = cells['names']
        # Convert to numpy if tensor
        if hasattr(cell_features, 'detach'):
            cell_features = cell_features.detach().cpu().numpy()
        
        # Find indices for relevant features
        layer_idx = cell_feat_names.index('cell_layer') if 'cell_layer' in cell_feat_names else None
        eta_idx = cell_feat_names.index('cell_deltaeta') if 'cell_deltaeta' in cell_feat_names else None
        phi_idx = cell_feat_names.index('cell_deltaphi') if 'cell_deltaphi' in cell_feat_names else None
        topo_idx = cell_feat_names.index('cell_topo_idx') if 'cell_topo_idx' in cell_feat_names else None
        energy_idx = cell_feat_names.index('cell_e') if 'cell_e' in cell_feat_names else None

        # On tente d'utiliser le normalizer du test_dataset, sinon val_dataset
        dataset = getattr(datamodule, "test_dataset", None)
        if dataset is None:
            dataset = getattr(datamodule, "val_dataset", None)
        if dataset is None or not hasattr(dataset, "normalizer"):
            raise ValueError("Impossible de trouver un normalizer dans le datamodule.")
        grp_key = dataset.var2grp.get('cell_deltaeta', 'cell_deltaeta')
        cell_features[:, eta_idx] = dataset.normalizer.denormalize(cell_features[:, eta_idx], name=grp_key)
        grp_key = dataset.var2grp.get('cell_deltaphi', 'cell_deltaphi')
        cell_features[:, phi_idx] = dataset.normalizer.denormalize(cell_features[:, phi_idx], name=grp_key)

        # Check if required features are available
        if None in (layer_idx, eta_idx, phi_idx):
            print("Warning: Required cell features not found. Skipping cell layer plots.")
            return
        
        # Create a figure with subplots for each layer (0-5)
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        
        # Constants for cell size representation
        square_size = 0.01  # Size of cell squares
        
        # Plot cells for each layer
        for layer in range(6):  # Layers 0-5
            ax = axs[layer]
            
            # Filter cells for this layer
            layer_mask = cell_features[:, layer_idx].astype(int) == layer
            if not any(layer_mask):
                ax.text(0.5, 0.5, f"No cells in layer {layer}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(f"Layer {layer}")
                continue
            
            # Get cell positions (relative to center)
            cell_eta = centre_eta + np.copy(cell_features[layer_mask, eta_idx])
            cell_phi = centre_phi + np.copy(cell_features[layer_mask, phi_idx])
            # Calculate delta eta for points with the same phi coordinate
            unique_phis = np.unique(cell_phi)
            delta_etas = []

            for phi in unique_phis:
                eta_values = cell_eta[cell_phi == phi]
                if len(eta_values) > 1:
                    delta_eta = np.min(np.diff(np.sort(eta_values)))
                    delta_etas.append(delta_eta)

            if delta_etas:
                square_size = np.min(delta_etas)
            else:
                square_size = 0.1  # Default value if no delta eta can be calculated
            # Get topocluster assignment if available
            if topo_idx is not None:
                topo_ids = cell_features[layer_mask, topo_idx].astype(int)
                unique_topos = np.unique(topo_ids)
                n_topos = len(unique_topos)
                
                # Create colormap for topoclusters
                norm = Normalize(vmin=min(unique_topos), vmax=max(unique_topos))
                
                # Plot cells as squares with colors by topocluster
                for i, (eta, phi, topo_id) in enumerate(zip(cell_eta, cell_phi, topo_ids)):
                    color = self.cmap(norm(topo_id % 20))  # Cycle through colors if more than 20 topoclusters
                    rect = patches.Rectangle(
                        (eta-square_size/2, phi-square_size/2), 
                        square_size, square_size,
                        color=color, alpha=0.7)
                    ax.add_patch(rect)
            else:
                # If no topocluster information, use a single color
                for eta, phi in zip(cell_eta, cell_phi):
                    rect = patches.Rectangle(
                        (eta-square_size/2, phi-square_size/2), 
                        square_size, square_size,
                        color='blue', alpha=0.7)
                    ax.add_patch(rect)
            
            for i in range(5):
                # Plot predicted
                mask = pred_cls == i
                ax.scatter(pred_eta[mask], pred_phi[mask], c=self.colors[i], marker=self.markers_pred[i], label=f'Pred {self.labels[i]}', alpha=0.7)
                # Plot truth
                mask_t = truth_cls == i
                ax.scatter(truth_eta[mask_t], truth_phi[mask_t], c=self.colors[i], marker=self.markers_truth[i], label=f'True {self.labels[i]}', alpha=0.7)
            # Set plot limits
            delta = 0.4
            ax.set_xlim(centre_eta - delta, centre_eta + delta)
            ax.set_ylim(centre_phi - delta, centre_phi + delta)
            
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("η")
            ax.set_ylabel("φ")
            ax.grid(True)
        
        
        plt.savefig(os.path.join(self.output_dir, f"event_{event_idx}_cells_by_layer.png"))
        plt.close()
        
    def save_pt_comparison(self, event, p4_names, event_idx):
        """Save pt comparison between predicted and truth values"""
        # Find the index of pt/logpt in p4_names
        pt_idx = None
        for i, name in enumerate(p4_names):
            if 'pt' in name.lower():
                pt_idx = i
                break
        
        if pt_idx is None:
            print("Warning: Could not find pt feature in p4 variables. Skipping pt comparison.")
            return
        
        pred_class = event["pred_class"]  # (n_queries,)
        pred_boxes = event["pred_boxes"]  # (n_queries, 4 or more)
        truth_labels = event["truth_labels"] 
        truth_boxes = event["truth_boxes"]
        
        # Filter out non-object predictions (class 5)
        valid_pred_mask = (pred_class >= 0) & (pred_class <= 4)
        valid_truth_mask = (truth_labels >= 0) & (truth_labels <= 4)
        
        # Get predicted and truth pt values
        pred_pt = pred_boxes[valid_pred_mask, pt_idx]
        truth_pt = truth_boxes[valid_truth_mask, pt_idx]
        pred_types = pred_class[valid_pred_mask]
        truth_types = truth_labels[valid_truth_mask]
        
        # Convert to particle type labels
        particle_types = ['muon_charged', 'electron_charged', 'muon', 'neutral_hadron', 'photon']
        pred_type_labels = [particle_types[int(t)] for t in pred_types]
        truth_type_labels = [particle_types[int(t)] for t in truth_types]
    
        
        # Create DataFrame for predicted values
        pred_df = pd.DataFrame({
            'PT': pred_pt,
            'Type': pred_type_labels,
            'Source': ['Predicted'] * len(pred_pt)
        })
        
        # Create DataFrame for truth values
        truth_df = pd.DataFrame({
            'PT': truth_pt,
            'Type': truth_type_labels,
            'Source': ['Truth'] * len(truth_pt)
        })
        
        # Combine and save to CSV
        combined_df = pd.concat([pred_df, truth_df])
        csv_path = os.path.join(self.output_dir, f"event_{event_idx}_pt_comparison.csv")
        combined_df.to_csv(csv_path, index=False)
        
class MainProcessor:
    def __init__(self, config_path, cuda_visible_device, checkpoint_path, seed, outputdir, conf_threshold, num_event_display, specific_events=None):
        self.config_path = config_path
        self.cuda_visible_device = cuda_visible_device
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.output_dir = outputdir
        self.conf_threshold = conf_threshold
        self.num_event_display = num_event_display
        
        # HARD-CODED OPTION: Uncomment and modify this list to display specific events
        # If left as None, will display the first num_event_display events
        self.specific_events = specific_events or [
            # List event indices to display, e.g., events with large jet pT errors
            0,  
            1,  
            2,  
            3,  
            4,  
            5,  
            6,  
            7,  
            8,  
            9, 
        ]
        
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as fp:
            return yaml.safe_load(fp)

    def setup_environment(self):
        return setup_environment(
            self.config,
            cuda_visible_device=self.cuda_visible_device,
            seed=self.seed,
        )

    def get_checkpoint_path(self):
        basename = os.path.basename(self.config_path).removesuffix(".yaml")
        lightning_logdir = f"./workspace/train/{basename}"

        if self.checkpoint_path is None:
            checkpoint_path = get_latest_checkpoints(lightning_logs=f"{lightning_logdir}")
        else:
            checkpoint_path = self.checkpoint_path
        if (checkpoint_path is None) or (not os.path.exists(checkpoint_path)):
            raise ValueError("Invalid checkpoint_path:", checkpoint_path)
        return checkpoint_path

    def prepare_output_dir(self):
        outputdir = Path(self.output_dir)
        outputdir.mkdir(parents=True, exist_ok=True)
        return outputdir

    def run(self):
        ngpus = self.setup_environment()
        
        # If specific events are requested, increase dataset size to make sure those events are included
        if self.specific_events and len(self.specific_events) > 0:
            max_requested_event = max(self.specific_events)
            # Ensure we have enough events in the dataset to include the largest requested event index
            self.config["dataset"]["num_events_test"] = max(max_requested_event + 1, self.num_event_display)
            print(f"Requesting {self.config['dataset']['num_events_test']} events to include event indices: {self.specific_events}")
        else:
            self.config["dataset"]["num_events_test"] = self.num_event_display
            
        self.config["dataset"]["batchsize"] = min(100, self.config["dataset"]["num_events_test"])  # Use reasonable batch size

        checkpoint_path = self.get_checkpoint_path()
        outputdir = self.prepare_output_dir()
        outputfilepath = os.path.join(outputdir, f"{os.path.basename(self.config_path).removesuffix('.yaml')}.npz")
        print("PFlow result will be stored in \n", outputfilepath)

        net = DETRLightningModule(self.config)
        datamodule = PFlowDataModule(self.config)

        trainer = Trainer(
            max_epochs=self.config["training"]["num_epochs"],
            accelerator="gpu" if ngpus > 0 else "cpu",
            default_root_dir=f"./workspace/train/{os.path.basename(self.config_path).removesuffix('.yaml')}",
            use_distributed_sampler=False,
            log_every_n_steps=50,
            logger=False,
        )

        predictions = trainer.predict(net, datamodule=datamodule, ckpt_path=checkpoint_path)
        input_data = datamodule.test_dataset

        results = process_predictions(predictions)

        # Compute pred_class from pred_logits with thresholding
        for event in results:
            logits = event["pred_logits"]  # shape: (n_queries, n_classes)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            max_probs = np.max(probs, axis=1)
            argmax_class = np.argmax(probs, axis=1)
            pred_class = argmax_class.copy()
            mask = ((argmax_class >= 3) & (argmax_class <= 5) & (max_probs < self.conf_threshold))
            pred_class[mask] = 5

            event["pred_class"] = pred_class  # shape: (n_queries,)
        
        # Apply denormalization to results
        print("\nApplying denormalization to prediction results...")
        results = denormalize_data(results, datamodule)
        print("Denormalization completed.")

        eventplotter = EventPlotter(outputdir)
        
        # Filter results to only include specific events if requested
        if self.specific_events and len(self.specific_events) > 0:
            # Check if we have enough events to satisfy the request
            if max(self.specific_events) >= len(results):
                print(f"Warning: Requested event {max(self.specific_events)} but only have {len(results)} events")
                print(f"Will only display available events from the requested list")
                valid_specific_events = [idx for idx in self.specific_events if idx < len(results)]
            else:
                valid_specific_events = self.specific_events
                
            if valid_specific_events:
                print(f"Displaying only specific events: {valid_specific_events}")
                filtered_results = [results[idx] for idx in valid_specific_events]
                filtered_input_data = [input_data[idx] for idx in valid_specific_events]
                eventplotter.plot_events(filtered_results, filtered_input_data, datamodule, event_indices=valid_specific_events)
            else:
                print("No valid specific events to display. Displaying first {self.num_event_display} events instead.")
                eventplotter.plot_events(results[:self.num_event_display], input_data[:self.num_event_display], datamodule, event_indices=list(range(self.num_event_display)))
        else:
            # Display the first num_event_display events
            eventplotter.plot_events(results[:self.num_event_display], input_data[:self.num_event_display], datamodule, event_indices=list(range(self.num_event_display)))

@click.command()
@click.option("--config_path", type=str, default="configs/origial_fulltransformer.yaml")
@click.option("--cuda_visible_device", type=str, default="0")
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--seed", help="random seed", type=int, default=None)
@click.option('--outputdir', type=str, required=True)
@click.option('--conf-threshold', type=float, default=0.1, show_default=True, help='Confidence threshold: predictions below are set to class 5 (non-object)')
@click.option('--num_event_display', type=int, default=10, show_default=True, help='Number of events to display')
def main(**args):
    processor = MainProcessor(**args)
    processor.run()

if __name__ == "__main__":
    main()
