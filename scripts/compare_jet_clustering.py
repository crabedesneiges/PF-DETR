import fastjet
import vector
import awkward as ak
from particle import Particle
from tqdm import tqdm
import click
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import math

def ptetaphim_to_cartesian(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    E  = math.sqrt(mass*mass + px*px + py*py + pz*pz)
    px, py, pz = map(float, (px, py, pz))
    return px, py, pz, E

def deltaR(j1, j2):
    dphi = np.mod(j1.phi() - j2.phi() + np.pi, 2 * np.pi) - np.pi
    deta = j1.eta() - j2.eta()
    return np.sqrt(dphi**2 + deta**2)


class DetrDataReader:
    def __init__(self, filename):
        data = np.load(filename)
        self.truth_labels = data['truth_labels']  # (n_events, n_truth)
        self.truth_boxes = data['truth_boxes']    # (..., 3) (pt, eta, phi)

        self.pred_logits = data['pred_logits']  # (n_events, n_queries, n_classes)
        self.pred_boxes = data['pred_boxes']    # (..., 3)
        self.pred_classes = data['pred_class']  # (n_events, n_queries)

        # Check if the boxes have already been denormalized during evaluation
        is_denormalized = data.get('is_denormalized', False)
        if is_denormalized:
            print("Data has already been denormalized during evaluation, skipping denormalization.")
        else:
            print("[WARNING] Data might need to be denormalized...")

        self.truth_pt = self.truth_boxes[...,0] / 1000  # GeV
        self.pred_pt = self.pred_boxes[...,0] / 1000  # GeV

        self.truth_eta = self.truth_boxes[...,1]
        self.pred_eta = self.pred_boxes[...,1]
        self.truth_phi = self.truth_boxes[...,2]
        self.pred_phi = self.pred_boxes[...,2]
        print('pred_pt:', self.pred_pt.shape, 'NaN:', np.isnan(self.pred_pt).sum(), 'Inf:', np.isinf(self.pred_pt).sum())

class HgpDataReader:
    def __init__(self, filename, matching="original"): # Matching option from original script
        data = np.load(filename)

        self.truth_class_orig = data["truth_class"] # Keep original shape for potential per-event processing
        self.pflow_class_orig = data["pflow_class"]

        self.truth_labels = data["truth_class"]
        self.pred_classes = data["pflow_class"]

        self.is_not_dummy = self.truth_labels != -999.0

        self.is_charged_truth = (data["truth_has_track"] == 1)
        self.is_neutral_truth = np.logical_not(self.is_charged_truth)
        
        # Calculate track_pt (sum of pT of nodes connected to a truth particle, if node is a track)
        # This requires truth_inc, node_pt, node_is_track from the HGP npz file
        if 'truth_inc' in data and 'node_pt' in data and 'node_is_track' in data:
            # Assuming truth_inc: (n_events, n_truth_particles, n_nodes)
            # node_pt: (n_events, n_nodes)
            # node_is_track: (n_events, n_nodes)
            # Ensure correct broadcasting and dimensions if they differ.
            # This is a simplified interpretation. Original might be more complex.
            node_pt_gev = data["node_pt"] / 1000.0
            track_pt_contributions = data["truth_inc"] * data["node_is_track"][:, np.newaxis, :] * node_pt_gev[:, np.newaxis, :]
            self.track_pt = track_pt_contributions.sum(axis=2).flatten()
        else:
            print("[WARNING] HGP Data: 'truth_inc', 'node_pt', or 'node_is_track' not found. Track pT will be zero.")
            self.track_pt = np.zeros_like(self.truth_labels, dtype=float)


        self.truth_pt = data["truth_pt"] / 1000
        self.pred_pt = data["pflow_pt"] / 1000
        self.pred_eta = data["pflow_eta"]
        self.truth_eta = data["truth_eta"]
        self.pred_phi = data["pflow_phi"]
        self.truth_phi = data["truth_phi"]

        
        self.pred_phi = np.where(self.pred_phi > np.pi, self.pred_phi - 2 * np.pi, self.pred_phi)
        self.pred_phi = np.where(self.pred_phi < -np.pi, self.pred_phi + 2 * np.pi, self.pred_phi)

        self.pred_indicator = data["pred_ind"]
        self.truth_indicator = data["truth_ind"]
        
        # Mask for valid (non-dummy) HGP truth particles that also have a positive truth_indicator
        self.hgp_mask_truth_valid_with_indicator = np.logical_and(self.is_not_dummy, self.truth_indicator.squeeze() > 0.5)
        # Mask for valid (non-dummy) HGP predicted particles that also have a positive pred_indicator
        self.hgp_mask_pred_valid_with_indicator = np.logical_and(self.is_not_dummy, self.pred_indicator.squeeze() > 0.5) # is_not_dummy helps align lengths

        print(f'HGP Data: truth_pt shape: {self.truth_pt.shape}')
        print(f'HGP Data: pred_pt shape: {self.pred_pt.shape}')
        print(f'HGP Data: pred_eta shape: {self.pred_eta.shape}')
        print(f'HGP Data: pred_phi shape: {self.pred_phi.shape}')
        print(f'HGP Data: truth_eta shape: {self.truth_eta.shape}')
        print(f'HGP Data: truth_phi shape: {self.truth_phi.shape}')

def cluster_jets(pred, truth, pt_pred, pt_truth, eta_pred, eta_truth, phi_pred, phi_truth):
    """
    Fonction de clustering améliorée avec gestion d'erreurs.
    Corrige le bug d'indexation de la version originale.
    """
    vector.register_awkward()
    cluster_list_pred = []
    cluster_list_truth = []
    
    n_events = pred.shape[0]
    n_queries = pred.shape[1]


    for ievt in tqdm(range(n_events)):
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        
        # Build a list of valid particles for this event
        particles_pred = []
        particles_truth = []

        # CORRECTION DU BUG : séparation des boucles pour pred et truth
        # Boucle pour les prédictions
        for pred_idx in range(n_queries):
            if pred[ievt][pred_idx] != 5 and pred[ievt][pred_idx] != -999.0:  # Classe valide
                try:
                    cartesian = ptetaphim_to_cartesian(
                        pt_pred[ievt][pred_idx], 
                        eta_pred[ievt][pred_idx], 
                        phi_pred[ievt][pred_idx], 
                        0.0
                    )
                    particle = fastjet.PseudoJet(cartesian[0], cartesian[1], cartesian[2], cartesian[3])
                    if particle.pt() > 1.0:  # Seuil minimum
                        particles_pred.append(particle)
                except Exception as e:
                    print(f"Warning: Error creating predicted particle in event {ievt}, index {pred_idx}: {e}")
                    continue
        
        # Boucle pour les vérités terrain  
        for truth_idx in range(min(n_queries, truth.shape[1])):  # Sécurité sur les dimensions
            if truth[ievt][truth_idx] != 5 and truth[ievt][truth_idx] != -999.0:  # Classe valide
                try:
                    cartesian = ptetaphim_to_cartesian(
                        pt_truth[ievt][truth_idx], 
                        eta_truth[ievt][truth_idx], 
                        phi_truth[ievt][truth_idx], 
                        0.0
                    )
                    particle = fastjet.PseudoJet(cartesian[0], cartesian[1], cartesian[2], cartesian[3])
                    if particle.pt() > 1.0:  # Seuil minimum
                        particles_truth.append(particle)
                except Exception as e:
                    print(f"Warning: Error creating truth particle in event {ievt}, index {truth_idx}: {e}")
                    continue
        
        # Clustering avec gestion d'erreurs
        try:
            if len(particles_pred) > 0:
                cluster_pred = fastjet.ClusterSequence(particles_pred, jetdef)
            else:
                cluster_pred = fastjet.ClusterSequence([], jetdef)  # Séquence vide
            cluster_list_pred.append(cluster_pred)
        except Exception as e:
            print(f"Warning: Clustering failed for predicted particles in event {ievt}: {e}")
            cluster_list_pred.append(fastjet.ClusterSequence([], jetdef))

        try:
            if len(particles_truth) > 0:
                cluster_truth = fastjet.ClusterSequence(particles_truth, jetdef)
            else:
                cluster_truth = fastjet.ClusterSequence([], jetdef)  # Séquence vide
            cluster_list_truth.append(cluster_truth)
        except Exception as e:
            print(f"Warning: Clustering failed for truth particles in event {ievt}: {e}")
            cluster_list_truth.append(fastjet.ClusterSequence([], jetdef))

    return cluster_list_pred, cluster_list_truth

def match_truth_to_reco_jets(jets_truth, jets_pred, deltaR_max=0.1):
    matched_residuals = {'pt': [], 'eta': [], 'phi': []}
    truth_matched_props = {'pt': [], 'eta': [], 'phi': []}
    matched_count = 0
    total_selected_truth = 0
    pred_pts = []

    for jets_t, jets_p in zip(jets_truth, jets_pred):
        pred_pts.extend([jet.pt() for jet in jets_p if jet.pt() > 20])
        # Select up to two leading truth jets with pT > 10 GeV
        selected_truth_jets = [jet for jet in jets_t if jet.pt() > 10]
        selected_truth_jets = sorted(selected_truth_jets, key=lambda j: j.pt(), reverse=True)[:2]
        total_selected_truth += len(selected_truth_jets)
        

        matched_preds = set()

        for jet_truth in selected_truth_jets:
            best_match = None
            best_deltaR = float('inf')
            for i, jet_pred in enumerate(jets_p):
                if i in matched_preds:
                    continue
                dr = deltaR(jet_truth, jet_pred)
                if dr < best_deltaR and dr <= deltaR_max:
                    best_deltaR = dr
                    best_match = i
            if best_match is not None:
                matched_preds.add(best_match)
                jet_pred = jets_p[best_match]
                matched_count += 1

                # Residuals
                pt_res = (jet_pred.pt() - jet_truth.pt()) / jet_truth.pt()
                eta_res = jet_pred.eta() - jet_truth.eta()
                phi_res = np.mod(jet_pred.phi() - jet_truth.phi() + np.pi, 2*np.pi) - np.pi

                matched_residuals['pt'].append(pt_res)
                matched_residuals['eta'].append(eta_res)
                matched_residuals['phi'].append(phi_res)

                truth_matched_props['pt'].append(jet_truth.pt())
                truth_matched_props['eta'].append(jet_truth.eta())
                truth_matched_props['phi'].append(jet_truth.phi())

    matching_efficiency = matched_count / total_selected_truth if total_selected_truth > 0 else 0.0

    return matched_residuals, truth_matched_props, pred_pts, matching_efficiency

def plot_model_comparison_2(jets_truth_model1, jets_pred_model1, jets_truth_model2, jets_pred_model2, outputdir, model1_name = "model1", model2_name = "model2", deltaR_matching=0.1):
    """Generate comparison plots between two models and truth jets."""
    color_map = {
    'truth': 'black',
    model1_name: '#1f77b4',  # blue
    model2_name: '#d62728',  # red
}
    # Collect jet properties
    truth_pts_model1 = []
    truth_pts_model2 = []

    pred_model1_pts = []
    pred_model2_pts = []

    matched_residuals_model1, truth_matched_props_model1, pred_model1_pts, eff_model1 = match_truth_to_reco_jets(jets_truth_model1, jets_pred_model1)
    matched_residuals_model2, truth_matched_props_model2, pred_model2_pts, eff_model2 = match_truth_to_reco_jets(jets_truth_model2, jets_pred_model2)
    print(f"Matching efficiency for {model1_name}: {eff_model1:.2%}")
    print(f"Matching efficiency for {model2_name}: {eff_model2:.2%}")
    
    # Define pT bins for resolution analysis
    pt_bins = [20, 50, 100, 200]  # Added 0 and 1000 as boundary bins
    pt_bin_centers = []
    pt_resolution_model1 = []
    pt_resolution_model2 = []
    pt_bias_model1 = []
    pt_bias_model2 = []
    
    # Calculate pT-binned resolution
    for i in range(len(pt_bins)-1):
        pt_low, pt_high = pt_bins[i], pt_bins[i+1]
        pt_bin_centers.append((pt_low + pt_high) / 2)
        
        # Model 1
        mask_m1 = np.array([(pt >= pt_low) and (pt < pt_high) for pt in truth_matched_props_model1['pt']])
        if np.sum(mask_m1) > 0:
            residuals_in_bin_m1 = np.array(matched_residuals_model1['pt'])[mask_m1]
            pt_resolution_model1.append(np.std(residuals_in_bin_m1))
            pt_bias_model1.append(np.mean(residuals_in_bin_m1))
        else:
            pt_resolution_model1.append(0)
            pt_bias_model1.append(0)
        
        # Model 2
        mask_m2 = np.array([(pt >= pt_low) and (pt < pt_high) for pt in truth_matched_props_model2['pt']])
        if np.sum(mask_m2) > 0:
            residuals_in_bin_m2 = np.array(matched_residuals_model2['pt'])[mask_m2]
            pt_resolution_model2.append(np.std(residuals_in_bin_m2))
            pt_bias_model2.append(np.mean(residuals_in_bin_m2))
        else:
            pt_resolution_model2.append(0)
            pt_bias_model2.append(0)
    
    # Create comparison plots - now with 4 rows instead of 3
    fig, axes = plt.subplots(4, 4, figsize=(20, 24))
    
    # Row 1: pT distributions and residuals
    # pT distributions
    bins = np.linspace(20, 300, 50)
    axes[0,0].hist(truth_pts_model1, bins=bins, alpha=0.7, label='Truth', density=True, color=color_map['truth'])
    #axes[0,0].hist(truth_pts_model2, bins=bins, alpha=0.7, label='Truth', density=True, color=color_map['truth'])
    axes[0,0].hist(pred_model1_pts, bins=bins, alpha=0.7, label=model1_name, density=True, color=color_map[model1_name])
    axes[0,0].hist(pred_model2_pts, bins=bins, alpha=0.7, label=model2_name, density=True, color=color_map[model2_name])
    axes[0,0].set_xlabel('Jet pT [GeV]')
    axes[0,0].set_ylabel('Normalized Count')
    axes[0,0].set_title('Jet pT Distributions')
    axes[0,0].legend()
    #axes[0,0].set_yscale('log')

    # Row 1: Number of particles per jet
    n_particles_truth = [len(jet.constituents()) for event in jets_truth_model1 for jet in event if jet.pt() > 20]
    n_particles_pred_model1 = [len(jet.constituents()) for event in jets_pred_model1 for jet in event if jet.pt() > 20]
    n_particles_pred_model2 = [len(jet.constituents()) for event in jets_pred_model2 for jet in event if jet.pt() > 20]
    bins_npart = np.arange(0, max(
        max(n_particles_truth, default=0),
        max(n_particles_pred_model1, default=0),
        max(n_particles_pred_model2, default=0),
    ) + 2) - 0.5
    axes[0,3].hist(n_particles_truth, bins=bins_npart, alpha=0.7, label='Truth', density=True, color=color_map['truth'])
    axes[0,3].hist(n_particles_pred_model1, bins=bins_npart, alpha=0.7, label=model1_name, density=True, color=color_map[model1_name])
    axes[0,3].hist(n_particles_pred_model2, bins=bins_npart, alpha=0.7, label=model2_name, density=True, color=color_map[model2_name])
    axes[0,3].set_xlabel('Number of particles per jet')
    axes[0,3].set_ylabel('Normalized Count')
    axes[0,3].set_title('Particles per Jet Distribution')
    axes[0,3].legend()
    #axes[0,3].set_yscale('log')
    
    # pT relative residuals
    if matched_residuals_model1['pt'] or matched_residuals_model2['pt']:
        bins_pt_res = np.linspace(-0.5, 0.5, 50)
        
        # model1
        if matched_residuals_model1['pt']:
            c_m1, edges = np.histogram(matched_residuals_model1['pt'], bins=bins_pt_res)
            c_m1_norm = c_m1 / c_m1.sum()
            bin_centers = (edges[:-1] + edges[1:]) / 2
            axes[0,1].step(bin_centers, c_m1_norm, where='mid', alpha=0.7,
                        label=f'{model1_name} (μ={np.mean(matched_residuals_model1["pt"]):.3f}, σ={np.std(matched_residuals_model1["pt"]):.3f})',
                        color=color_map[model1_name])
        
        # model2
        if matched_residuals_model2['pt']:
            c_m2, edges = np.histogram(matched_residuals_model2['pt'], bins=bins_pt_res)
            c_m2_norm = c_m2 / c_m2.sum()
            bin_centers = (edges[:-1] + edges[1:]) / 2
            axes[0,1].step(bin_centers, c_m2_norm, where='mid', alpha=0.7,
                        label=f'{model2_name} (μ={np.mean(matched_residuals_model2["pt"]):.3f}, σ={np.std(matched_residuals_model2["pt"]):.3f})',
                        color=color_map[model2_name])

        axes[0,1].set_xlabel('pT Relative Residual')
        axes[0,1].set_ylabel('Normalized Count')
        axes[0,1].set_title('pT Relative Residuals')
        axes[0,1].legend(loc='lower center')
        axes[0,1].grid(alpha=0.3)
        axes[0,1].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # pT residuals vs truth pT
    axes[0,2].scatter(truth_matched_props_model1['pt'], matched_residuals_model1['pt'], 
                     alpha=0.3, s=1, label=model1_name, color=color_map[model1_name])
    axes[0,2].scatter(truth_matched_props_model2['pt'], matched_residuals_model2['pt'], 
                     alpha=0.3, s=1, label=model2_name, color=color_map[model2_name])
    axes[0,2].set_xlabel('Truth pT [GeV]')
    axes[0,2].set_ylabel('pT Relative Residual')
    axes[0,2].set_title('pT Residuals vs Truth pT')
    axes[0,2].legend()
    axes[0,2].grid(alpha=0.3)
    axes[0,2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0,2].set_ylim(-0.5, 0.5)
    
    # Row 2: eta residuals
    if matched_residuals_model1['eta'] or matched_residuals_model2['eta']:
        # eta residual distributions
        bins_eta_res = np.linspace(-0.2, 0.2, 50)
        
        # model1
        if matched_residuals_model1['eta']:
            c_eta_m1, edges_eta = np.histogram(matched_residuals_model1['eta'], bins=bins_eta_res)
            c_eta_m1_norm = c_eta_m1 / c_eta_m1.sum()
            bin_centers_eta = (edges_eta[:-1] + edges_eta[1:]) / 2
            axes[1,0].step(bin_centers_eta, c_eta_m1_norm, where='mid', alpha=0.7,
                        label=f'{model1_name} (μ={np.mean(matched_residuals_model1["eta"]):.4f}, σ={np.std(matched_residuals_model1["eta"]):.4f})',
                        color=color_map[model1_name])
        
        # model2
        if matched_residuals_model2['eta']:
            c_eta_m2, edges_eta = np.histogram(matched_residuals_model2['eta'], bins=bins_eta_res)
            c_eta_m2_norm = c_eta_m2 / c_eta_m2.sum()
            bin_centers_eta = (edges_eta[:-1] + edges_eta[1:]) / 2
            axes[1,0].step(bin_centers_eta, c_eta_m2_norm, where='mid', alpha=0.7,
                        label=f'{model2_name} (μ={np.mean(matched_residuals_model2["eta"]):.4f}, σ={np.std(matched_residuals_model2["eta"]):.4f})',
                        color=color_map[model2_name])

        axes[1,0].set_xlabel('η Residual')
        axes[1,0].set_ylabel('Fraction of Entries')
        axes[1,0].set_title('η Residuals')
        axes[1,0].legend(fontsize=9)
        axes[1,0].grid(alpha=0.3)
        axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # eta residuals vs truth eta
        axes[1,1].scatter(truth_matched_props_model1['eta'], matched_residuals_model1['eta'], 
                     alpha=0.3, s=1, label=model1_name, color=color_map[model1_name])
        axes[1,1].scatter(truth_matched_props_model2['eta'], matched_residuals_model2['eta'], 
                     alpha=0.3, s=1, label=model2_name, color=color_map[model2_name])
        axes[1,1].set_xlabel('Truth η')
        axes[1,1].set_ylabel('η Residual')
        axes[1,1].set_title('η Residuals vs Truth η')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)
        axes[1,1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_ylim(-0.2, 0.2)
        
        # eta residuals vs truth pT
        axes[1,2].scatter(truth_matched_props_model1['pt'], matched_residuals_model1['eta'], 
                     alpha=0.3, s=1, label=model1_name, color=color_map[model1_name])
        axes[1,2].scatter(truth_matched_props_model2['pt'], matched_residuals_model2['eta'], 
                     alpha=0.3, s=1, label=model2_name, color=color_map[model2_name])
        axes[1,2].set_xlabel('Truth pT [GeV]')
        axes[1,2].set_ylabel('η Residual')
        axes[1,2].set_title('η Residuals vs Truth pT')
        axes[1,2].legend()
        axes[1,2].grid(alpha=0.3)
        axes[1,2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1,2].set_ylim(-0.2, 0.2)
        
        # Add pT binned counts histogram
        axes[1,3].hist([truth_matched_props_model1['pt'], truth_matched_props_model2['pt']], 
                      bins=pt_bins[1:-1], alpha=0.7, 
                      label=['Model 1 matches', 'Model 2 matches'],
                      color=[color_map[model1_name], color_map[model2_name]])
        axes[1,3].set_xlabel('Truth pT [GeV]')
        axes[1,3].set_ylabel('Number of matched jets')
        axes[1,3].set_title('Matched Jets per pT Bin')
        axes[1,3].legend()
        axes[1,3].set_xscale('log')
        axes[1,3].grid(alpha=0.3)
    
    # Row 3: phi residuals and pT-binned resolution
    if matched_residuals_model1['phi'] or matched_residuals_model2['phi']:
        # phi residual distributions
        bins_phi_res = np.linspace(-0.2, 0.2, 50)
        
        # model1
        if matched_residuals_model1['phi']:
            c_phi_m1, edges_phi = np.histogram(matched_residuals_model1['phi'], bins=bins_phi_res)
            c_phi_m1_norm = c_phi_m1 / c_phi_m1.sum()
            bin_centers_phi = (edges_phi[:-1] + edges_phi[1:]) / 2
            axes[2,0].step(bin_centers_phi, c_phi_m1_norm, where='mid', alpha=0.7,
                        label=f'{model1_name} (μ={np.mean(matched_residuals_model1["phi"]):.4f}, σ={np.std(matched_residuals_model1["phi"]):.4f})',
                        color=color_map[model1_name])
        
        # model2
        if matched_residuals_model2['phi']:
            c_phi_m2, edges_phi = np.histogram(matched_residuals_model2['phi'], bins=bins_phi_res)
            c_phi_m2_norm = c_phi_m2 / c_phi_m2.sum()
            bin_centers_phi = (edges_phi[:-1] + edges_phi[1:]) / 2
            axes[2,0].step(bin_centers_phi, c_phi_m2_norm, where='mid', alpha=0.7,
                        label=f'{model2_name} (μ={np.mean(matched_residuals_model2["phi"]):.4f}, σ={np.std(matched_residuals_model2["phi"]):.4f})',
                        color=color_map[model2_name])

        axes[2,0].set_xlabel('φ Residual')
        axes[2,0].set_ylabel('Fraction of Entries')
        axes[2,0].set_title('φ Residuals')
        axes[2,0].legend(fontsize=9)
        axes[2,0].grid(alpha=0.3)
        axes[2,0].axvline(0, color='black', linestyle='--', alpha=0.5)
                
        # phi residuals vs truth phi
        axes[2,1].scatter(truth_matched_props_model1['phi'], matched_residuals_model1['phi'], 
                    alpha=0.3, s=1, label=model1_name, color=color_map[model1_name])
        axes[2,1].scatter(truth_matched_props_model2['phi'], matched_residuals_model2['phi'], 
                    alpha=0.3, s=1, label=model2_name, color=color_map[model2_name])
        axes[2,1].set_xlabel('Truth φ')
        axes[2,1].set_ylabel('φ Residual')
        axes[2,1].set_title('φ Residuals vs Truth φ')
        axes[2,1].legend()
        axes[2,1].grid(alpha=0.3)
        axes[2,1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[2,1].set_ylim(-0.2, 0.2)

        # pT-binned resolution plot
        valid_bins = [i for i in range(len(pt_bin_centers)) if pt_resolution_model1[i] > 0 or pt_resolution_model2[i] > 0]
        if valid_bins:
            x_centers = [pt_bin_centers[i] for i in valid_bins]
            res_m1 = [pt_resolution_model1[i] for i in valid_bins]
            res_m2 = [pt_resolution_model2[i] for i in valid_bins]
            
            axes[2,2].plot(x_centers, res_m1, 'o-', label=f'{model1_name} Resolution', 
                          color=color_map[model1_name], linewidth=2, markersize=6)
            axes[2,2].plot(x_centers, res_m2, 's-', label=f'{model2_name} Resolution', 
                          color=color_map[model2_name], linewidth=2, markersize=6)
            axes[2,2].set_xlabel('pT Bin Center [GeV]')
            axes[2,2].set_ylabel('pT Resolution (σ)')
            axes[2,2].set_title('pT Resolution vs pT')
            axes[2,2].legend()
            axes[2,2].grid(alpha=0.3)
            
            
        # pT-binned bias plot
        if valid_bins:
            bias_m1 = [pt_bias_model1[i] for i in valid_bins]
            bias_m2 = [pt_bias_model2[i] for i in valid_bins]
            
            axes[2,3].plot(x_centers, bias_m1, 'o-', label=f'{model1_name} Bias', 
                          color=color_map[model1_name], linewidth=2, markersize=6)
            axes[2,3].plot(x_centers, bias_m2, 's-', label=f'{model2_name} Bias', 
                          color=color_map[model2_name], linewidth=2, markersize=6)
            axes[2,3].set_xlabel('pT Bin Center [GeV]')
            axes[2,3].set_ylabel('pT Bias (mean residual)')
            axes[2,3].set_title('pT Bias vs pT')
            axes[2,3].legend()
            axes[2,3].grid(alpha=0.3)
            axes[2,3].axhline(0, color='black', linestyle='--', alpha=0.5)
            

        # Summary statistics
        mean_truth_model1 = np.mean(truth_pts_model1) if truth_pts_model1 else 0
        mean_truth_model2 = np.mean(truth_pts_model2) if truth_pts_model2 else 0
        mean_model1 = np.mean(pred_model1_pts) if pred_model1_pts else 0
        mean_model2 = np.mean(pred_model2_pts) if pred_model2_pts else 0
        
    # Calculate residual statistics
    pt_res_stats = {}
    eta_res_stats = {}
    phi_res_stats = {}
    
    for key, data in [(model1_name, 'model1'), (model2_name, 'model2')]: 
        res_data = matched_residuals_model1 if data == 'model1' else matched_residuals_model2
        if res_data['pt']:
            pt_res_stats[key] = {
                'mean': np.mean(res_data['pt']),
                'std': np.std(res_data['pt']),
                'rms': np.sqrt(np.mean(np.array(res_data['pt'])**2))
            }
            eta_res_stats[key] = {
                'mean': np.mean(res_data['eta']),
                'std': np.std(res_data['eta']),
                'rms': np.sqrt(np.mean(np.array(res_data['eta'])**2))
            }
            phi_res_stats[key] = {
                'mean': np.mean(res_data['phi']),
                'std': np.std(res_data['phi']),
                'rms': np.sqrt(np.mean(np.array(res_data['phi'])**2))
            }
    
    # Row 4: Summary statistics and pT-binned resolution table
    summary_text = f"""Model Comparison Summary Statistics

Mean pT:
Truth:   {mean_truth_model1:.2f} GeV
Truth:   {mean_truth_model2:.2f} GeV
Model 1: {mean_model1:.2f} GeV
Model 2: {mean_model2:.2f} GeV

pT Residuals:
"""
    for key, stats in pt_res_stats.items():
        summary_text += f"  {key}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, RMS={stats['rms']:.4f}\n"

    summary_text += "\nη Residuals:\n"
    for key, stats in eta_res_stats.items():
        summary_text += f"  {key}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, RMS={stats['rms']:.4f}\n"

    summary_text += "\nφ Residuals:\n"
    for key, stats in phi_res_stats.items():
        summary_text += f"  {key}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, RMS={stats['rms']:.4f}\n"

    
    axes[3,0].text(0.05, 0.95, summary_text, transform=axes[3,0].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[3,0].set_xlim(0, 1)
    axes[3,0].set_ylim(0, 1)
    axes[3,0].axis('off')
    
    # pT-binned resolution table
    pt_table_text = "pT-Binned Resolution Analysis\n\n"
    pt_table_text += f"{'pT Range':<12} {'Model1 σ':<10} {'Model1 μ':<10} {'Model2 σ':<10} {'Model2 μ':<10}\n"
    pt_table_text += "-" * 60 + "\n"
    
    for i in range(len(pt_bins)-1):
        if pt_bins[i+1] < 1000:  # Skip the overflow bin
            pt_range = f"{pt_bins[i]}-{pt_bins[i+1]}"
            res1 = f"{pt_resolution_model1[i]:.4f}" if pt_resolution_model1[i] > 0 else "N/A"
            bias1 = f"{pt_bias_model1[i]:.4f}" if pt_resolution_model1[i] > 0 else "N/A"
            res2 = f"{pt_resolution_model2[i]:.4f}" if pt_resolution_model2[i] > 0 else "N/A"
            bias2 = f"{pt_bias_model2[i]:.4f}" if pt_resolution_model2[i] > 0 else "N/A"
            pt_table_text += f"{pt_range:<12} {res1:<10} {bias1:<10} {res2:<10} {bias2:<10}\n"
    
    axes[3,1].text(0.05, 0.95, pt_table_text, transform=axes[3,1].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=8)
    axes[3,1].set_xlim(0, 1)
    axes[3,1].set_ylim(0, 1)
    axes[3,1].axis('off')
    
    # Hide unused subplots
    axes[3,2].axis('off')
    axes[3,3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'{model1_name}_vs_{model2_name}_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(outputdir, f'{model1_name}_vs_{model2_name}_model_comparison.png')}")
    
    # Print pT-binned resolution summary to console
    print("\npT-Binned Resolution Summary:")
    print(f"{'pT Range':<12} {'Model1 σ':<10} {'Model1 μ':<10} {'Model2 σ':<10} {'Model2 μ':<10}")
    print("-" * 60)
    for i in range(len(pt_bins)-1):
        if pt_bins[i+1] < 1000:
            pt_range = f"{pt_bins[i]}-{pt_bins[i+1]}"
            res1 = f"{pt_resolution_model1[i]:.4f}" if pt_resolution_model1[i] > 0 else "N/A"
            bias1 = f"{pt_bias_model1[i]:.4f}" if pt_resolution_model1[i] > 0 else "N/A"
            res2 = f"{pt_resolution_model2[i]:.4f}" if pt_resolution_model2[i] > 0 else "N/A"
            bias2 = f"{pt_bias_model2[i]:.4f}" if pt_resolution_model2[i] > 0 else "N/A"
            print(f"{pt_range:<12} {res1:<10} {bias1:<10} {res2:<10} {bias2:<10}")

def process_hgp(hgp_inputfile):

    print(f"\nLoading HGP data from: {hgp_inputfile}")
    data_hgp = HgpDataReader(hgp_inputfile)
    
    print(f"Applying indicator threshold 0.5...")
    
    for event_idx in tqdm(range(data_hgp.pred_classes.shape[0])):

        pred_cl = data_hgp.pred_classes[event_idx]
        pred_cl[data_hgp.pred_indicator[event_idx].squeeze() < 0.5] = -999
        data_hgp.pred_classes[event_idx] = pred_cl 
    print("HGP filtered")
    cluster_list_pred_hgp, cluster_list_truth_hgp = cluster_jets(
        data_hgp.pred_classes, data_hgp.truth_labels,
        data_hgp.pred_pt, data_hgp.truth_pt,
        data_hgp.pred_eta, data_hgp.truth_eta,
        data_hgp.pred_phi, data_hgp.truth_phi
    )

    print("HGP filtered")
    # Extract jets with pT threshold
    pt_threshold = 20.0  # GeV
    jets_pred_hgp = []
    jets_truth_hgp = [] 


    for cluster_pred_hgp, cluster_truth_hgp in zip(cluster_list_pred_hgp, cluster_list_truth_hgp):
        # Predicted jets
        jets_pred_event_hgp = cluster_pred_hgp.inclusive_jets(pt_threshold)

        jets_pred_event_hgp = [jet for jet in jets_pred_event_hgp
                           if len(jet.constituents()) > 1]

        jets_pred_event_hgp = sorted(jets_pred_event_hgp, key=lambda x: x.pt(), reverse=True)
        jets_pred_hgp.append(jets_pred_event_hgp)
        
        # Truth jets  
        jets_truth_event_hgp = cluster_truth_hgp.inclusive_jets(pt_threshold)

        jets_truth_event_hgp = [jet for jet in jets_truth_event_hgp
                            if len(jet.constituents()) > 1]

        jets_truth_event_hgp = sorted(jets_truth_event_hgp, key=lambda x: x.pt(), reverse=True)
        jets_truth_hgp.append(jets_truth_event_hgp)
    
    print(f"Found {sum(len(jets) for jets in jets_pred_hgp)} predicted jets "
          f"and {sum(len(jets) for jets in jets_truth_hgp)} truth jets above {pt_threshold} GeV")   
    return data_hgp, jets_truth_hgp, jets_pred_hgp,cluster_list_pred_hgp, cluster_list_truth_hgp

def process_detr(detr_inputfile, detr_conf_threshold):

    print(f"\nLoading DETR data from: {detr_inputfile}")
    data_detr_raw = DetrDataReader(detr_inputfile)

    # Compute softmax probabilities from logits and apply threshold filtering
    print(f"Applying confidence threshold {detr_conf_threshold}...")
    logits = data_detr_raw.pred_logits
    pred_classes_filtered = np.zeros_like(data_detr_raw.pred_classes)
    total_filtered = 0

    for event_idx in tqdm(range(logits.shape[0])):
        # shape: (n_queries, n_classes)
        event_logits = logits[event_idx]
        
        # Compute softmax probabilities (stable computation)
        exp_logits = np.exp(event_logits - np.max(event_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get predicted class and its associated probability
        max_probs = np.max(probs, axis=1)
        argmax_class = np.argmax(probs, axis=1)
        pred_class = argmax_class.copy()
        mask = ((argmax_class >= 3) & (argmax_class <= 5) & (max_probs < detr_conf_threshold))
        # Apply confidence threshold
        pred_class[mask] = 5
        
        pred_classes_filtered[event_idx] = pred_class
        
        # Count filtered predictions
        low_conf_mask = max_probs < detr_conf_threshold
        total_filtered += np.sum(low_conf_mask)
    
    print(f"Applied confidence threshold {detr_conf_threshold}: {total_filtered} predictions filtered out")
    data_detr_raw.detr_pred_classes_final = pred_classes_filtered
    
    # Cluster jets
    print("Clustering jets...")
    cluster_list_pred_detr, cluster_list_truth_detr = cluster_jets(
        data_detr_raw.detr_pred_classes_final, data_detr_raw.truth_labels,
        data_detr_raw.pred_pt, data_detr_raw.truth_pt,
        data_detr_raw.pred_eta, data_detr_raw.truth_eta,
        data_detr_raw.pred_phi, data_detr_raw.truth_phi
    )

    # Extract jets with pT threshold
    pt_threshold = 20.0  # GeV
    jets_pred_detr = []
    jets_truth_detr = []
    
    for cluster_pred_detr, cluster_truth_detr in zip(cluster_list_pred_detr, cluster_list_truth_detr):
            # Predicted jets
            jets_pred_event_detr = cluster_pred_detr.inclusive_jets(pt_threshold)

            jets_pred_event_detr = [jet for jet in jets_pred_event_detr
                            if len(jet.constituents()) > 1]

            jets_pred_event_detr = sorted(jets_pred_event_detr, key=lambda x: x.pt(), reverse=True)
            jets_pred_detr.append(jets_pred_event_detr)
            
            # Truth jets  
            jets_truth_event_detr = cluster_truth_detr.inclusive_jets(pt_threshold)

            jets_truth_event_detr = [jet for jet in jets_truth_event_detr
                                if len(jet.constituents()) > 1]

            jets_truth_event_detr = sorted(jets_truth_event_detr, key=lambda x: x.pt(), reverse=True)
            jets_truth_detr.append(jets_truth_event_detr)
        
    print(f"Found {sum(len(jets) for jets in jets_pred_detr)} predicted jets "
            f"and {sum(len(jets) for jets in jets_truth_detr)} truth jets above {pt_threshold} GeV")
    
    return data_detr_raw, jets_truth_detr, jets_pred_detr,cluster_list_pred_detr, cluster_list_truth_detr

@click.command()

@click.option('--model1_inputfile', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the .npz file for model 1.")
@click.option('--model2_inputfile', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the .npz file for model 2.")
@click.option('--outputdir', type=click.Path(file_okay=False, writable=True), required=True, help="Directory to save the combined plots.")
@click.option('--detr_conf_threshold', type=float, default=0.5, show_default=True, help='Confidence threshold for DETR predictions (predictions below are set to class 5/fake).')
@click.option('--n-bins', type=int, default=41, show_default=True,
              help='Number of pT bins for calibration')
@click.option('--model1_name', type=str, default='DETR', show_default=True,
              help='Name of the first model')
@click.option('--model2_name', type=str, default='HGP', show_default=True,
              help='Name of the second model')
def main(model1_inputfile, model2_inputfile, outputdir, detr_conf_threshold, n_bins, model1_name, model2_name):

    """
    Generates combined comparison plots for DETR and HGP models.
    """
    os.makedirs(outputdir, exist_ok=True)
    print(f"Output directory: {outputdir}")

    # --- Load Data ---
    print(f"\nLoading {model1_name} data from: {model1_inputfile}")
    # Detect which process function to use for each model
    def detect_and_process(model_name, inputfile, conf_threshold=None):
        if 'detr' in model_name.lower():
            return process_detr(inputfile, conf_threshold)
        elif 'hgp' in model_name.lower():
            return process_hgp(inputfile)
        else:
            raise ValueError(f"Unknown model type in name '{model_name}'. Must contain 'detr' or 'hgp'.")

    print(f"\nProcessing model1 ({model1_name}) from: {model1_inputfile}")
    if 'detr' in model1_name.lower():
        _, jets_truth_model1, jets_pred_model1, cluster_list_pred_model1, cluster_list_truth_model1 = detect_and_process(model1_name, model1_inputfile, detr_conf_threshold)
    else:
        _, jets_truth_model1, jets_pred_model1, cluster_list_pred_model1, cluster_list_truth_model1 = detect_and_process(model1_name, model1_inputfile)

    print(f"\nProcessing model2 ({model2_name}) from: {model2_inputfile}")
    if 'detr' in model2_name.lower():
        _, jets_truth_model2, jets_pred_model2, cluster_list_pred_model2, cluster_list_truth_model2 = detect_and_process(model2_name, model2_inputfile, detr_conf_threshold)
    else:
        _, jets_truth_model2, jets_pred_model2, cluster_list_pred_model2, cluster_list_truth_model2 = detect_and_process(model2_name, model2_inputfile)

    plot_model_comparison_2(jets_truth_model1, jets_pred_model1, jets_truth_model2, jets_pred_model2, outputdir, model1_name, model2_name)

    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()