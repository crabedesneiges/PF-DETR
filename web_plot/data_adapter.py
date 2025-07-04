# data_adapter.py
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
import numpy as np
class JETDetrDataReader:
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

class JETHgpDataReader:
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
class DetrDataReader:
    def __init__(self, filename):
        data = np.load(filename)
        self.truth_labels = data['truth_labels']  # (n_events, n_truth)
        self.truth_boxes = data['truth_boxes']    # (..., 3) (pt, eta, phi)
        
        self.pred_logits = data['pred_logits']  # (n_events, n_queries, n_classes)
        self.pred_boxes = data['pred_boxes']    # (..., 3)
        self.pred_classes = data['pred_class']  # (n_events, n_queries)
        # Check if the boxes have already been denormalized during evaluation
        # If not, we need to load the normalization parameters and denormalize here
        is_denormalized = data.get('is_denormalized', False)
        if is_denormalized:
            print("Data has already been denormalized during evaluation, skipping denormalization.")
        else:
            print("[ERROR]Data needs to be denormalized...")

        self.pred_classes_flat = self.pred_classes.flatten()
        self.truth_labels_flat = self.truth_labels.flatten()

        self.truth_pt = self.truth_boxes[...,0].flatten() / 1000  # GeV
        self.pred_pt = self.pred_boxes[...,0].flatten() / 1000  # GeV

        self.truth_eta = self.truth_boxes[...,1].flatten()
        self.pred_eta = self.pred_boxes[...,1].flatten()
        self.truth_phi = self.truth_boxes[...,2].flatten()
        self.pred_phi = self.pred_boxes[...,2].flatten()
        print('pred_pt:', self.pred_pt.shape, 'NaN:', np.isnan(self.pred_pt).sum(), 'Inf:', np.isinf(self.pred_pt).sum())

class HGPDataReader:
    def __init__(self, filename, matching="original"):
        # ================= #
        # === Load Data === #
        # ================= #
        data = np.load(filename)

        # ================= #
        # === Variables === #
        # ================= #
        self.truth_class = data["truth_class"].flatten()
        self.pflow_class = data["pflow_class"].flatten()

        self.is_not_dummy = self.truth_class != -999.0

        self.is_charged = (data["truth_has_track"] == 1).flatten()
        self.is_neutral = np.logical_not(self.is_charged)

        track_pt = (
            data["node_pt"][:, np.newaxis, :]
            * data["node_is_track"][:, np.newaxis, :]
            * data["truth_inc"]
        ).sum(axis=2)
        self.track_pt = track_pt.flatten() / 1000

        self.truth_pt = data["truth_pt"].flatten() / 1000
        self.pflow_pt = data["pflow_pt"].flatten() / 1000
        self.pflow_eta = data["pflow_eta"].flatten()
        self.truth_eta = data["truth_eta"].flatten()
        self.pflow_phi = data["pflow_phi"].flatten()
        self.truth_phi = data["truth_phi"].flatten()

        if matching == "custom":
            indices_sort = self.custom_matching(data, self.is_not_dummy)

            self.pflow_class = self.pflow_class[indices_sort]
            self.track_pt = self.track_pt[indices_sort]
            self.pflow_pt = self.pflow_pt[indices_sort]
            self.pflow_eta = self.pflow_eta[indices_sort]
            self.pflow_phi = self.pflow_phi[indices_sort]
        elif matching == "original":
            pass
        else:
            raise NotImplementedError()

        self.pflow_phi = np.where(
            self.pflow_phi > np.pi, self.pflow_phi - 2 * np.pi, self.pflow_phi
        )
        self.pflow_phi = np.where(
            self.pflow_phi < -np.pi, self.pflow_phi + 2 * np.pi, self.pflow_phi
        )

        # ==================================== #
        # === Incidence matrix / Indicator === #
        # ==================================== #
        def _extract_incidence_matrix(incidence_matrix, num_nodes):
            return [x[:, :n] for x, n in zip(incidence_matrix, num_nodes)]

        pred_indicator = data["pred_ind"]
        truth_indicator = data["truth_ind"]
        self.pred_indicator = pred_indicator.flatten()
        self.truth_indicator = truth_indicator.flatten()

        pred_incidence = _extract_incidence_matrix(data["pred_inc"], data["num_nodes"])
        truth_incidence = _extract_incidence_matrix(data["truth_inc"], data["num_nodes"])
        pred_incidence = np.concatenate([v.flatten() for v in pred_incidence], axis=-1)
        truth_incidence = np.concatenate([v.flatten() for v in truth_incidence], axis=-1)
        self.pred_incidence = pred_incidence.flatten()
        self.truth_incidence = truth_incidence.flatten()

        is_true_particle = self.is_truth_objects().reshape((-1, 30))
        is_true_particle = np.repeat(is_true_particle, data["num_nodes"], axis=0).flatten()
        self.is_true_particle = is_true_particle

    def custom_matching(selfm, data, is_not_dummy):
        indices_sort = np.empty(is_not_dummy.shape, dtype=np.int32)

        for ie, (pt_t, pt_r, eta_t, eta_r, phi_t, phi_r, ind_t, ind_r) in enumerate(
            zip(
                data["truth_pt"],
                data["pflow_pt"],
                data["truth_eta"],
                data["pflow_eta"],
                data["truth_phi"],
                data["pflow_phi"],
                data["truth_ind"],
                data["pred_ind"],
            )
        ):
            pt_r = np.tile(pt_r, (30, 1))
            pt_t = np.tile(pt_t.reshape(-1, 1), (1, 30))
            eta_r = np.tile(eta_r, (30, 1))
            eta_t = np.tile(eta_t.reshape(-1, 1), (1, 30))
            phi_r = np.tile(phi_r, (30, 1))
            phi_t = np.tile(phi_t.reshape(-1, 1), (1, 30))
            delta_pt = (pt_t - pt_r) / pt_t
            delta_eta = eta_t - eta_r
            delta_phi = np.mod(phi_t - phi_r + np.pi, 2 * np.pi) - np.pi
            d2 = delta_pt**2 + 25 * (delta_eta**2 + delta_phi**2)

            d2 = np.where(np.tile(ind_r.flatten(), (30, 1)) > 0.5, d2, 1e10)
            d2 = np.where(np.tile(ind_t.reshape(-1, 1), (1, 30)) > 0.5, d2, 1e10)

            indices = linear_sum_assignment(d2)[1]
            shift = 30 * ie
            indices_sort[shift : shift + 30] = indices + shift

        return indices_sort

    def mask_is_charged(self):
        mask = np.logical_and(self.is_charged, self.is_not_dummy)

        # Remove the events s.t. track pt = 0
        mask = np.logical_and(mask, self.track_pt != 0.0)

        return mask

    def is_truth_objects(self, classid=None):
        if classid is None:
            return np.logical_and(self.is_not_dummy, self.truth_indicator > 0.5)
        else:
            return np.logical_and(self.is_truth_objects(), self.truth_class == classid)

    def is_pflow_objects(self, classid=None):
        if classid is None:
            return np.logical_and(self.is_not_dummy, self.pred_indicator > 0.5)
        else:
            return np.logical_and(self.is_pflow_objects(), self.pflow_class == classid)

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

# --- Mocks pour les DataReaders (à remplacer par vos vrais fichiers) ---
def process_hgp(hgp_inputfile):

    print(f"\nLoading HGP data from: {hgp_inputfile}")
    data_hgp = JETHgpDataReader(hgp_inputfile)
    
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
    data_detr_raw = JETDetrDataReader(detr_inputfile)

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

# --- Copiez/Collez ici les fonctions de votre script ---
# J'ai légèrement modifié les fonctions pour qu'elles ne fassent plus d'impressions (print)
# afin de garder l'interface propre.

# Renommage des classes pour correspondre à votre script
DetrDataReader = DetrDataReader
HGPDataReader = HGPDataReader

# DETR class mapping
DETR_CLASS_MAP_VIS = {"charged_hadron": [0], "electron": [1], "muon": [2], "neutral_hadron": [3], "photon": [4], "padding": [5]}

# HGP class mapping
HGP_TO_DETR_LABEL_MAP = {0: 4, 1: 3, 2: 0, 3: 1, 4: 2}

CLASS_NAMES = ["charged_hadron", "electron", "muon", "neutral_hadron", "photon"]

def adapt_detr_data(data, conf_threshold=0.1):
    """Adapte les données DETR à un format commun."""
    truth = data.truth_labels_flat
    pred = data.pred_classes_flat.copy()
    pt_truth = data.truth_pt
    pt_pred = data.pred_pt

    logits_flat = data.pred_logits.reshape(-1, data.pred_logits.shape[-1])
    exp_l = np.exp(logits_flat - logits_flat.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    maxp = probs.max(axis=1)
    argmax_class = np.argmax(probs, axis=1)
    mask = ((argmax_class >= 3) & (argmax_class <= 5) & (maxp < conf_threshold))
    pred[mask] = 5

    mask_truth = truth != 5
    mask_pred = pred != 5
    mask_truth_and_pred = np.logical_and(mask_truth, mask_pred)
    
    mask_neutral_pred = pred > 2
    mask_truth_pred_neutral = np.logical_and(mask_truth_and_pred, mask_neutral_pred)
    mask_truth_pred_charged = np.logical_and(mask_truth_and_pred, np.logical_not(mask_neutral_pred))

    eta_truth = data.truth_eta
    eta_pred = data.pred_eta
    phi_truth = data.truth_phi
    phi_pred = data.pred_phi
    
    return {
        "truth": truth, "pred": pred, "pt_truth": pt_truth, "pt_pred": pt_pred,
        "eta_truth": eta_truth, "eta_pred": eta_pred, "phi_truth": phi_truth,
        "phi_pred": phi_pred, "mask_truth_and_pred": mask_truth_and_pred,
        "mask_pred": mask_pred, "mask_truth": mask_truth,
        "mask_truth_pred_neutral": mask_truth_pred_neutral,
        "mask_truth_pred_charged": mask_truth_pred_charged
    }


def adapt_hgp_data(data):
    """Adapte les données HGP à un format commun."""
    padding_idx = DETR_CLASS_MAP_VIS.get("padding", [5])[0]
    map_func = np.vectorize(lambda x: HGP_TO_DETR_LABEL_MAP.get(x, padding_idx))
    
    truth = map_func(data.truth_class)
    pred = map_func(data.pflow_class)
    pt_truth = data.truth_pt
    pt_pred = data.pflow_pt
    eta_truth = data.truth_eta
    eta_pred = data.pflow_eta
    phi_truth = data.truth_phi
    phi_pred = data.pflow_phi

    mask_pred = data.pred_indicator > 0.5
    mask_truth = data.truth_indicator > 0.5
    mask_truth_and_pred = np.logical_and(mask_truth, mask_pred)
    
    mask_neutral_pred = pred > 2 
    mask_truth_pred_neutral = np.logical_and(mask_truth_and_pred, mask_neutral_pred)
    mask_truth_pred_charged = np.logical_and(mask_truth_and_pred, np.logical_not(mask_neutral_pred))

    return {
        "truth": truth, "pred": pred, "pt_truth": pt_truth, "pt_pred": pt_pred,
        "eta_truth": eta_truth, "eta_pred": eta_pred, "phi_truth": phi_truth,
        "phi_pred": phi_pred, "mask_truth_and_pred": mask_truth_and_pred,
        "mask_pred": mask_pred, "mask_truth": mask_truth,
        "mask_truth_pred_neutral": mask_truth_pred_neutral,
        "mask_truth_pred_charged": mask_truth_pred_charged
    }