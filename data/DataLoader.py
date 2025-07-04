import hashlib
import inspect
import math
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import BatchSampler, Dataset
from yaml import safe_load


from .normalizer import Normalizer
from .ntuple_reader import NtupleReader



class PFlowDataset(Dataset):


    def __init__(
        self,
        filename='/data/multiai/data3/HyperGraph-2212.01328/singleQuarkJet_train.root',
        num_read=-1,
        cache=False,
        use_boolean_incident_matrix=False,
        isEval=False,
        build_topocluster=True,
        max_particles=30,
        normalization_path="",
        num_workers=None,
        add_vars_for_visualization=False,
        event_mask=None,
        is_private_sample=False,
        incidence_energy_threshold=None,
    ):
        self.cache = cache
        self.incidence_energy_threshold = incidence_energy_threshold
        self.use_boolean_incident_matrix = use_boolean_incident_matrix
        self.max_particles = max_particles
        self.add_vars_for_visualization = add_vars_for_visualization

        self.num_read = num_read


        # Read input variables list
        with open("data/input_variables.yaml") as f:
            self.input_variables = safe_load(f)

        self.normalizer = Normalizer(normalization_path)

        # Liste des variables à NE PAS normaliser (flags, index, etc.)
        self.no_norm_vars = [
            "cell_layer", "cell_topo_idx", "track_isMuon"
        ]

        self.reader = NtupleReader(
            filename=filename,
            num_read=num_read,
            input_variables=self.input_variables["rootfile"],
            isEval=isEval,
            build_topocluster=build_topocluster,
            normalization_path=normalization_path,
            event_mask=event_mask,
            is_private_sample=is_private_sample,
        )
        # Build var->group mapping for normalization
        self.normalization_groups = self.input_variables.get("normalization_groups", {})
        self.var2grp = {}
        for grp, var_list in self.normalization_groups.items():
            for v in var_list:
                self.var2grp[v] = grp

    def __len__(self):
        return len(self.reader)

    def count(self, var, idx=None):
        if not var.startswith("n_"):
            var = "n_" + var

        if idx is None:
            return self.reader.count_data[var]
        else:
            return self.reader.count_data[var][idx]

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def _cell(self, var, idx):
        return self.reader._cell(var, idx)

    def _track(self, var, idx):
        return self.reader._track(var, idx)

    def _topocluster(self, var, idx):
        return self.reader._topocluster(var, idx)

    def _particle(self, var, idx):
        return self.reader._particle(var, idx)

    def _node(self, var, idx):
        return self.reader._node(var, idx)

    def _global(self, var, idx):
        return self.reader._global(var, idx)

    def _meta(self, var, idx):
        return self.reader._meta(var, idx)

    def dump_normalization_params(self, outputname):
        # Collect raw arrays
        raw_targets = {}
        for grp in ["cells","tracks","topoclusters"]:
            for name in self.input_variables["graph"][grp]["node_features"]:
                name_raw = name.replace(":normed", "")
                raw_targets[name_raw] = self.reader.data[name_raw]
        for name in self.input_variables["graph"]["truths"]["p4"]:
            name_raw = name.replace(":normed", "")
            raw_targets[name_raw] = self.reader.data[name_raw]

        # Build grouped targets
        group_defs = self.input_variables.get("normalization_groups", {})
        grouped_targets = {}
        grouped_vars = set()
        for grp, var_list in group_defs.items():
            arrays = []
            for v in var_list:
                if v in raw_targets:
                    arrays.append(raw_targets[v].ravel())
                    grouped_vars.add(v)
            if arrays:
                grouped_targets[grp] = np.concatenate(arrays)
        # Add remaining individually
        for name_raw, arr in raw_targets.items():
            if name_raw not in grouped_vars:
                grouped_targets[name_raw] = arr if arr.ndim>1 else arr.ravel()

        self.reader.normalizer.fit_and_dump(grouped_targets, outputname)


    def get_single_item(self, idx):
        # === Input features === #
        # 1. Cells
        cell_feat_names = [name.replace(":normed", "") for name in self.input_variables["graph"]["cells"]["node_features"]]
        cell_feats = []
        for name in self.input_variables["graph"]["cells"]["node_features"]:
            name_raw = name.replace(":normed", "")
            arr = self._cell(name_raw, idx)
            if name_raw in self.no_norm_vars:
                arr = arr.astype(np.float32)
            else:
                key = self.var2grp.get(name_raw, name_raw)
                arr = self.normalizer.normalize(arr, key)
            cell_feats.append(arr)
        cell_feats = np.stack(cell_feats, axis=-1)
        cell_feats = torch.tensor(cell_feats).float()

        # 2. Tracks
        track_feat_names = [name.replace(":normed", "") for name in self.input_variables["graph"]["tracks"]["node_features"]]
        track_feats = []
        for name in self.input_variables["graph"]["tracks"]["node_features"]:
            name_raw = name.replace(":normed", "")
            arr = self._track(name_raw, idx)
            if name_raw in self.no_norm_vars:
                arr = arr.astype(np.float32)
            else:
                key = self.var2grp.get(name_raw, name_raw)
                arr = self.normalizer.normalize(arr, key)
            track_feats.append(arr)
        track_feats = np.stack(track_feats, axis=-1)
        track_feats = torch.tensor(track_feats).float()

        # 3. Topoclusters
        topo_feat_names = [name.replace(":normed", "") for name in self.input_variables["graph"]["topoclusters"]["node_features"]]
        topo_feats = []
        for name in self.input_variables["graph"]["topoclusters"]["node_features"]:
            name_raw = name.replace(":normed", "")
            arr = self._topocluster(name_raw, idx)
            if name_raw in self.no_norm_vars:
                arr = arr.astype(np.float32)
            else:
                key = self.var2grp.get(name_raw, name_raw)
                arr = self.normalizer.normalize(arr, key)
            topo_feats.append(arr)
        topo_feats = np.stack(topo_feats, axis=-1)
        topo_feats = torch.tensor(topo_feats).float()

        # === Target: Particles === #
        # p4 vector
        p4_vars = self.input_variables["graph"]["truths"]["p4"]
        p4_names = [name.replace(":normed", "") for name in p4_vars]
        p4 = []
        for name in p4_vars:
            # raw read
            name_raw = name.replace(":normed", "")
            arr = self.reader._particle(name_raw, idx)
            # normalization by group
            key = self.var2grp.get(name_raw, name_raw)
            arr = self.normalizer.normalize(arr, key)
            p4.append(arr)
        p4 = np.stack(p4, axis=-1)
        p4 = torch.tensor(p4).float()

        # class
        cls = self._particle("particle_class", idx)
        cls = torch.tensor(cls).long()

        # is_track (flag)
        is_track = self._particle("particle_has_track", idx)
        is_track = torch.tensor(is_track).float()

        global_eta = torch.tensor(self.reader._global("global_eta", idx)).float()
        global_phi = torch.tensor(self.reader._global("global_phi", idx)).float()

        incidence_matrix, ghost_particle_idx, missing_particle_idx = self.build_incidence_matrix(idx)
        return {
            "cells": {"features": cell_feats, "names": cell_feat_names},
            "tracks": {"features": track_feats, "names": track_feat_names},
            "topoclusters": {"features": topo_feats, "names": topo_feat_names},
            "target": {
                "p4": {"values": p4, "names": p4_names},
                "class": {"values": cls, "name": "particle_class"},
                "is_track": {"values": is_track, "name": "particle_has_track"},
                "inc": {
                    "incidence_matrix": incidence_matrix,
                    "ghost_particle_idx": ghost_particle_idx,
                    "missing_particle_idx": missing_particle_idx,
                },

            },
            "global_eta": global_eta,
            "global_phi": global_phi
        }

    def build_incidence_matrix(self, idx):
        n_cells = self.count("cells", idx)
        n_tracks = self.count("tracks", idx)
        n_topoclusters = self.count("topoclusters", idx)
        n_particles = self.count("particles", idx)
        n_pflows = self.max_particles
        n_nodes = n_topoclusters + n_tracks

        particle_to_node_idx = self._meta("meta_particle_to_node_idx", idx)
        particle_to_node_weight = self._meta("meta_particle_to_node_weight", idx)
        particle_has_track = self._particle("particle_has_track", idx)
        particle_dep_energy = self._particle("particle_dep_energy", idx)
        cell_topo_idx = self._cell("cell_topo_idx", idx)

        incidence = np.zeros((n_pflows, n_nodes))

        # Build incidence matrix
        for p_idx, (node_idx, node_weights, has_track) in enumerate(
            zip(particle_to_node_idx, particle_to_node_weight, particle_has_track)
        ):
            node_idx = np.array(node_idx, dtype=int)
            node_weights = np.array(node_weights)

            # ghost particles
            if len(node_idx) == 0:
                continue

            # assert
            if (has_track or node_idx[-1] >= n_cells) and node_weights[-1] != 0.5:
                raise ValueError("Something wrong... Please check node_weights property")

            # track attention
            if has_track:
                track_idx = node_idx[-1] - n_cells
                incidence[p_idx, track_idx + n_topoclusters] = 1

                # Remove tracks
                node_weights = node_weights[node_idx < n_cells]
                node_idx = node_idx[node_idx < n_cells]
                node_weights *= 2

            # topocluster attention
            bc = np.bincount(cell_topo_idx[node_idx], weights=node_weights)
            incidence[p_idx, : len(bc)] = bc

        # Renormalize incidence matrix with removing negative energy contribution
        if False:
            # charged particles with tracks but no deposited energy
            no_energy_dep_ch_mask = np.logical_and(
                particle_has_track,  # associated with tracks
                particle_dep_energy == 0.0,  # no energy deposited
            )
            particle_dep_energy[no_energy_dep_ch_mask] = 1  # 1 MeV  =? Why 1 MeV?

        particle_dep_energy_padded = np.zeros(n_pflows)
        particle_dep_energy_padded[:n_particles] = particle_dep_energy
        incidence = incidence * particle_dep_energy_padded.reshape(-1, 1)

        # Remove negative energy contribution
        incidence[incidence < 0] = 0

        # Remove negligible energy contribution
        if self.incidence_energy_threshold is not None:
            incidence[incidence < self.incidence_energy_threshold] = 0.0

        # Fake node treatment:
        # Assign the fake node to a ghost particle. They are ignored in the loss calculation
        missing_node_idx = np.where(incidence.sum(axis=0) == 0)[0]
        missing_particle_idx = np.where(incidence.sum(axis=1) == 0)[0]
        n_fake_nodes = len(missing_node_idx)
        if n_fake_nodes > 0:
            ghost_particle_idx = np.arange(n_fake_nodes) + n_particles
            incidence[ghost_particle_idx, missing_node_idx] = 1
        else:
            ghost_particle_idx = None

        # Normalize (sum_particles incidence = 1)
        incidence = incidence / incidence.sum(axis=0, keepdims=True)

        if self.use_boolean_incident_matrix is True:
            incidence = incidence > 0.01  # Why 0.01 ???

        return incidence, ghost_particle_idx, missing_particle_idx

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect PFlowDataset structure")
    parser.add_argument("--filename", "-f", default="/data/multiai/data3/HyperGraph-2212.01328/singleQuarkJet_train.root")
    parser.add_argument("--num_read", "-n", type=int, default=-1, help="Number of events to read (-1 for all)")
    parser.add_argument("--cache", action="store_true", help="Use cached data if available")
    args = parser.parse_args()

    ds = PFlowDataset(
        filename=args.filename,
        num_read=args.num_read,
        cache=args.cache,
        isEval=True,
        max_particles=30,
    )
    print(f"Dataset length (events): {len(ds)}")
    ds.build_incidence_matrix(0)
    print()
    # Inspect events to find max node counts
    n_inspect = len(ds) if args.num_read < 0 else min(len(ds), args.num_read)
    max_cells = max_tracks = max_topo = max_particles = 0
    for i in range(n_inspect):
        sample = ds[i]
        max_cells = max(max_cells, sample["cells"]["features"].shape[0])
        max_tracks = max(max_tracks, sample["tracks"]["features"].shape[0])
        max_topo = max(max_topo, sample["topoclusters"]["features"].shape[0])
        max_particles = max(max_particles, sample["target"]["p4"]["values"].shape[0])
    print(f"Max cells: {max_cells} | Max tracks: {max_tracks} | Max topoclusters: {max_topo} | Max particles: {max_particles}")

    # Display feature dims and names for a sample event
    sample = ds[0]
    dims = (
        sample["cells"]["features"].shape[1],
        sample["tracks"]["features"].shape[1],
        sample["topoclusters"]["features"].shape[1],
        sample["target"]["p4"]["values"].shape[1]
    )
    print("Feature dims (cells, tracks, topoclusters, p4):", *dims)
    print("Cell feature names:", sample["cells"]["names"])
    print("Track feature names:", sample["tracks"]["names"])
    print("Topocluster feature names:", sample["topoclusters"]["names"])
    print("Particle p4 names:", sample["target"]["p4"]["names"])
    print("Particle class name:", sample["target"]["class"]["name"])
    print("Particle is_track name:", sample["target"]["is_track"]["name"])