import glob

import numpy as np
import uproot
import vector

from .normalizer import Normalizer
from .pdgid_converter import charge_label, class_label


class NtupleReader:

    def __init__(
        self,
        filename="/home/saito/data/workspace/2212.01328/data/singleQuarkJet_train.root",
        num_read=-1,
        input_variables=None,
        isEval=False,
        build_topocluster=True,
        normalization_path="",
        add_vars_for_visualization=False,
        event_mask=None,
        is_private_sample=False,
    ):
        self.num_read = num_read
        self.input_variables = input_variables
        self.isEval = isEval
        self.event_mask = event_mask
        self.is_private_sample = is_private_sample
        self.add_vars_for_visualization = add_vars_for_visualization

        self.normalizer = Normalizer(normalization_path)

        # Modify branch name
        self.input_variables = {
            k: self._modify_branch_name(v) for k, v in self.input_variables.items()
        }

        # Open root file and ttree
        data = []
        num_events = 0
        for name in glob.glob(filename):
            if self.is_private_sample:
                name = name + ":Out_Tree"
            else:
                name = name + ":Low_Tree"

            tree = uproot.open(name)
            if self.num_read > 0:
                _data, n = self._read_tbranch(tree, self.num_read - num_events)
            else:
                _data, n = self._read_tbranch(tree, -1)

            num_events += n
            data += [_data]
            #print(name, n, tree.num_entries, num_events, self.num_read)
            if num_events >= self.num_read:
                break


        data = {k: np.concatenate([d[k] for d in data], axis=0) for k in data[0].keys()}

        if self.num_read < 0:
            self.num_read = num_events

        print("=== filtering / preprocessing ===")
        data = self._filtering(data)
        data = self._preprocessing(data, build_topocluster)
        self.data = data
        print("=== filtering / preprocessing : completed ===")

    def __len__(self):
        return self.num_read

    def clear(self):
        self.data = None
        self.count_data = None

    def _get_from_ttree(self, tree, var, num_read, copy=False):
        if num_read < 0:
            num_read = tree.num_entries
        array = tree[var].array(library="np", entry_stop=num_read)

        if copy is True:
            return np.copy(array)
        else:
            return array

    def _modify_branch_name(self, branch_names):
        if self.is_private_sample:
            return [v.removesuffix(":private") for v in branch_names if not v.endswith(":original")]
        else:
            return [v.removesuffix(":original") for v in branch_names if not v.endswith(":private")]

    def _read_tbranch(self, tree, num_read):

        data = {}
        num_events = set()

        # ================== #
        # === Read TTree === #
        # ================== #
        for var in self.input_variables["normal"]:
            data[var] = self._get_from_ttree(tree, var, num_read)
            num_events.add(len(data[var]))

        for var in self.input_variables["meta"]:
            data["meta_" + var] = self._get_from_ttree(tree, var, num_read)
            num_events.add(len(data["meta_" + var]))

        if len(num_events) != 1:
            raise ValueError("num_event is inconsistent in branchs.")

        num_events = list(num_events)[0]

        return data, num_events

    def _filtering(self, data):
        # ================= #
        # === Filtering === #
        # ================= #
        if self.event_mask is not None:
            for key in data:
                data[key] = data[key][self.event_mask]
            self.num_read = len(self.event_mask)

        if False:
            print("This filtering is for debugging !!!!!")
            num_particles = np.array([len(v) for v in data["particle_pdgid"]])
            mask = num_particles < 10
            for key in data:
                data[key] = data[key][mask]
            self.num_read = mask.sum()

        # Remove bad tracks
        if self.is_private_sample:
            is_good_tracks = [
                np.logical_and(reco >= 0, acc >= 0)
                for reco, acc in zip(data["track_reconstructed"], data["track_in_acceptance"])
            ]
        else:
            is_good_tracks = [not_reg == 0 for not_reg in data["track_not_reg"]]

        for var in self.input_variables["normal"]:
            if not var.startswith("track_"):
                continue
            data[var] = np.array(
                [np.array(v[mask]) for v, mask in zip(data[var], is_good_tracks)], dtype=object
            )

        # Remove no track events
        num_tracks = np.array([len(v) for v in data["track_qoverp"]])
        for key in data:
            data[key] = data[key][num_tracks > 0]
        self.num_read = (num_tracks > 0).sum()

        # Remove no cell events
        num_cells = np.array([len(v) for v in data["cell_e"]])
        for key in data:
            data[key] = data[key][num_cells > 0]
        self.num_read = (num_cells > 0).sum()

        # Remove bad incidence matrix events
        # mask = np.array(
        #     [
        #         len(ak.flatten(data["meta_particle_to_node_idx"][nev])) > 0
        #         for nev in range(self.num_read)
        #     ]
        # )
        # for key in data:
        #     data[key] = data[key][mask]
        # self.num_read = mask.sum()

        # === AJOUTEZ CE BLOC MIS À JOUR POUR FILTRER ET AFFICHER LES SUPPRESSIONS ===
        print("=== Filtrage des événements avec des indices de topocluster incohérents ===")

        if not all(isinstance(v, np.ndarray) for v in data["cell_topo_idx"]):
            data["cell_topo_idx"] = np.array(data["cell_topo_idx"], dtype=object)
        
        # Calculer le nombre de topoclusters uniques pour chaque événement
        num_topoclusters_per_event = np.array(
            [len(np.unique(v)) if v.size > 0 else 0 for v in data["cell_topo_idx"]]
        )
        
        # Calculer la valeur d'indice maximale pour chaque événement
        max_indice_topoclusters = np.array(
            [v.max() if v.size > 0 else -1 for v in data["cell_topo_idx"]]
        )

        # La condition pour un événement "valide" 
        valid_event_mask = max_indice_topoclusters <= num_topoclusters_per_event

        # --- NOUVEAU : Boucle pour afficher les détails des événements supprimés ---
        removed_indices = np.where(~valid_event_mask)[0]
        if len(removed_indices) > 0:
            print("--- Début de la liste des événements supprimés ---")
            for idx in removed_indices:
                max_idx = max_indice_topoclusters[idx]
                num_clusters = num_topoclusters_per_event[idx]
                print(
                    f"  Événement à l'indice {idx}: Supprimé. "
                    f"Raison: Indice max trouvé ({max_idx}) >= Nombre de clusters ({num_clusters})"
                )
            print(f"--- Fin de la liste. Total: {len(removed_indices)} événements supprimés. ---")
        # --- Fin de la boucle d'affichage ---

        # Appliquer le masque à chaque variable du dictionnaire de données
        for key in data:
            if len(data[key]) == len(valid_event_mask):
                data[key] = data[key][valid_event_mask]

        # Mettre à jour le nombre total d'événements
        self.num_read = valid_event_mask.sum()     
        return data

    def _preprocessing(self, data, build_topocluster):
        # ======================== #
        # === Rename variables === #
        # ======================== #
        if self.is_private_sample:
            pass
        else:
            data["particle_track_idx"] = data["particle_to_track"]
            data.pop("particle_to_track")

        # ======================== #
        # === Define variables === #
        # ======================== #
        # Define track pdgid
        # if isinstance(data["track_parent_idx"], list):
        #     data["track_parent_idx"] = np.array(data["track_parent_idx"])
        # ensure integer indices for track_parent_idx
        data["track_parent_pdgid"] = np.array(
            [pdgid[np.asarray(idx, dtype=int)] for pdgid, idx in zip(
                data["particle_pdgid"], data["track_parent_idx"]
            )],
            dtype=object,
        )

        # Counting important stat
        self._counting(data, self.num_read)

        # Flatten array
        # i.e. [[evt1_trk1, evt1_trk2], [env2_trk1, env2_trk2]] -> [evt1_trk1, evt1_trk2, env2_trk1, env2_trk2]
        for var in data.keys():
            if var.startswith("meta_"):
                continue
            data[var] = np.concatenate(data[var])

        # Special variables (log(d0), log(e), eta, pt, etc.)
        self._make_special_variables(data)

        self._cleaning_event(data)

        # cell_topo_idx starts with 1. So decrement idx
        data["cell_topo_idx"] -= 1

        self._define_global_etaphi(data)
        self._define_delta_variables(data)

        # Define topo clusters
        if build_topocluster:
            self._build_topoclusters(data)

        return data

    def _make_special_variables(self, data):
        # particle properties
        if self.is_private_sample:
            p4 = vector.array(
                {
                    "pt": data["particle_pt"],
                    "eta": data["particle_eta"],
                    "phi": data["particle_phi"],
                    "energy": data["particle_e"],
                }
            )
        else:
            p4 = vector.array(
                {
                    "px": data["particle_px"],
                    "py": data["particle_py"],
                    "pz": data["particle_pz"],
                    "energy": data["particle_e"],
                }
            )
        data["particle_phi"] = p4.phi
        data["particle_theta"] = p4.theta
        data["particle_eta"] = p4.eta
        data["particle_xhat"] = np.cos(p4.phi)
        data["particle_yhat"] = np.sin(p4.phi)
        data["particle_pt"] = p4.pt
        data["particle_logpt"] = np.log(p4.pt)
        data["particle_m"] = p4.m

        # add a new class for charged with no tracks - check if this is needed or not
        #WARNING combined_index dangerous TO REMOVE
        data["particle_class"] = class_label(data["particle_pdgid"], combined_index=True)
        data["particle_charge"] = charge_label(data["particle_pdgid"])

        for var in data.keys():
            if "_phi" in var:
                data[var][data[var] > np.pi] -= 2 * np.pi
                data[var][data[var] < -np.pi] += 2 * np.pi

        # phi to sin/cos
        for ilayer in range(6):
            p4 = vector.array(
                {
                    "x": data["track_x_layer_" + str(ilayer)],
                    "y": data["track_y_layer_" + str(ilayer)],
                    "z": data["track_z_layer_" + str(ilayer)],
                }
            )
            data["track_eta_layer_" + str(ilayer)] = p4.eta
            data["track_phi_layer_" + str(ilayer)] = p4.phi
        for var in [
            "track_phi",
            "track_phi_layer_0",
            "track_phi_layer_1",
            "track_phi_layer_2",
            "track_phi_layer_3",
            "track_phi_layer_4",
            "track_phi_layer_5",
            "cell_phi",
        ]:
            data[var.replace("phi", "sinphi")] = np.sin(data[var])
            data[var.replace("phi", "cosphi")] = np.cos(data[var])

        data["track_logd0"] = np.sign(data["track_d0"]) * np.log(1 + 50.0 * abs(data["track_d0"]))
        data["track_logz0"] = np.sign(data["track_z0"]) * np.log(1 + 50.0 * abs(data["track_z0"]))

        data["cell_loge"] = np.log(data["cell_e"])
        data["cell_et"] = data["cell_e"] / np.cosh(data["cell_eta"])
        data["cell_loget"] = np.log(data["cell_et"])
        data["particle_loge"] = np.log(data["particle_e"])

        data["track_eta"] = -np.log(np.tan(data["track_theta"] / 2))

        data["track_pt"] = np.abs(1.0 / data["track_qoverp"]) * np.sin(data["track_theta"])
        data["track_logpt"] = np.log(data["track_pt"])

        # Add isMuon (and isIso)
        data["track_isMuon"] = np.zeros_like(data["track_parent_pdgid"], dtype=np.int32)
        data["track_isMuon"][np.abs(data["track_parent_pdgid"]) == 13] = 1

        # Cell variables
        if self.add_vars_for_visualization:
            # Add cell energy / noise as cell and node feature ###
            layer_noise = np.array([13.0, 34.0, 41.0, 75.0, 50.0, 25.0])
            cell_e = data["cell_e"]
            cell_layer = data["cell_layer"]
            cell_noise = layer_noise[cell_layer]
            data["cell_zeta"] = cell_e / cell_noise

    def _cleaning_event(self, data):
        # transform particle_to_track in -1 and 1
        data["particle_has_track"] = np.where(data["particle_track_idx"] >= 0, True, False)

        # charged particles with tracks that don't deposit any energy are defined to be electrons
        no_energy_dep_ch_mask = np.logical_and.reduce(
            [
                data["particle_has_track"],  # Original code might be buggy.
                data["particle_dep_energy"] == 0,
                data["particle_class"] != 2,  # Not Muon
            ]
        )

        # !!!!!!!!! Hard-coded, we don't care about it during evaluation
        data["particle_class"][no_energy_dep_ch_mask] = 1  # Electron label

    def _counting(self, data, num_read):
        self.count_data = {}

        # Counting
        self.count_data["n_cells"] = [len(x) for x in data["cell_x"]]
        self.count_data["n_tracks"] = [len(x) for x in data["track_d0"]]
        self.count_data["n_particles"] = [len(x) for x in data["particle_pdgid"]]
        self.count_data["n_topojets"] = [len(x) for x in data["topo_jet_pt"]]
        self.count_data["n_topoclusters"] = [len(np.unique(x)) for x in data["cell_topo_idx"]]
        self.count_data["cell_cumsum"] = np.cumsum([0] + self.count_data["n_cells"])
        self.count_data["track_cumsum"] = np.cumsum([0] + self.count_data["n_tracks"])
        self.count_data["particle_cumsum"] = np.cumsum([0] + self.count_data["n_particles"])
        self.count_data["topojet_cumsum"] = np.cumsum([0] + self.count_data["n_topojets"])
        self.count_data["topocluster_cumsum"] = np.cumsum([0] + self.count_data["n_topoclusters"])
        self.count_data["n_cells"] = np.array(self.count_data["n_cells"])
        self.count_data["n_tracks"] = np.array(self.count_data["n_tracks"])
        self.count_data["n_particles"] = np.array(self.count_data["n_particles"])
        self.count_data["n_topojets"] = np.array(self.count_data["n_topojets"])
        self.count_data["n_topoclusters"] = np.array(self.count_data["n_topoclusters"])

        # needed for batch sampling
        self.count_data["n_nodes"] = np.array(
            [
                self.count_data["n_topoclusters"][i] + self.count_data["n_tracks"][i]
                for i in range(num_read)
            ]
        )

    def _define_global_etaphi(self, data):
        cumsum = self.count_data["cell_cumsum"][:-1]
        x = data["cell_x"]
        y = data["cell_y"]
        z = data["cell_z"]
        eta = data["cell_eta"]
        phi = data["cell_phi"]
        et = data["cell_et"]
        et_sum = np.add.reduceat(et, cumsum)
        data["global_x"] = np.add.reduceat(et * x, cumsum) / et_sum
        data["global_y"] = np.add.reduceat(et * y, cumsum) / et_sum
        data["global_z"] = np.add.reduceat(et * z, cumsum) / et_sum
        data["global_eta"] = np.add.reduceat(et * eta, cumsum) / et_sum
        data["global_phi"] = np.add.reduceat(et * phi, cumsum) / et_sum

    def _define_delta_variables(self, data):
        n_cells = self.count_data["n_cells"]
        data["cell_deltax"] = data["cell_x"] - np.repeat(data["global_x"], n_cells)
        data["cell_deltay"] = data["cell_y"] - np.repeat(data["global_y"], n_cells)
        data["cell_deltaz"] = data["cell_z"] - np.repeat(data["global_z"], n_cells)
        data["cell_deltaeta"] = data["cell_eta"] - np.repeat(data["global_eta"], n_cells)
        data["cell_deltaphi"] = data["cell_phi"] - np.repeat(data["global_phi"], n_cells)
        data["cell_deltaphi"] = np.mod(data["cell_deltaphi"] + np.pi, 2 * np.pi) - np.pi

        n_tracks = self.count_data["n_tracks"]
        data["track_deltaeta"] = data["track_eta"] - np.repeat(data["global_eta"], n_tracks)
        data["track_deltaphi"] = data["track_phi"] - np.repeat(data["global_phi"], n_tracks)
        data["track_deltaphi"] = np.mod(data["track_deltaphi"] + np.pi, 2 * np.pi) - np.pi
        for i in range(6):
            data["track_deltaeta_layer_" + str(i)] = data["track_eta_layer_" + str(i)] - np.repeat(
                data["global_eta"], n_tracks
            )
            data["track_deltaphi_layer_" + str(i)] = data["track_phi_layer_" + str(i)] - np.repeat(
                data["global_phi"], n_tracks
            )
            data["track_deltaphi_layer_" + str(i)] = (
                np.mod(data["track_deltaphi_layer_" + str(i)] + np.pi, 2 * np.pi) - np.pi
            )

        #TODO SHOULD BE A NEW VARIABLE BUT FOR FASTER TEST I PUT IN GENERAL
        n_particles = self.count_data["n_particles"]
        data["particle_eta"] = data["particle_eta"] - np.repeat(data["global_eta"], n_particles)
        data["particle_phi"] = data["particle_phi"] - np.repeat(data["global_phi"], n_particles)

    def _build_topoclusters(self, data):
        offset = np.repeat(self.count_data["topocluster_cumsum"][:-1], self.count_data["n_cells"])
        cell_topo_idx = data["cell_topo_idx"] + offset
        cell_topo_idx[data["cell_topo_idx"] == -1] = -1

        idx_cell_sort = cell_topo_idx.argsort()
        cell_sorted = {}
        cell_sorted["e"] = data["cell_e"][idx_cell_sort]
        cell_sorted["eta"] = data["cell_eta"][idx_cell_sort]
        cell_sorted["phi"] = data["cell_phi"][idx_cell_sort]
        cell_sorted["layer"] = data["cell_layer"][idx_cell_sort]
        cell_sorted["x"] = data["cell_x"][idx_cell_sort]
        cell_sorted["y"] = data["cell_y"][idx_cell_sort]
        cell_sorted["z"] = data["cell_z"][idx_cell_sort]
        cell_topo_idx = cell_topo_idx[idx_cell_sort]

        topocoluster_idx, edge_idx = np.unique(cell_topo_idx, return_index=True)

        if topocoluster_idx[0] == -1:
            print(topocoluster_idx)
            print(edge_idx)
            raise ValueError()

        data["topocluster_e"] = np.add.reduceat(cell_sorted["e"], edge_idx)

        def _mean(x):
            return np.add.reduceat(cell_sorted["e"] * x, edge_idx) / data["topocluster_e"]

        data["topocluster_loge"] = np.log(data["topocluster_e"])
        data["topocluster_eta"] = _mean(cell_sorted["eta"])
        data["topocluster_loget"] = np.log(data["topocluster_e"] / np.cosh(data["topocluster_eta"]))
        sinphi = _mean(np.sin(cell_sorted["phi"]))
        cosphi = _mean(np.cos(cell_sorted["phi"]))
        data["topocluster_phi"] = np.arctan2(sinphi, cosphi)
        data["topocluster_sinphi"] = sinphi
        data["topocluster_cosphi"] = cosphi

        data["topocluster_emfrac"] = _mean(cell_sorted["layer"] <= 3)
        data["topocluster_hadfrac"] = _mean(cell_sorted["layer"] >= 4)

        data["topocluster_x"] = _mean(cell_sorted["x"])
        data["topocluster_y"] = _mean(cell_sorted["y"])
        data["topocluster_z"] = _mean(cell_sorted["z"])

        data["topocluster_layer"] = _mean(cell_sorted["layer"]).astype(np.float32)

        data["topocluster_idx"] = topocoluster_idx

        n = self.count_data["n_topoclusters"]
        data["topocluster_deltax"] = data["topocluster_x"] - np.repeat(data["global_x"], n)
        data["topocluster_deltay"] = data["topocluster_y"] - np.repeat(data["global_y"], n)
        data["topocluster_deltaz"] = data["topocluster_z"] - np.repeat(data["global_z"], n)
        data["topocluster_deltaeta"] = data["topocluster_eta"] - np.repeat(data["global_eta"], n)
        data["topocluster_deltaphi"] = data["topocluster_phi"] - np.repeat(data["global_phi"], n)
        data["topocluster_deltaphi"] = (
            np.mod(data["topocluster_deltaphi"] + np.pi, 2 * np.pi) - np.pi
        )

    def _cell_iter(self, idx):
        return slice(self.count_data["cell_cumsum"][idx], self.count_data["cell_cumsum"][idx + 1])

    def _track_iter(self, idx):
        return slice(self.count_data["track_cumsum"][idx], self.count_data["track_cumsum"][idx + 1])

    def _topocluster_iter(self, idx):
        return slice(
            self.count_data["topocluster_cumsum"][idx],
            self.count_data["topocluster_cumsum"][idx + 1],
        )

    def _particle_iter(self, idx):
        return slice(
            self.count_data["particle_cumsum"][idx], self.count_data["particle_cumsum"][idx + 1]
        )

    def _cell(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][self._cell_iter(idx)]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][self._cell_iter(idx)]
            return self.normalizer.normalize(value, var)

    def _track(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][self._track_iter(idx)]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][self._track_iter(idx)]
            return self.normalizer.normalize(value, var)

    def _topocluster(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][self._topocluster_iter(idx)]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][self._topocluster_iter(idx)]
            return self.normalizer.normalize(value, var)

    def _particle(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][self._particle_iter(idx)]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][self._particle_iter(idx)]
            return self.normalizer.normalize(value, var)

    def _node(self, var, idx):
        n_topoclusters = self.count_data["n_topoclusters"][idx]
        n_tracks = self.count_data["n_tracks"][idx]

        if var.endswith(":normed"):
            raise NotImplementedError("node variables should not be normalized.")

        var_raw = var.replace(":normed", "")
        if var_raw == "isTrack":
            return np.concatenate(
                [
                    np.zeros((n_topoclusters,), dtype=int),
                    np.ones((n_tracks,), dtype=int),
                ],
                axis=0,
            )
        elif var_raw in ["emfrac", "hadfrac"]:
            return np.concatenate(
                [
                    self._topocluster(f"topocluster_{var}", idx),
                    np.full((n_tracks,), -1.0),
                ],
                axis=0,
            )
        elif var_raw in ["eta", "phi", "sinphi", "cosphi", "deltaeta", "deltaphi"]:
            return np.concatenate(
                [
                    self._topocluster(f"topocluster_{var}", idx),
                    self._track(f"track_{var}", idx),
                ],
                axis=0,
            )
        elif var_raw == "logpt":
            return np.concatenate(
                [
                    self._topocluster("topocluster_loget", idx),
                    self._track("track_logpt", idx),
                ],
                axis=0,
            )
        else:
            raise NotImplementedError(f"{var} is not implemented.")

    def _global(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][idx]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][idx]
            return self.normalizer.normalize(value, var)

    def _meta(self, var, idx):
        if not var.endswith(":normed"):
            return self.data[var][idx]
        else:
            var = var.replace(":normed", "")
            value = self.data[var][idx]
            return self.normalizer.normalize(value, var)
