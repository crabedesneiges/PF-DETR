"""
PyTorch Lightning DataModule for PFlowDataset.
"""
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from .DataLoader import PFlowDataset


class PFlowDataModule(pl.LightningDataModule):
    """
    DataModule wrapping PFlowDataset for train/val/test splits.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config["dataset"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # No-op: PFlowDataset reads data lazily
        pass

    def setup(self, stage=None):
        # Called on every GPU in distributed settings
        # prepare PFlowDataset kwargs from config
        kwargs = dict(
            cache=self.config["enable_dataloader_cache"],
            use_boolean_incident_matrix=self.config.get("bool_inc", False),
            build_topocluster=self.config.get("build_topocluster", True),
            max_particles=self.config["max_particles"],
            normalization_path=self.config["path_to_normalization_params"],
            num_workers=self.config["num_dataloader_workers"],
            is_private_sample=self.config.get("is_private_sample", False),
        )

        if stage == 'fit' or stage is None:
            if "path_to_train_valid" in self.config:
                num_events = self.config["num_events_train_valid"]
                indices = np.random.permutation(num_events)
                split_point = int(num_events * self.config["frac_train"])
                train_idx = indices[:split_point]
                valid_idx = indices[split_point:]

                self.train_dataset = PFlowDataset(
                    filename=self.config["path_to_train_valid"],
                    num_read=num_events,
                    isEval=False,
                    event_mask=train_idx,
                    **kwargs
                )
                self.val_dataset = PFlowDataset(
                    filename=self.config["path_to_train_valid"],
                    num_read=num_events,
                    isEval=False,
                    event_mask=valid_idx,
                    **kwargs
                )
            else:
                self.train_dataset = PFlowDataset(
                    filename=self.config["path_to_train"],
                    num_read=self.config["num_events_train"],
                    isEval=False,
                    **kwargs
                )
                self.val_dataset = PFlowDataset(
                    filename=self.config["path_to_valid"],
                    num_read=self.config["num_events_valid"],
                    isEval=False,
                    **kwargs
                )

        if stage == 'test' or stage is None:
            self.test_dataset = PFlowDataset(
                filename=self.config["path_to_test"],
                num_read=self.config.get("num_events_test", -1),
                isEval=True,
                **kwargs
            )

    def collate_fn(self, batch):
        incidence_matrix = [torch.from_numpy(item["target"]["inc"]["incidence_matrix"]) for item in batch]
        #ghost_particle_idx = [torch.from_numpy(item["target"]["inc"]["ghost_particle_idx"]) for item in batch]
        missing_particle_idx = [torch.from_numpy(item["target"]["inc"]["missing_particle_idx"]) for item in batch]

        # Cells: pad to max nodes in batch
        cell_feats = [item['cells']['features'].permute(1, 0) for item in batch]
        max_nc = max(cf.size(1) for cf in cell_feats)
        padded_cells = torch.stack([F.pad(cf, (0, max_nc - cf.size(1)), value=0) for cf in cell_feats])
        cell_mask = torch.zeros(len(cell_feats), max_nc, dtype=torch.bool)
        for i, cf in enumerate(cell_feats):
            cell_mask[i, :cf.size(1)] = True

        # Topoclusters
        topo_feats = [item['topoclusters']['features'].permute(1, 0) for item in batch]
        max_nt = max(tf.size(1) for tf in topo_feats)
        padded_topo = torch.stack([F.pad(tf, (0, max_nt - tf.size(1)), value=0) for tf in topo_feats])
        topo_mask = torch.zeros(len(topo_feats), max_nt, dtype=torch.bool)
        for i, tf in enumerate(topo_feats):
            topo_mask[i, :tf.size(1)] = True

        # Tracks
        track_feats = [item['tracks']['features'].permute(1, 0) for item in batch]
        max_nk = max(tr.size(1) for tr in track_feats)
        padded_tracks = torch.stack([F.pad(tr, (0, max_nk - tr.size(1)), value=0) for tr in track_feats])
        track_mask = torch.zeros(len(track_feats), max_nk, dtype=torch.bool)
        for i, tr in enumerate(track_feats):
            track_mask[i, :tr.size(1)] = True

        inputs = (padded_cells, padded_topo, padded_tracks)

        # Targets: pad/truncate to fixed num_queries
        num_q = self.config['max_particles']
        padding_idx = self.config.get('num_classes', 5)  # Default to 5 if not present
        p4_list = [item['target']['p4']['values'] for item in batch]
        cls_list = [item['target']['class']['values'] for item in batch]
        is_track_list = [item['target']['is_track']['values'] for item in batch]
        boxes, labels, is_track_temp = [], [], []
        for p4, cls, is_track in zip(p4_list, cls_list, is_track_list):
            n = p4.size(0)
            if n >= num_q:
                p4_pad, cls_pad = p4[:num_q], cls[:num_q]
                is_track_pad = is_track[:num_q]
            else:
                pad = num_q - n
                p4_pad = torch.cat([p4, torch.zeros(pad, p4.size(1), dtype=p4.dtype, device=p4.device)], dim=0)
                cls_pad = torch.cat([
                    cls,
                    torch.full((pad,), padding_idx, dtype=cls.dtype, device=cls.device)
                ], dim=0)
                is_track_pad = torch.cat([
                    is_track,
                    torch.full((pad,), 0, dtype=is_track.dtype, device=is_track.device)
                ], dim=0)

            boxes.append(p4_pad)
            labels.append(cls_pad)
            is_track_temp.append(is_track_pad)
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)
        is_track = torch.stack(is_track_temp)

        targets = {
            'labels': labels,
            'boxes': boxes,
            'padding_mask': (labels != padding_idx),
            'is_track': is_track,
            'incidence_matrix': incidence_matrix,
            #'ghost_particle_idx': ghost_particle_idx,
            'missing_particle_idx': missing_particle_idx,
        }

        return {
            'input': inputs,
            'target': targets,
            'cell_mask': cell_mask,
            'topo_mask': topo_mask,
            'track_mask': track_mask,
        }

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config["batchsize"],
            shuffle=False,
            num_workers=self.config["num_dataloader_workers"],
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return dl

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batchsize"],
            shuffle=False,
            num_workers=self.config["num_dataloader_workers"],
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup(stage='test')
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batchsize"],
            shuffle=False,
            num_workers=self.config["num_dataloader_workers"],
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
