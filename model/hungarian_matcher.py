import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np
class HungarianMatcher:
    """
    Implements the Hungarian matching for set prediction (DETR style).
    """
    def __init__(self, cls_cost=1.0, bbox_cost=1.0, padding_idx=5):
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.padding_idx = padding_idx

    @torch.no_grad()
    def __call__(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes):
        """
        Args:
            pred_logits: [batch_size, num_queries, num_classes]
            pred_boxes:  [batch_size, num_queries, box_dim]
            tgt_labels:  list of [num_targets_i] LongTensor for each batch
            tgt_boxes:   list of [num_targets_i, box_dim] FloatTensor for each batch
        Returns:
            List of (index_pred, index_tgt) for each batch
        """
        batch_size, num_queries, num_classes = pred_logits.shape
        indices = []
        for b in range(batch_size):
            out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
            out_bbox = pred_boxes[b]               # [num_queries, box_dim]
            tgt_lbl = tgt_labels[b]
            tgt_box = tgt_boxes[b]
            num_tgt = tgt_lbl.shape[0]
            # Indicator: 1 si vrai objet, 0 si padding
            if hasattr(self, 'padding_idx'):
                padding_idx = self.padding_idx
            else:
                padding_idx = num_classes
            if self.padding_idx is None:
                # Pas de padding, tous les objets sont vrais
                indicator = torch.ones(num_tgt, device=tgt_lbl.device, dtype=torch.float)
            else:
                # Avec padding
                indicator = (tgt_lbl != self.padding_idx).float()  # [num_targets]
            # Cost class: -p_sigma(i)(ci) si vrai objet, 0 sinon
            cost_class = -out_prob[:, tgt_lbl] * indicator  # [num_queries, num_targets]
            # Cost bbox: L1 distance si vrai objet, 0 sinon
            cost_bbox = torch.cdist(out_bbox, tgt_box, p=1) * indicator
            C = self.cls_cost * cost_class + self.bbox_cost * cost_bbox
            C = C.cpu().detach().numpy()
            if not np.all(np.isfinite(C)):
                print("Invalid entries in cost matrix C!")
                print("C:", C)
                print("NaNs:", np.isnan(C).sum(), "Infs:", np.isinf(C).sum())
                raise ValueError("Cost matrix contains NaN or Inf values.")

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long),
                            torch.as_tensor(col_ind, dtype=torch.long)))
        return indices
