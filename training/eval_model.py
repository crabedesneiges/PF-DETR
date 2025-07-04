import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import click

import tqdm
import yaml



sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import get_latest_checkpoints, setup_environment
from data.pflow_datamodule import PFlowDataModule
from model.lightning_V3 import DETRLightningModule
from model.hungarian_matcher import HungarianMatcher

@click.command()
@click.option("--config_path", type=str, default="configs/test.yaml")
@click.option("--cuda_visible_device", type=str, default="")
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--seed", help="random seed", type=int, default=None)
@click.option("--eval_train", is_flag=True, help="Evaluate on training dataset instead of test", default=False)
@click.option("--batchsize", type=int, default=None, help="Override batch size for evaluation")

def main(**args):
    with open(args["config_path"], "r") as fp:
        config = yaml.safe_load(fp)

    # OPTIMIZATION 1: Increase batch size for evaluation if not specified
    if args["batchsize"] is not None:
        config["dataset"]["batchsize"] = args["batchsize"]
    elif "batchsize" not in config["dataset"] or config["dataset"]["batchsize"] < 16:
        config["dataset"]["batchsize"] = min(32, config["dataset"].get("batchsize", 8) * 2)
        print(f"Increased batch size to {config['dataset']['batchsize']} for faster evaluation")

    ngpus = setup_environment(
        config,
        cuda_visible_device=args["cuda_visible_device"],
        seed=args["seed"],
    )
    import torch
    from pytorch_lightning import Trainer
    import torch.nn.functional as F 
    basename = os.path.basename(args["config_path"]).removesuffix(".yaml")
    lightning_logdir = f"./workspace/train/{basename}"

    if args["checkpoint_path"] is None:
        checkpoint_path = get_latest_checkpoints(lightning_logs=f"{lightning_logdir}")
    else:
        checkpoint_path = args["checkpoint_path"]
    if (checkpoint_path is None) or (not os.path.exists(checkpoint_path)):
        raise ValueError("Invalid checkpoint_path:", checkpoint_path)

    outputdir = Path("./workspace/npz")
    outputdir.mkdir(parents=True, exist_ok=True)
    
    dataset_type = "train" if args["eval_train"] else "test"
    outputfilepath = os.path.join(outputdir, f"{basename}_{dataset_type}.npz")
    print(f"PFlow result will be stored in (evaluating on {dataset_type} dataset)\n", outputfilepath)

    # Load model from checkpoint
    net = DETRLightningModule.load_from_checkpoint(checkpoint_path, config=config)
    datamodule = PFlowDataModule(config)

    # OPTIMIZATION 2: Disable unnecessary training features for inference
    trainer = Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="gpu" if ngpus > 0 else "cpu",
        default_root_dir=lightning_logdir,
        use_distributed_sampler=False,
        log_every_n_steps=50,
        logger=False,
        enable_checkpointing=False,  # Disable checkpointing during eval
        enable_progress_bar=True,    # Keep progress bar for monitoring
        precision="16-mixed" if ngpus > 0 else 32,  # Use mixed precision if GPU available
    )

    # Choose which dataset to evaluate on based on eval_train flag
    if args["eval_train"]:
        print("Evaluating on training dataset...")
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        predictions = trainer.predict(net, dataloaders=train_dataloader)
    else:
        print("Evaluating on test dataset...")
        predictions = trainer.predict(net, datamodule=datamodule)

    def process_predictions_optimized(predictions):
        """
        Optimized version with vectorized operations and reduced memory allocation
        """
        # OPTIMIZATION 3: Pre-allocate matcher once
        matcher = HungarianMatcher(cls_cost=1.0, bbox_cost=1.0, padding_idx=5)
        results = []
        
        # OPTIMIZATION 4: Process with progress bar and batch pre-allocation
        total_events = sum(batch['pred_logits_track'].shape[0] for batch in predictions)
        print(f"Processing {total_events} events...")
        
        with tqdm.tqdm(total=total_events, desc="Processing predictions") as pbar:
            for batch_idx, batch in enumerate(predictions):
                batch_size = batch["pred_logits_track"].shape[0]
                
                # Pre-extract batch data to avoid repeated indexing
                pred_logits_track = batch["pred_logits_track"]
                pred_boxes_track = batch["pred_boxes_track"]
                pred_logits_notrack = batch["pred_logits_notrack"]  
                pred_boxes_notrack = batch["pred_boxes_notrack"]
                truth_labels = batch["ground_truth"]["labels"].clone()  # Clone once
                truth_boxes = batch["ground_truth"]["boxes"]
                truth_is_track = batch["ground_truth"]["is_track"]

                # OPTIMIZATION 5: Vectorized label modification
                condition1_mask = (truth_labels == 0) & (truth_is_track == 0)
                condition2_mask = (truth_labels == 1) & (truth_is_track == 0)
                truth_labels[condition1_mask] = 3
                truth_labels[condition2_mask] = 4
                
                # Pre-allocate tensors for this batch
                MAX_TRACK_PARTICLE = 15
                device = pred_logits_track.device
                
                for b in range(batch_size):
                    # OPTIMIZATION 6: Use boolean indexing more efficiently
                    is_track_mask = truth_is_track[b] == 1
                    is_notrack_mask = truth_is_track[b] == 0

                    tgt_labels_charged = truth_labels[b][is_track_mask]
                    tgt_boxes_charged = truth_boxes[b][is_track_mask]
                    tgt_labels_neutral = truth_labels[b][is_notrack_mask]
                    tgt_boxes_neutral = truth_boxes[b][is_notrack_mask]

                    real_neutral_mask = (tgt_labels_neutral != 5)
                    tgt_labels_real_neutral = tgt_labels_neutral[real_neutral_mask]
                    tgt_boxes_real_neutral = tgt_boxes_neutral[real_neutral_mask]

                    # Track matching
                    matcher.padding_idx = None
                    num_tgt = tgt_boxes_charged.shape[0]

                    if num_tgt > 0:
                        truncated_pred_boxes_track = pred_boxes_track[b][:num_tgt]
                        truncated_pred_logits_track = pred_logits_track[b][:num_tgt]
                        
                        indices_batch_track = matcher(
                            truncated_pred_logits_track.float().unsqueeze(0), 
                            truncated_pred_boxes_track.float().unsqueeze(0),
                            [tgt_labels_charged], 
                            [tgt_boxes_charged.float()]
                        )
                        src_idx, tgt_idx = indices_batch_track[0]
                        
                        # OPTIMIZATION 7: Use advanced indexing instead of loops
                        aligned_preds_boxes_track = torch.zeros_like(truncated_pred_boxes_track)
                        aligned_preds_logit_track = torch.zeros_like(truncated_pred_logits_track)
                        aligned_preds_boxes_track[tgt_idx] = truncated_pred_boxes_track[src_idx]
                        aligned_preds_logit_track[tgt_idx] = truncated_pred_logits_track[src_idx]
                    else:
                        aligned_preds_boxes_track = torch.zeros((0, pred_boxes_track.shape[-1]), device=device)
                        aligned_preds_logit_track = torch.zeros((0, pred_logits_track.shape[-1]), device=device)

                    # No-track matching
                    matcher.padding_idx = 2
                    
                    if len(tgt_labels_real_neutral) > 0:
                        tgt_labels_real_neutral_adjusted = torch.clamp(tgt_labels_real_neutral - 3, min=0)
                        
                        indices_batch_notrack = matcher(
                            pred_logits_notrack[b].float().unsqueeze(0), 
                            pred_boxes_notrack[b].float().unsqueeze(0),
                            [tgt_labels_real_neutral_adjusted], 
                            [tgt_boxes_real_neutral.float()]
                        )
                        src_idx, tgt_idx = indices_batch_notrack[0]
                        
                        aligned_tgt_boxes_notrack = torch.zeros_like(pred_boxes_notrack[b])
                        aligned_tgt_label_notrack = torch.full_like(pred_logits_notrack[b, :, 0], fill_value=2, dtype=torch.int64)
                        aligned_tgt_boxes_notrack[src_idx] = tgt_boxes_real_neutral[tgt_idx].to(aligned_tgt_boxes_notrack.dtype)
                        aligned_tgt_label_notrack[src_idx] = tgt_labels_real_neutral_adjusted[tgt_idx]
                    else:
                        aligned_tgt_boxes_notrack = torch.zeros_like(pred_boxes_notrack[b])
                        aligned_tgt_label_notrack = torch.full_like(pred_logits_notrack[b, :, 0], fill_value=2, dtype=torch.int64)

                    # OPTIMIZATION 8: Pre-allocate final tensors
                    num_track = aligned_preds_logit_track.shape[0]
                    
                    # Track logits padding
                    track_logits = torch.full((MAX_TRACK_PARTICLE, 6), -1e4, device=device)
                    if num_track > 0:
                        track_logits[:num_track, :3] = aligned_preds_logit_track
                    track_logits[num_track:, 5] = 0.0

                    # Track boxes padding
                    track_boxes_pred = torch.zeros((MAX_TRACK_PARTICLE, 3), device=device)
                    if num_track > 0:
                        track_boxes_pred[:num_track] = aligned_preds_boxes_track

                    # Track labels padding  
                    track_labels = torch.full((MAX_TRACK_PARTICLE,), 5, device=device)
                    if num_track > 0:
                        track_labels[:num_track] = tgt_labels_charged

                    # Track truth boxes padding
                    track_truth_boxes = torch.zeros((MAX_TRACK_PARTICLE, 3), device=device)
                    if num_track > 0:
                        track_truth_boxes[:num_track] = tgt_boxes_charged

                    # No-track logits formatting
                    notrack_logits = torch.full((pred_logits_notrack[b].shape[0], 6), -1e4, device=device)
                    notrack_logits[:, 3:] = pred_logits_notrack[b]

                    # Final concatenation
                    matched_pred_logit = torch.cat([track_logits, notrack_logits], dim=0)
                    matched_pred_boxes = torch.cat([track_boxes_pred, pred_boxes_notrack[b]], dim=0)
                    aligned_tgt_label_notrack_offset = aligned_tgt_label_notrack + 3
                    matched_truth_label = torch.cat([track_labels, aligned_tgt_label_notrack_offset], dim=0)
                    matched_truth_boxes = torch.cat([track_truth_boxes, aligned_tgt_boxes_notrack], dim=0)

                    # OPTIMIZATION 9: Move to CPU and convert to numpy in one step
                    event = {
                        "pred_logits": matched_pred_logit.detach().cpu().numpy(),
                        "pred_boxes": matched_pred_boxes.detach().cpu().numpy(), 
                        "truth_labels": matched_truth_label.detach().cpu().numpy(),
                        "truth_boxes": matched_truth_boxes.detach().cpu().numpy(),
                    }
                    results.append(event)
                    pbar.update(1)

        return results

    def denormalize_data_optimized(results, datamodule):
        """
        Optimized denormalization with reduced repeated operations
        """
        # Get dataset and normalizer once
        if args["eval_train"]:
            dataset = getattr(datamodule, "train_dataset", None)
        else:
            dataset = getattr(datamodule, "test_dataset", None)
            if dataset is None:
                dataset = getattr(datamodule, "val_dataset", None)
        
        if dataset is None or not hasattr(dataset, "normalizer"):
            raise ValueError("Cannot find normalizer in datamodule.")
        
        normalizer = dataset.normalizer
        p4_names = dataset.input_variables["graph"]["truths"]["p4"]
        clean_p4_names = [name.replace(":normed", "") for name in p4_names]
        
        print(f"Denormalizing {len(results)} events...")
        
        # OPTIMIZATION 10: Process in chunks and use vectorized operations where possible
        with tqdm.tqdm(total=len(results), desc="Denormalizing") as pbar:
            for event_idx, event in enumerate(results):
                # Cache global offsets
                global_eta = dataset[event_idx]["global_eta"]
                global_phi = dataset[event_idx]["global_phi"]
                eta_offset = global_eta.item() if hasattr(global_eta, 'item') else global_eta
                phi_offset = global_phi.item() if hasattr(global_phi, 'item') else global_phi
                
                # OPTIMIZATION 11: Denormalize all dimensions at once where possible
                for i, (name, clean_name) in enumerate(zip(p4_names, clean_p4_names)):
                    if i >= event["pred_boxes"].shape[1]:
                        continue
                        
                    grp_key = dataset.var2grp.get(clean_name, clean_name)
                    
                    # Denormalize pred and truth boxes for this dimension
                    event["pred_boxes"][:, i] = normalizer.denormalize(event["pred_boxes"][:, i], name=grp_key)
                    event["truth_boxes"][:, i] = normalizer.denormalize(event["truth_boxes"][:, i], name=grp_key)

                # Apply global offsets (vectorized)
                event["pred_boxes"][:, 1] += eta_offset  
                event["pred_boxes"][:, 2] += phi_offset
                event["truth_boxes"][:, 1] += eta_offset
                event["truth_boxes"][:, 2] += phi_offset
                
                pbar.update(1)

        # Convert logpt to pt if needed
        if "particle_logpt" in clean_p4_names:
            logpt_idx = clean_p4_names.index("particle_logpt")
            print(f"Converting logpt to pt at index {logpt_idx}")
            
            for event in results:
                if logpt_idx < event["pred_boxes"].shape[1]:
                    event["pred_boxes"][:, logpt_idx] = np.exp(event["pred_boxes"][:, logpt_idx])
                    event["truth_boxes"][:, logpt_idx] = np.exp(event["truth_boxes"][:, logpt_idx])
        
        return results
    
    print("Processing predictions...")
    results = process_predictions_optimized(predictions)
    print("Denormalizing data...")
    results = denormalize_data_optimized(results, datamodule)
    print("Saving results...")

    # OPTIMIZATION 12: Use more efficient numpy array creation
    outputdir = Path("./workspace/eval")
    outputdir.mkdir(parents=True, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args["config_path"]))[0]
    
    dataset_type = "train" if args["eval_train"] else "test"
    outputfilepath = os.path.join(outputdir, f"{basename}_{dataset_type}.npz")

    # Pre-allocate arrays for better memory efficiency
    n_events = len(results)
    if n_events > 0:
        n_queries = results[0]["pred_logits"].shape[0]
        n_classes = results[0]["pred_logits"].shape[1]
        n_box_features = results[0]["pred_boxes"].shape[1]
        
        # Pre-allocate arrays
        pred_logits = np.empty((n_events, n_queries, n_classes), dtype=np.float32)
        pred_boxes = np.empty((n_events, n_queries, n_box_features), dtype=np.float32)
        truth_labels = np.empty((n_events, n_queries), dtype=np.int64)
        truth_boxes = np.empty((n_events, n_queries, n_box_features), dtype=np.float32)
        
        # Fill arrays
        for i, event in enumerate(results):
            pred_logits[i] = event["pred_logits"]
            pred_boxes[i] = event["pred_boxes"]
            truth_labels[i] = event["truth_labels"]
            truth_boxes[i] = event["truth_boxes"]
    else:
        pred_logits = np.array([])
        pred_boxes = np.array([])
        truth_labels = np.array([])
        truth_boxes = np.array([])

    # Compute pred_class efficiently
    pred_logits_tensor = torch.from_numpy(pred_logits)
    pred_probs = F.softmax(pred_logits_tensor, dim=-1)
    pred_class = pred_probs.argmax(dim=-1).numpy()

    save_dict = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_class": pred_class,
        "truth_labels": truth_labels,
        "truth_boxes": truth_boxes,
        "is_denormalized": True,
    }

    # OPTIMIZATION 13: Use compression for smaller file size and faster I/O
    np.savez_compressed(outputfilepath, **save_dict)
    print(f"Saved prediction results to {outputfilepath}")
    print("Done.")

if __name__ == "__main__":
    main()