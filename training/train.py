#!/usr/bin/env python
"""Training script for the DETR pipeline using PyTorch Lightning."""

import os
import argparse
import yaml

# --- Early argument parsing to set CUDA_VISIBLE_DEVICES before importing torch ---
# --- Analyse précoce des arguments pour définir CUDA_VISIBLE_DEVICES avant d'importer torch ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="config/config_DETR.yaml", help="Path to YAML config")
parser.add_argument("--gpu-device", type=str, default=None, help="CUDA device id(s) to use, e.g. '0' or '0,1'")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
args, _ = parser.parse_known_args()

# --- Set CUDA_VISIBLE_DEVICES before importing torch ---
# --- Définir CUDA_VISIBLE_DEVICES avant d'importer torch ---
if args.gpu_device:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES to: {args.gpu_device}")
    else:
        print(f"[INFO] CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

# --- Now safe to import torch and others ---
# --- Maintenant, il est sûr d'importer torch et les autres bibliothèques ---
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from callbacks.global_summary import GlobalSummaryCallback
from callbacks.loss_history_csv import LossHistoryCSVCallback
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

from data.pflow_datamodule import PFlowDataModule
from model.lightning_V3 import DETRLightningModule


def validate_config(config):
    """Valide la configuration."""
    for sec in ("dataset", "model", "training"):
        if sec not in config:
            raise KeyError(f"Config missing section '{sec}'")

    ds = config["dataset"]
    if not ds.get("path_to_train_valid") and not (ds.get("path_to_train") and ds.get("path_to_valid")):
        raise KeyError("Dataset must define 'path_to_train_valid' or both 'path_to_train' and 'path_to_valid'")
    if "path_to_test" not in ds:
        raise KeyError("Dataset missing 'path_to_test'")
    if "num_epochs" not in config["training"]:
        raise KeyError("Training config missing 'num_epochs'")
    return True


def main():
    """Fonction principale d'entraînement."""
    config_path = args.config
    resume_ckpt = args.resume
    original_gpu_devices = args.gpu_device
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"[INFO] Using GPU devices: {original_gpu_devices} (renumbered internally to 0-{num_gpus - 1})")
    print(f"[INFO] PyTorch sees {num_gpus} GPU(s)")

    # Load config
    # Charger la configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)

    config_name = os.path.splitext(os.path.basename(config_path))[0]

    # Optional info log
    # Journal d'informations facultatif
    warmup_steps = config["training"].get("warmup_steps", 0)
    if warmup_steps > 0:
        print(f"[INFO] Warm-up steps: {warmup_steps}")
        print(f"[INFO] Warm-up start LR: {config['training'].get('warmup_start_lr', 1e-6)}")
        print(f"[INFO] Base LR: {config['training'].get('base_lr', config['training']['learningrate'])}")
    if config["training"].get("lr_scheduler", {}).get("name") == "CosineAnnealingLR":
        print("[INFO] Scheduler: CosineAnnealingLR")

    # Initialize model & datamodule
    # Initialiser le modèle et le datamodule
    dm = PFlowDataModule(config)
    model = DETRLightningModule(config)

    # Callbacks
    monitor = config["training"].get("monitoring_metric", "val_loss")
    callbacks = [
        GlobalSummaryCallback(),
        LossHistoryCSVCallback(),
        ModelCheckpoint(
            monitor=monitor,
            save_top_k=-1,
            every_n_epochs=25,
            mode="min",
            dirpath=os.path.join("checkpoints", config_name),
            filename=f"detr-epoch{{epoch:02d}}-{{{monitor}:.4f}}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config["training"].get("earlystopping", False):
        callbacks.append(EarlyStopping(
            monitor=monitor,
            patience=config["training"].get("earlystopping_patience", 10),
            mode="min"
        ))

    # Loggers
    loggers = [TensorBoardLogger(save_dir="lightning_logs", name=config_name)]

    # Trainer configuration
    # Configuration de l'entraîneur
    trainer_kwargs = {
        "max_epochs": config["training"]["num_epochs"],
        "callbacks": callbacks,
        "logger": loggers,
        "precision": "16-mixed",
        "log_every_n_steps": config["training"].get("log_every_n_steps", 25),
        "gradient_clip_val": config["training"].get("gradient_clip_val", 0.0),
        "accumulate_grad_batches": config["training"].get("accumulate_grad_batches", 1),
        "enable_checkpointing": True,
    }

    if num_gpus > 0:
        trainer_kwargs.update({
            "accelerator": "gpu",
            "devices": num_gpus,
        })
        if num_gpus > 1:
            if config["training"].get("fsdp", False):
                from pytorch_lightning.strategies import FSDPStrategy
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                trainer_kwargs["strategy"] = FSDPStrategy(
                    auto_wrap_policy=size_based_auto_wrap_policy(min_num_params=1e6),
                    activation_checkpointing=True,
                    state_dict_type="full",
                    cpu_offload=False
                )
            else:
                trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
    else:
        trainer_kwargs.update({
            "accelerator": "cpu",
            "devices": 1
        })

    # Final logging
    # Journalisation finale
    print("[INFO] Final training config:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")
        print(f"[INFO] Default CUDA device: cuda:{torch.cuda.current_device()}")

    trainer = pl.Trainer(**trainer_kwargs)

    # Start training
    # Démarrer l'entraînement
    if resume_ckpt:
        print(f"[INFO] Resuming training from checkpoint: {resume_ckpt}")
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, dm)

    # Test phase
    # Phase de test
    print("[INFO] Starting test phase...")
    test_trainer = pl.Trainer(
        logger=loggers,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else 1,
        precision=trainer_kwargs.get("precision", "32-true"),
    )
    test_trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()