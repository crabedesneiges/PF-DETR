#!/usr/bin/env python
"""Training script for the DETR pipeline using PyTorch Lightning."""

import os
import argparse
import yaml

# --- Early argument parsing to set CUDA_VISIBLE_DEVICES before importing torch ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="config/config_DETR.yaml", help="Path to YAML config")
parser.add_argument("--gpu-device", type=str, default=None, help="CUDA device id(s) to use, e.g. '0' or '0,1'")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
parser.add_argument("--run-name", type=str, default=None, help="Name for the MLflow run")
parser.add_argument("--single-mlflow-run", action="store_true", help="Use a single MLflow run")
args, _ = parser.parse_known_args()

# --- Set CUDA_VISIBLE_DEVICES before importing torch ---
if args.gpu_device:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES to: {args.gpu_device}")
    else:
        print(f"[INFO] CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

# --- Now safe to import torch and others ---
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from callbacks.global_summary import GlobalSummaryCallback
from callbacks.loss_history_csv import LossHistoryCSVCallback
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
import mlflow

torch.set_float32_matmul_precision('medium')

from data.pflow_datamodule import PFlowDataModule
from model.lightning_V3_refactored import DETRLightningModule


def validate_config(config):
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
    config_path = args.config
    resume_ckpt = args.resume
    original_gpu_devices = args.gpu_device
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"[INFO] Using GPU devices: {original_gpu_devices} (renumbered internally to 0-{num_gpus - 1})")
    print(f"[INFO] PyTorch sees {num_gpus} GPU(s)")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)

    config_name = os.path.splitext(os.path.basename(config_path))[0]

    # Override config with CLI args
    if args.mlflow_tracking_uri:
        config.setdefault("training", {})["mlflow_tracking_uri"] = args.mlflow_tracking_uri
    if args.single_mlflow_run:
        config.setdefault("training", {})["single_mlflow_run"] = True

    # Optional info log
    warmup_steps = config["training"].get("warmup_steps", 0)
    if warmup_steps > 0:
        print(f"[INFO] Warm-up steps: {warmup_steps}")
        print(f"[INFO] Warm-up start LR: {config['training'].get('warmup_start_lr', 1e-6)}")
        print(f"[INFO] Base LR: {config['training'].get('base_lr', config['training']['learningrate'])}")
    if config["training"].get("lr_scheduler", {}).get("name") == "CosineAnnealingLR":
        print("[INFO] Scheduler: CosineAnnealingLR")

    # MLflow setup
    mlflow_run = None
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("pflow_DETR_experiments")
        run_name = args.run_name if args.run_name else config_name
        mlflow_run = mlflow.start_run(run_name=run_name)

        with mlflow_run:
            clean_config = {}
            for section, params in config.items():
                if isinstance(params, dict):
                    clean_config[section] = {
                        k: v.tolist() if hasattr(v, 'tolist') else v if isinstance(v, (int, float, str, bool, list)) else str(v)
                        for k, v in params.items()
                    }
                else:
                    clean_config[section] = str(params)
            for section, params in clean_config.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        mlflow.log_param(f"{section}.{k}", v)
                else:
                    mlflow.log_param(section, params)
            mlflow.set_tags({
                "config_file": os.path.basename(config_path),
                "model_version": "DETR_V2" if config["model"].get("use_DETR_V2") else "DETR_V1",
                "num_epochs": config["training"]["num_epochs"],
                "gpu_device": original_gpu_devices or "cpu"
            })

    # Initialize model & datamodule
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
    if args.mlflow_tracking_uri:
        loggers.append(MLFlowLogger(
            experiment_name="pflow_DETR_experiments",
            tracking_uri=args.mlflow_tracking_uri,
            run_name=args.run_name or config_name,
            run_id=mlflow_run.info.run_id if args.single_mlflow_run and mlflow_run else None
        ))

    # Trainer configuration
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
    print("[INFO] Final training config:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")
        print(f"[INFO] Default CUDA device: cuda:{torch.cuda.current_device()}")

    trainer = pl.Trainer(**trainer_kwargs)

    # Start training
    if resume_ckpt:
        print(f"[INFO] Resuming training from checkpoint: {resume_ckpt}")
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, dm)

    # Test phase
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
