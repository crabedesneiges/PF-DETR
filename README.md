Of course, here is the refactored README file.

# pflow-DETR: Particle Flow with DETR

\<div align="center"\>

\</div\>

A deep learning framework for particle flow reconstruction in high-energy physics, leveraging the power of the DETR (DEtection TRansformer) architecture.

-----

## 🚀 Introduction

**pflow-DETR** is a framework designed for particle classification and parameter reconstruction in particle physics experiments. It utilizes a transformer-based approach, specifically the DETR model, to analyze high-energy collision events.

### Key Features

  * **Two DETR Versions**: Implements two iterations of the DETR architecture with ongoing improvements.
  * **Experiment Tracking**: Integrates seamlessly with Tensorboard for tracking and managing experiments.
  * **Visualization Tools**: Offers utilities for visualizing results, statistics, and event displays.

-----

## 🛠️ Installation

### Prerequisites

  * Python 3.8+
  * PyTorch 2.0+
  * CUDA (recommended for GPU acceleration)

### Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/username/pflow_DETR_V3.git
    cd pflow_DETR_V3
    ```

2.  **Install dependencies:**

      * **Option 1: Direct Installation**
        ```bash
        pip install -r requirements.txt
        ```
      * **Option 2: Virtual Environment (Recommended)**
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        pip install -r requirements.txt
        ```

-----

## 📂 Project Structure

```
.
├── callbacks/          # Custom PyTorch Lightning callbacks
├── config/             # YAML configuration files
├── data/               # Data loading and preparation scripts
├── model/              # DETR model architecture definitions
├── scripts/            # Utility, analysis, and visualization scripts
├── training/           # Training, evaluation, and comparison tools
├── mlruns/             # (Auto-generated) MLflow experiment data
├── lightning_logs/     # (Auto-generated) PyTorch Lightning logs
├── checkpoints/        # (Auto-generated) Model checkpoints
├── train.py            # Main script for training models
├── mlflow_experiment.py # Script for running multiple experiments
└── README.md           # This file
```

-----

## ⚙️ Usage

### Training a Model

  * **Single Run**: Train a model using a specific configuration file on a designated GPU.

    ```bash
    python training/train.py --gpu-device 0 --config config/test.yaml
    ```

      * `--gpu-device`: The ID of the GPU to use.
      * `--config`: Path to the YAML configuration file.
      * `--resume`: Resule from checkpoints.

### Experiment Tracking and Visualization

  * **TensorBoard**: Launch TensorBoard to monitor training progress in real-time.

    ```bash
    tensorboard --logdir lightning_logs --port 6006
    ```

    Access the dashboard at `http://localhost:6006`.

### Evaluation and Analysis

  * **Model Evaluation**: Evaluate a trained model's performance on the test or training dataset.

    ```bash
    python training/eval_model.py \
        --config_path config/my_benchmark/base.yaml \
        --checkpoint_path checkpoints/base/model.ckpt \
        --cuda_visible_device "1"
    ```

  * **Result Visualization**: Generate plots and visualizations from an evaluation output file.

    ```bash
    python training/visualize.py \
        --inputfile workspace/eval/my_experiment.npz \
        --outputdir workspace/my_experiment/plots \
        --conf-threshold 0.75
    ```

  * **Event Display**: Create visual representations of individual collision events.

    ```bash
    python scripts/eventdisplay.py \
        --config_path config/my_benchmark/base.yaml \
        --checkpoint_path checkpoints/base/model.ckpt \
        --outputdir workspace/my_experiment/events \
        --conf-threshold 0.75
    ```

-----

## 🔬 Advanced Analysis

### Model and Performance Comparison

This framework includes scripts to compare pflow-DETR with other models (like HGP) or to compare different versions of pflow-DETR against each other.

  * **Particle-Level Comparison**:

    ```bash
    python training/compare_particle_level.py \
        --model1_file workspace/eval/detr_model_A.npz --model1_name "DETR_A" \
        --model2_file workspace/eval/detr_model_B.npz --model2_name "DETR_B" \
        --outputdir workspace/comparison/particle_level/ \
        --detr_conf_threshold 0.75
    ```

  * **Jet-Level Comparison**:

    ```bash
    python scripts/compare_jet_clustering.py \
        --model1_inputfile workspace/eval/detr_model_A.npz --model1_name "DETR_A" \
        --model2_inputfile workspace/original_hgp_results.npz --model2_name "HGP" \
        --outputdir workspace/comparison/jet_level/ \
        --detr_conf_threshold 0.75
    ```


## 📚 References

  * [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
  * [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  * [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
  * [HyperGraph DNN for Particle Flow (source)](https://github.com/saitoicepp/hg-tspn-pflow-simple)