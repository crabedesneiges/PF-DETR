# pflow-DETR: Particle Flow with DETR

\<div align="center"\>

\</div\>

A deep learning framework for particle flow reconstruction in high-energy physics, leveraging the power of the DETR (DEtection TRansformer) architecture.

-----

## 🚀 Introduction

**pflow-DETR** is a framework designed for particle classification and parameter reconstruction in particle physics experiments. It utilizes a transformer-based approach, specifically the DETR model, to analyze high-energy collision events.

### Key Features

  * **PF-DETR Implementation**: Implements a transformer architecture inspired by DETR architecture with ongoing improvements.
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
    git clone https://github.com/crabedesneiges/PF-DETR.git
    cd PF-DETR
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
├── training/           # Training, evaluation tools
├── web_plot/           # Streamlit web application for visualization
├── lightning_logs/     # (Auto-generated) PyTorch Lightning logs
├── checkpoints/        # (Auto-generated) Model checkpoints
└── README.md           # This file
```

-----

## ⚙️ Usage

### 1\. Generate Normalization Parameters

This only needs to be run once before the first training.

```bash
python scripts/make_normalization_params.py --inputfilename '/data/multiai/data3/HyperGraph-2212.01328/singleQuarkJet_train.root'
```

### 2\. Training a Model

  * **Start a new training run**: Train a model using a specific configuration file on a designated GPU.

    ```bash
    python -m training.train --gpu-device 0 --config config/test.yaml
    ```

  * **Resume from a checkpoint**:

    ```bash
    python -m training.train --gpu-device 0 --config config/test.yaml --resume checkpoints/chek_file.ckpt
    ```

      * `--gpu-device`: The ID of the GPU to use.
      * `--config`: Path to the YAML configuration file.
      * `--resume`: Path to the model checkpoint file to resume training.

### 3\. Model Evaluation

Evaluate a trained model's performance and save the output to a `.npz` file in the `workspace/npz/` directory.

```bash
python training/eval_model.py \
    --config_path config/test.yaml \
    --checkpoint_path checkpoints/test/detr-epochepoch=24-val_loss=1.9253.ckpt \
    --cuda_visible_device "1"
```

*Additional options:*

  * `--seed`: Set a random seed for reproducibility.
  * `--eval_train`: Evaluate on the training dataset instead of the test set.
  * `--batchsize`: Override the batch size specified in the config file.

### 4\. Experiment Tracking

Launch TensorBoard to monitor training progress in real-time.

```bash
tensorboard --logdir lightning_logs --port 6006
```

Access the dashboard at `http://localhost:6006`.

-----

## 🔬 Advanced Analysis & Visualization

### Model and Performance Comparison

Compare pflow-DETR with other models (like HGP) or different pflow-DETR runs.

  * **Particle-Level Comparison**:

    ```bash
    python scripts/compare_particle_level.py \
        --model1_file workspace/npz/test_test.npz \
        --model1_type detr \
        --model1_name "test" \
        --model2_file workspace/npz/original_refiner.npz \
        --model2_type hgp \
        --model2_name "hgp" \
        --outputdir workspace/particle/test/ \
        --detr_conf_threshold 0.75
    ```

  * **Jet-Level Comparison**:

    ```bash
    python scripts/compare_jet_clustering.py \
        --model1_inputfile workspace/npz/test_test.npz --model1_name "detr_test" \
        --model2_inputfile workspace/npz/original_refiner.npz --model2_name "hgp" \
        --outputdir workspace/jet_level/test/ \
        --detr_conf_threshold 0.75
    ```

### Interactive Visualization Web App

Use the Streamlit web app to view plots at the Particle and Jet Level and compare multiple runs.

```bash
streamlit run web_plot/app.py
```

### Static Visualization

  * **Event Display**: Create visual representations of individual collision events.

    ```bash
    python scripts/eventdisplay.py \
        --config_path config/my_benchmark/base.yaml \
        --checkpoint_path checkpoints/base/model.ckpt \
        --outputdir workspace/my_experiment/events \
        --conf-threshold 0.75
    ```

  * **Result Visualization**: Generate plots from an evaluation output file.

    ```bash
    python training/visualize.py \
        --inputfile workspace/eval/my_experiment.npz \
        --outputdir workspace/my_experiment/plots \
        --conf-threshold 0.75
    ```

-----

## 📚 References

  * [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
  * [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  * [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
  * [HyperGraph DNN for Particle Flow (source)](https://github.com/saitoicepp/hg-tspn-pflow-simple)