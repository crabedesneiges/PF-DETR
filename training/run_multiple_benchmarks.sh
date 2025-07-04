#!/bin/bash

# Liste des configs à utiliser
CONFIGS=(
    "base_20k_private.yaml"
    "base_40k_private.yaml"
    "base_60k_private.yaml"
)

# Specify the starting GPU index
START_GPU=3

# Configuration MLflow
MLFLOW_URI="mlruns"  # Stockage local, ou utilisez http://localhost:5000 si le serveur est démarré

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Démarrer les entraînements avec MLflow
echo "Démarrage des benchmarks avec tracking MLflow (URI: $MLFLOW_URI)"

for i in ${!CONFIGS[@]}; do
    gpu_id=$((START_GPU + i))
    config_name=$(basename "${CONFIGS[$i]}" .yaml)
    log_file="logs/mlflow_${config_name}_gpu${gpu_id}.log"
    
    echo "Lancement de ${config_name} sur GPU ${gpu_id}, log: ${log_file}"
    
    # Lancer l'entraînement avec MLflow tracking
    python train_V3.py \
        --config "config/private/${CONFIGS[$i]}" \
        --gpu-device $gpu_id \
        > "$log_file" 2>&1 &
    
    # Attendre un peu pour éviter des problèmes d'accès concurrents
    sleep 2
done

# Attendre que tous les processus se terminent
wait
echo "Tous les benchmarks sont terminés."
echo "Pour visualiser les résultats, lancez: ./start_mlflow_server.sh"
