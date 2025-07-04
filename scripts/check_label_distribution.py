import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from data.DataLoader import PFlowDataset
from data.pflow_datamodule import PFlowDataModule
import yaml

# Charge la config YAML
with open("config/test.yaml") as f:
    config = yaml.safe_load(f)

dm = PFlowDataModule(config)
dm.setup(stage="fit")
test_loader = dm.train_dataloader()

all_labels = []
padding_idx = config["model"].get("num_classes", 5)  # Par défaut 5
num_classes = padding_idx + 1

# For progressive plotting
progress_counts = []
running_counts = np.zeros(num_classes, dtype=int)
sample_steps = []
samples_seen = 0

for batch_idx, batch in enumerate(test_loader):
    if batch_idx == 1:
        break
    print("eta", batch["target"]["boxes"][0][0])
    print("track", batch["input"].keys())
    labels = batch["target"]["labels"]  # (B, num_queries)
    labels = labels.cpu().numpy().ravel()
    all_labels.extend(labels.tolist())
    # Update running counts
    counts_this = np.bincount(labels, minlength=num_classes)
    running_counts += counts_this
    samples_seen += len(labels)
    # Save every N batches (or every batch if small)
    if (batch_idx % 10 == 0) or (batch_idx == 0):
        progress_counts.append(running_counts.copy())
        sample_steps.append(samples_seen)

# Compte par classe
label_counts = Counter(all_labels)
print("Distribution des labels (incluant padding):")
for c in range(padding_idx+1):
    print(f"Classe {c}: {label_counts.get(c, 0)}")

if padding_idx in label_counts:
    print(f"(padding_idx={padding_idx}, count={label_counts[padding_idx]})")
else:
    print(f"(padding_idx={padding_idx}, count=0)")

# Pour voir les classes présentes hors padding
classes_present = [c for c in range(padding_idx) if label_counts.get(c, 0) > 0]
print("Classes présentes dans le test set (hors padding):", classes_present)

# --- Plot progressive class counts ---
progress_counts = np.stack(progress_counts, axis=0)  # (steps, num_classes)
plt.figure(figsize=(10,6))
for c in range(num_classes):
    plt.plot(sample_steps, progress_counts[:,c], label=f"Classe {c}")
plt.xlabel("Nombre de labels vus")
plt.ylabel("Nombre cumulé par classe")
plt.title("Progression du nombre de labels vus par classe dans le dataset")
plt.legend()
plt.tight_layout()
plt.savefig("label_distribution_progression.png")
print("\nCourbe de progression sauvegardée dans label_distribution_progression.png")
