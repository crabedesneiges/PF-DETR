import pytorch_lightning as pl
import os
import csv

class LossHistoryCSVCallback(pl.Callback):
    """
    Callback qui sauvegarde train_loss et val_loss à chaque epoch dans un CSV.
    """
    def __init__(self, filename="loss_history.csv"):
        super().__init__()
        self.filename = filename
        self.loss_history = []

    def on_validation_end(self, trainer, pl_module):
        # Récupère les métriques de la fin d'epoch
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        row = {"epoch": epoch}
        for key in ["train_loss", "val_loss"]:
            if key in metrics:
                row[key] = float(metrics[key])
        # Ajoute toutes les accuracy par classe si présentes
        for k in metrics:
            if k.startswith("train_loss_acc_class_") or k.startswith("val_loss_acc_class_"):
                row[k] = float(metrics[k])
        self.loss_history.append(row)

    def on_train_end(self, trainer, pl_module):
        # Sauvegarde le CSV à la fin du training
        if not self.loss_history:
            return
        log_dir = trainer.logger.log_dir if trainer.logger else "."
        filepath = os.path.join(log_dir, self.filename)
        # Collecte tous les champs utilisés dans l'historique
        all_keys = set()
        for row in self.loss_history:
            all_keys.update(row.keys())
        keys = ["epoch"] + sorted(k for k in all_keys if k != "epoch")
        # Remplit les valeurs manquantes avec ''
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for row in self.loss_history:
                full_row = {k: row.get(k, '') for k in keys}
                writer.writerow(full_row)
        print(f"[LossHistoryCSVCallback] Courbe sauvegardée dans {filepath}")
