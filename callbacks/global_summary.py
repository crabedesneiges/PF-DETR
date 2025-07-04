import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

class GlobalSummaryCallback(pl.Callback):
    """
    Callback qui affiche un résumé global des métriques train, val, test à la fin du test.
    """
    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        summary = []
        for key in ['train_loss', 'val_loss', 'test_loss']:
            if key in metrics:
                summary.append(f"{key}: {metrics[key]:.4f}")
        if summary:
            rank_zero_info("\n===== BILAN GLOBAL DES METRIQUES =====\n" + '\n'.join(summary) + "\n====================================\n")
        else:
            rank_zero_info("\nAucune métrique à afficher dans le bilan global.\n")
