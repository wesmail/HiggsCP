# Generic imports
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.metrics import roc_curve, auc

# PyTorch Lightning imports
from lightning.pytorch.callbacks import Callback


class TestCallback(Callback):
    def on_test_epoch_end(self, trainer, module):
        predictions = np.asarray(module.test_predictions)
        theta_true = np.asarray(module.test_theta)
        targets = np.asarray(module.test_targets)

        # --------------------------------------------------------------------------------
        # ROC Curve
        # --------------------------------------------------------------------------------
        # create ROC curve
        fpr, tpr, threshold = roc_curve(targets, predictions[:, 1])

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            color="b",
            label="GNN (area = {:.3f}%)".format(auc(fpr, tpr) * 100),
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.title("ROC curve")
        plt.grid(linestyle="--", color="k", linewidth=1.1)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_roc.png", dpi=300)

        # --------------------------------------------------------------------------------
        # Angular distribution
        # --------------------------------------------------------------------------------
        ang_dist = theta_true
        idx_pred = np.where(tpr > 0.5)  # TODO: remove hardcoded numbers
        idx_true = np.where(targets == 1)
        theta_model = theta_true[idx_pred]
        theta_true = theta_true[idx_true]

        plt.figure(figsize=(10, 7))
        plt.hist(
            theta_true.flatten()[: theta_model.shape[0]],
            bins=25,
            histtype="step",
            fill=False,
            density=True,
            label="Ideal dist",
            linewidth=2,
        )
        plt.hist(
            theta_model.flatten(),
            bins=25,
            histtype="step",
            fill=False,
            density=True,
            label=r"$tpr>0.5$",
            linewidth=2,
            linestyle="--",
        )
        plt.xlim(0, np.pi)
        plt.legend(fontsize=30, frameon=False)
        plt.grid(linestyle="--", color="k", linewidth=1.0, alpha=1)
        plt.xlabel(r"$\phi^\ast$", fontsize=25)
        plt.ylabel("Events (Normalized)", fontsize=25)
        plt.tick_params(axis="both", labelsize=20)
        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_angluar_distribution.png", dpi=300)

        # free up the memory
        module.test_predictions.clear()
        module.test_targets.clear()
        module.test_theta.clear()
