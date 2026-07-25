import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Evaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_detection(self, y_true_binary, y_scores, y_pred_binary):
        f1 = f1_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_true_binary, y_scores)
            auc_pr = average_precision_score(y_true_binary, y_scores)
        except Exception:
            auc_roc = 0.0
            auc_pr = 0.0
            
        # FPR at top 1% alert budget
        threshold = np.percentile(y_scores, 100 * (1 - config.ALERT_BUDGET_PERCENTILE))
        y_pred_budget = (y_scores >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_budget, labels=[0, 1]).ravel()
        fpr_at_budget = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        res = {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "fpr_at_1_percent_budget": float(fpr_at_budget)
        }
        self.metrics["detection"] = res
        return res

    def evaluate_classification(self, y_true_multiclass, y_pred_multiclass, y_proba=None):
        res = {
            "f1_macro": float(f1_score(y_true_multiclass, y_pred_multiclass, average='macro', zero_division=0)),
            "f1_weighted": float(f1_score(y_true_multiclass, y_pred_multiclass, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_true_multiclass, y_pred_multiclass, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_true_multiclass, y_pred_multiclass, average='macro', zero_division=0))
        }
        
        # Per class metrics
        f1_per_class = f1_score(y_true_multiclass, y_pred_multiclass, average=None, zero_division=0)
        res["f1_per_class"] = f1_per_class.tolist()
        
        cm = confusion_matrix(y_true_multiclass, y_pred_multiclass)
        res["confusion_matrix"] = cm.tolist()
        
        self.metrics["classification"] = res
        return res

    def plot_roc_curve(self, y_true, y_scores, save_path):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_scores):.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_scores, save_path):
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        plt.figure()
        plt.plot(rec, prec, label=f"AP = {average_precision_score(y_true, y_scores):.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_risk_score_distribution(self, scores, labels, save_path):
        plt.figure()
        sns.histplot(x=scores, hue=labels, bins=50, kde=True)
        plt.xlabel('Risk Score')
        plt.title('Risk Score Distribution')
        plt.savefig(save_path)
        plt.close()

    def generate_all_plots(self, results, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        if "y_true_binary" in results and "y_scores" in results:
            self.plot_roc_curve(results["y_true_binary"], results["y_scores"], os.path.join(save_dir, "roc_curve.png"))
            self.plot_precision_recall_curve(results["y_true_binary"], results["y_scores"], os.path.join(save_dir, "pr_curve.png"))
            self.plot_risk_score_distribution(results["y_scores"], results["y_true_binary"], os.path.join(save_dir, "risk_scores.png"))
            
        if "y_true_multiclass" in results and "y_pred_multiclass" in results:
            self.plot_confusion_matrix(
                results["y_true_multiclass"], 
                results["y_pred_multiclass"], 
                config.ALL_LABELS, 
                os.path.join(save_dir, "confusion_matrix.png")
            )

    def save_metrics(self, metrics_dict, path):
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
