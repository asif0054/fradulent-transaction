"""
╔══════════════════════════════════════════════════════════════════════════╗
║        STEP 5: Evaluation & Visualization Utilities                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  This module provides all evaluation metrics and visualizations:       ║
║                                                                        ║
║  1. ROC CURVE        — True Positive Rate vs False Positive Rate       ║
║  2. CONFUSION MATRIX — Visual heatmap of predictions vs reality        ║
║  3. CLASSIFICATION REPORT — Precision, Recall, F1-Score per class      ║
║  4. FEATURE IMPORTANCE — Bar chart of top features                     ║
║  5. MODEL COMPARISON  — Side-by-side metrics table                     ║
║                                                                        ║
║  UNDERSTANDING THE METRICS:                                            ║
║  ──────────────────────────                                            ║
║  • Precision — Of all "fraud" predictions, how many were actually      ║
║                fraud? (Avoid false alarms)                              ║
║  • Recall    — Of all actual frauds, how many did we catch?            ║
║                (Avoid missed frauds — MOST IMPORTANT!)                 ║
║  • F1-Score  — Harmonic mean of Precision & Recall                     ║
║  • AUC-ROC   — Area Under the ROC Curve (1.0 = perfect, 0.5 = random) ║
║                                                                        ║
║  USAGE:                                                                ║
║    from utils.evaluation import plot_roc_curve, plot_confusion_matrix  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import os

# ──────────────────────────────────────────────────────────────────────
# STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'primary': '#6C63FF',    # Purple accent
    'secondary': '#FF6584',  # Pink accent
    'success': '#2ECC71',    # Green
    'danger': '#E74C3C',     # Red
    'bg_dark': '#0E1117',    # Dark background
    'bg_card': '#1E2130',    # Card background
    'text': '#FAFAFA',       # White text
    'grid': '#2D3250',       # Grid lines
}


def plot_roc_curve(y_true, y_scores, model_name="XGBoost", save_path=None):
    """
    STEP 5a: Plot the ROC (Receiver Operating Characteristic) Curve.
    
    Parameters:
        y_true: True labels (0 or 1)
        y_scores: Predicted probabilities or decision scores
        model_name: Name for the legend
        save_path: Where to save the plot image
        
    Returns:
        fig: matplotlib Figure object
        roc_auc: Area Under the ROC Curve score
    
    WHAT IS THE ROC CURVE?
        ─────────────────────
        • X-axis: False Positive Rate (FPR) — legit transactions flagged as fraud
        • Y-axis: True Positive Rate (TPR) — actual frauds correctly caught
        • The diagonal line = random guessing (AUC = 0.5)
        • A perfect model hugs the top-left corner (AUC = 1.0)
        • The more area under the curve, the better the model
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_card'])
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
            label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier baseline)
    ax.plot([0, 1], [0, 1], color=COLORS['danger'], linestyle='--', 
            linewidth=1, alpha=0.7, label='Random Classifier (AUC = 0.50)')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
    
    # Labels and styling
    ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['text'])
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14, 
                  fontweight='bold', color=COLORS['text'], pad=15)
    ax.legend(loc='lower right', fontsize=10, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg_dark'])
        print(f"   📊 ROC curve saved to: {save_path}")
    
    return fig, roc_auc


def plot_multi_roc(results_dict, y_true, save_path=None):
    """
    STEP 5b: Plot multiple ROC curves on one chart for model comparison.
    
    Parameters:
        results_dict: Dict of {model_name: y_scores}
        y_true: True labels
        save_path: Where to save
        
    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_card'])
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    
    for i, (name, scores) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    
    ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.3, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['text'])
    ax.set_title('ROC Curve Comparison', fontsize=14,
                  fontweight='bold', color=COLORS['text'], pad=15)
    ax.legend(loc='lower right', fontsize=10, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg_dark'])
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name="XGBoost", save_path=None):
    """
    STEP 5c: Plot a confusion matrix heatmap.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels (binary)
        model_name: Name for the title
        save_path: Where to save
        
    Returns:
        fig: matplotlib Figure
        cm: The confusion matrix array
    
    READING THE CONFUSION MATRIX:
        ────────────────────────────
        
                         Predicted
                      Legit    Fraud
        Actual Legit  [ TN  |  FP  ]   TN = True Negatives (correct legit)
        Actual Fraud  [ FN  |  TP  ]   FP = False Positives (false alarm!)
                                        FN = False Negatives (missed fraud!)
                                        TP = True Positives (caught fraud)
        
        We want: HIGH TN + HIGH TP (diagonal = correct predictions)
        We avoid: HIGH FN (missed frauds are costly!)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                linewidths=2, linecolor=COLORS['bg_dark'],
                annot_kws={"size": 16, "fontweight": "bold"},
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, color=COLORS['text'], labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, color=COLORS['text'], labelpad=10)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14,
                  fontweight='bold', color=COLORS['text'], pad=15)
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg_dark'])
        print(f"   📊 Confusion matrix saved to: {save_path}")
    
    return fig, cm


def plot_feature_importance(importance_dict, top_n=15, save_path=None):
    """
    STEP 5d: Plot a horizontal bar chart of feature importance.
    
    Parameters:
        importance_dict: Dict of {feature_name: importance_score}
        top_n: How many top features to show
        save_path: Where to save
        
    Returns:
        fig: matplotlib Figure
        
    WHAT IS FEATURE IMPORTANCE?
        ──────────────────────────
        • Shows which features XGBoost used most in its decision trees
        • Higher importance = feature had more influence on predictions
        • Helps understand WHAT makes a transaction look fraudulent
        • e.g., V14, V17, V12 are often top features for fraud detection
    """
    # Sort and take top N
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_features][::-1]
    scores = [f[1] for f in sorted_features][::-1]
    
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_card'])
    
    # Create gradient bars
    colors = plt.cm.RdPu(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, scores, color=colors, edgecolor='none', height=0.6)
    
    ax.set_xlabel('Importance Score', fontsize=12, color=COLORS['text'])
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14,
                  fontweight='bold', color=COLORS['text'], pad=15)
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, axis='x', alpha=0.2, color=COLORS['grid'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['bg_dark'])
    
    return fig


def get_metrics_summary(y_true, y_pred, y_scores=None):
    """
    STEP 5e: Calculate a comprehensive metrics summary.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Probability scores (for AUC calculation)
        
    Returns:
        dict with all metrics
    """
    metrics = {
        "accuracy": float(accuracy_score_calc(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics["auc_roc"] = float(auc(fpr, tpr))
    
    return metrics


def accuracy_score_calc(y_true, y_pred):
    """Helper: Calculate accuracy."""
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def get_classification_report_dict(y_true, y_pred):
    """
    Get classification report as a dictionary for display.
    """
    report = classification_report(y_true, y_pred,
                                    target_names=["Legitimate", "Fraud"],
                                    output_dict=True,
                                    zero_division=0)
    return report


# ──────────────────────────────────────────────────────────────────────
# MAIN — Run standalone to test visualizations
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.array([0]*90 + [1]*10)
    y_scores = np.random.rand(100)
    y_pred = (y_scores > 0.5).astype(int)
    
    print("📊 Testing evaluation utilities...")
    
    fig_roc, auc_val = plot_roc_curve(y_true, y_scores, save_path="test_roc.png")
    print(f"   AUC: {auc_val:.4f}")
    
    fig_cm, cm = plot_confusion_matrix(y_true, y_pred, save_path="test_cm.png")
    print(f"   Confusion Matrix:\n{cm}")
    
    print("   ✓ All evaluation functions working!")
    plt.close('all')
