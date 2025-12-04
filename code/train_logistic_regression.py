import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
import joblib
import os
from matplotlib import pyplot as plt
import seaborn as sns

# Import 'preprocessor.py' assuming it is in the same 'code/' directory
try:
    import preprocessor as pp
except ImportError:
    print("Error: 'preprocessor.py' not found.")
    print("Make sure 'preprocessor.py' is in the 'code/' directory.")
    exit()

RANDOM_STATE = 35

def train_model(X_train, y_train):
    """
    Trains the Logistic Regression model.
    
    Args:
        X_train (pd.DataFrame): Preprocessed and resampled training features.
        y_train (pd.DataFrame): Resampled training target.
    
    Returns:
        model: The trained scikit-learn model.
    """
    print("Training Logistic Regression model...")
    
    # Initialize the model
    # We don't use class_weight='balanced' because SMOTE already handled the imbalance.
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    
    # Train the model
    # Use .values.ravel() to convert the y_train DataFrame back to a 1D array
    model.fit(X_train, y_train.values.ravel())
    
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the (unbalanced) test set.
    """
    print("\n" + "=" * 30)
    print("Model Evaluation on Test Set")
    print("=" * 30)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities for AUC-ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification Report 
    # This is the most important metric for imbalanced data
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default (0)', 'Default (1)']))
    
    # --- Confusion Matrix ---
    print("Confusion Matrix:")
    print("          [Pred No] [Pred Yes]")
    cm = confusion_matrix(y_test, y_pred)
    print(f"[True No]   {cm[0][0]:<7} {cm[0][1]}")
    print(f"[True Yes]  {cm[1][0]:<7} {cm[1][1]}")
    
    # AUC-ROC Score; key metric for imbalanced classification
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC Score: {auc:.4f}")

def save_model(model, path="models/logistic_regression.joblib"):
    """
    Saves the trained model to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")

def plot_model(model, X_test, y_test, save_path='./data/logistic_regression_confusion_matrix.png'):
    """
    Create a single 2x2 figure containing:
      - Confusion matrix (top-left)
      - ROC curve (top-right)
      - Precision-Recall curve (bottom-left)
      - Precision / Recall / F1 vs Threshold (bottom-right)

    Saves the figure to `save_path` or a sibling path in `./data/metrics/`.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs(os.path.dirname(save_path) or './data', exist_ok=True)
    save_dir = os.path.join(os.path.dirname(save_path), 'metrics')
    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, 'logistic_regression_metrics_overview.png')

    # Predictions and probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Confusion matrix (non-normalized counts for clarity)
    cm = confusion_matrix(y_test, y_pred)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Precision-Recall
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # Metrics vs threshold
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        preds_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, preds_t, zero_division=0))
        recalls.append(recall_score(y_test, preds_t, zero_division=0))
        f1s.append(f1_score(y_test, preds_t, zero_division=0))

    best_idx = int(np.nanargmax(f1s))
    best_threshold = thresholds[best_idx]

    # Build the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Top-left: Confusion matrix (counts)
    ax = axes[0, 0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default (0)', 'Default (1)'])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title('Confusion Matrix (threshold=0.5)')

    # Top-right: ROC curve
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Bottom-left: Precision-Recall curve
    ax = axes[1, 0]
    ax.plot(recall_vals, precision_vals, lw=2, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    # Bottom-right: Precision / Recall / F1 vs Threshold
    ax = axes[1, 1]
    ax.plot(thresholds, precisions, label='Precision', lw=2)
    ax.plot(thresholds, recalls, label='Recall', lw=2)
    ax.plot(thresholds, f1s, label='F1', lw=2)
    ax.axvline(best_threshold, color='r', linestyle='--', label=f'Best F1 thresh={best_threshold:.3f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 vs Threshold')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Combined metrics figure saved to {outpath}")

def main():
    """
    Main workflow for training and evaluating the model.
    """
    # 1. Load and preprocess data using script
    # We want scaling and SMOTE, but can turn off visualization
    # to keep the log clean during training.
    X_train, X_test, y_train, y_test = pp.preprocess_data(
        scale=True,
        smote=True,
        visualize=False,  # Set to False to speed up training
        explore=False     # Set to False to speed up training
    )
    
    # 2. Train the model
    model = train_model(X_train, y_train)
    
    # 3. Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # 4. Save the model
    save_model(model)

    # 5. Plot and save combined metrics figure (confusion matrix, ROC, PR, threshold curves)
    plot_model(model, X_test, y_test)

if __name__ == "__main__":
    main()