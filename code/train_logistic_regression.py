import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
    Plots and saves the confusion matrix for the model.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    y_pred = model.predict(X_test)
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['No Default (0)', 'Default (1)'], cmap=plt.cm.Blues, normalize='true'
    )
    cm_display.ax_.set_title('Logistic Regression Confusion Matrix (Normalized)')
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

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

    # 5. Plot and save confusion matrix
    plot_model(model, X_test, y_test)

if __name__ == "__main__":
    main()