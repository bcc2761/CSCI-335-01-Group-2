import preprocessor as pp
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = pp.preprocess_data(
    scale=True,
    test_size=0.2,
    random_state=pp.RANDOM_STATE,
    visualize=False,
    explore=False,
    smote=False
  )

  X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=pp.RANDOM_STATE, stratify=y_train
  )

  X_train_split = X_train_split.values
  X_val = X_val.values
  X_test = X_test.values
  y_train_split = y_train_split.values.astype(int).ravel()
  y_val = y_val.values.astype(int).ravel()
  y_test = y_test.values.astype(int).ravel()

  # Determine number of classes
  num_classes = len(np.unique(y_train))
  input_dim = X_train.shape[1]

  print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
  print(f"Number of features: {input_dim}, Number of classes: {num_classes}")

  # Build MLP model
  model = keras.Sequential([
      layers.Input(shape=(input_dim,)),

      layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.0005)),
      layers.BatchNormalization(),
      layers.Dropout(0.4),
  
      layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.Dropout(0.3),

      layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.001)),
      layers.BatchNormalization(),
      layers.Dropout(0.2),

      layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.001)),
      layers.Dropout(0.1),
      layers.Dense(1, activation='sigmoid')
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.0008),
      loss=BinaryCrossentropy(from_logits=False),
      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
  )

  # Display model architecture
  print("Model Architecture:")
  model.summary()
  print()

  # Early stopping for better training
  callbacks = [
    EarlyStopping(
       monitor='val_loss', 
       patience=20, 
       restore_best_weights=True, 
       verbose=1,
       mode='min',
       min_delta=0.0001
       )
  ]

  # Train the model
  print("\nTraining MLP model...")
  history = model.fit(
      X_train_split, y_train_split,
      validation_data=(X_val, y_val),
      epochs=200,
      batch_size=128,
      callbacks=callbacks,
      verbose=1,
      class_weight={0: 1.0, 1: 4.0}  # penalize missing defaults more (FNs)
  )

  y_pred_proba = model.predict(X_test, verbose=0).ravel()   # shape (N,)

  threshold = 0.6
  y_pred = (y_pred_proba >= threshold).astype(int)
  
  test_auc = roc_auc_score(y_test, y_pred_proba)
  
  # Compute Precision-Recall metrics
  ap = average_precision_score(y_test, y_pred_proba)
  precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
  
  # Find threshold that maximizes F1
  f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
  best_f1_idx = np.nanargmax(f1_vals)
  if best_f1_idx == 0:
      best_threshold_f1 = 0.5
  else:
      best_threshold_f1 = pr_thresholds[best_f1_idx - 1]
  best_f1_score = f1_vals[best_f1_idx]
  
  # Metrics at best F1 threshold
  y_pred_best_f1 = (y_pred_proba >= best_threshold_f1).astype(int)

  print("\n===== Training Diagnostics =====")
  print(f"Final Training AUC: {history.history['auc'][-1]:.4f}")
  print(f"Final Validation AUC: {history.history['val_auc'][-1]:.4f}")
  print(f"Best Validation AUC: {max(history.history['val_auc']):.4f}")
  print(f"Epochs trained: {len(history.history['loss'])}")

  # Check if model is predicting variety
  print(f"\nPrediction distribution:")
  print(f"Min probability: {y_pred_proba.min():.4f}")
  print(f"Max probability: {y_pred_proba.max():.4f}")
  print(f"Mean probability: {y_pred_proba.mean():.4f}")
  print(f"Std probability: {y_pred_proba.std():.4f}")

  # Evaluate the model
  test_loss, test_accuracy, test_auc_eval = model.evaluate(X_test, y_test, verbose=0)
  train_loss, train_accuracy, train_auc_eval = model.evaluate(X_train_split, y_train_split, verbose=0)

  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  print("\n===== Final Results =====")
  print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
  print(f"Testing Accuracy: {test_accuracy:.4f}, Testing Loss: {test_loss:.4f}")
  print(f"\nRanking Metrics:")
  print(f"  ROC-AUC:  {test_auc:.4f}")
  print(f"  PR-AUC (Average Precision): {ap:.4f}")
  print(f"\nMetrics at threshold {threshold}:")
  print(f"  Precision: {precision:.4f}")
  print(f"  Recall:    {recall:.4f}")
  print(f"  F1-Score:  {f1:.4f}")
  print(f"\nMetrics at best F1 threshold ({best_threshold_f1:.3f}):")
  print(f"  Precision: {precision_score(y_test, y_pred_best_f1):.4f}")
  print(f"  Recall:    {recall_score(y_test, y_pred_best_f1):.4f}")
  print(f"  F1-Score:  {best_f1_score:.4f}")
  print(f"\nConfusion Matrix (threshold={threshold}):")
  print(cm)

  # Plot training history
  fig = plt.figure(figsize=(18, 12))

  # Plot 1: Confusion Matrix
  ax1 = plt.subplot(2, 3, 1)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
              xticklabels=['No Default', 'Default'],
              yticklabels=['No Default', 'Default'])
  ax1.set_title('Confusion Matrix (threshold=0.6)', fontsize=12, fontweight='bold')
  ax1.set_ylabel('True Label')
  ax1.set_xlabel('Predicted Label')

  # Plot 2: ROC Curve
  ax2 = plt.subplot(2, 3, 2)
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  ax2.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {test_auc:.4f}')
  ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
  ax2.set_xlabel('False Positive Rate')
  ax2.set_ylabel('True Positive Rate')
  ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  # Plot 3: Precision-Recall Curve
  ax3 = plt.subplot(2, 3, 3)
  ax3.plot(recall_vals, precision_vals, linewidth=2, label=f'PR-AUC = {ap:.4f}')
  ax3.axvline(recall_score(y_test, y_pred_best_f1), color='r', linestyle='--', 
              linewidth=1, label=f'Best F1 (thresh={best_threshold_f1:.3f})')
  ax3.set_xlabel('Recall')
  ax3.set_ylabel('Precision')
  ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
  ax3.legend()
  ax3.grid(True, alpha=0.3)
  ax3.set_xlim([0, 1])
  ax3.set_ylim([0, 1])

  # Plot 4: AUC curves (training history)
  ax4 = plt.subplot(2, 3, 4)
  ax4.plot(history.history['auc'], label='Training', linewidth=2)
  ax4.plot(history.history['val_auc'], label='Validation', linewidth=2)
  ax4.set_xlabel('Epoch')
  ax4.set_ylabel('AUC')
  ax4.set_title('Training vs Validation AUC', fontsize=12, fontweight='bold')
  ax4.legend()
  ax4.grid(True, alpha=0.3)

  # Plot 5: Loss curves
  ax5 = plt.subplot(2, 3, 5)
  ax5.plot(history.history['loss'], label='Training', linewidth=2)
  ax5.plot(history.history['val_loss'], label='Validation', linewidth=2)
  ax5.set_xlabel('Epoch')
  ax5.set_ylabel('Loss')
  ax5.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
  ax5.legend()
  ax5.grid(True, alpha=0.3)

  # Plot 6: Metrics comparison at different thresholds
  ax6 = plt.subplot(2, 3, 6)
  thresholds_to_plot = np.linspace(0, 1, 100)
  precisions_by_thresh = []
  recalls_by_thresh = []
  f1s_by_thresh = []
  for t in thresholds_to_plot:
      y_pred_t = (y_pred_proba >= t).astype(int)
      if y_pred_t.sum() > 0:  # avoid zero predictions
          precisions_by_thresh.append(precision_score(y_test, y_pred_t, zero_division=0))
          recalls_by_thresh.append(recall_score(y_test, y_pred_t, zero_division=0))
          f1s_by_thresh.append(f1_score(y_test, y_pred_t, zero_division=0))
      else:
          precisions_by_thresh.append(0)
          recalls_by_thresh.append(0)
          f1s_by_thresh.append(0)
  ax6.plot(thresholds_to_plot, precisions_by_thresh, label='Precision', linewidth=2)
  ax6.plot(thresholds_to_plot, recalls_by_thresh, label='Recall', linewidth=2)
  ax6.plot(thresholds_to_plot, f1s_by_thresh, label='F1-Score', linewidth=2)
  ax6.axvline(best_threshold_f1, color='r', linestyle='--', linewidth=1, label=f'Best F1')
  ax6.axvline(threshold, color='g', linestyle='--', linewidth=1, label=f'Current={threshold}')
  ax6.set_xlabel('Threshold')
  ax6.set_ylabel('Score')
  ax6.set_title('Metrics vs Classification Threshold', fontsize=12, fontweight='bold')
  ax6.legend()
  ax6.grid(True, alpha=0.3)
  ax6.set_xlim([0, 1])

  plt.tight_layout()
  plt.show()