import preprocessor as pp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = pp.preprocess_data(
    scale=True,
    test_size=0.2,
    random_state=pp.RANDOM_STATE,
    visualize=False,
    explore=False,
    smote=True
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
                  kernel_regularizer=regularizers.l2(0.005)),
      layers.Dropout(0.4),
      layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.005)),
      layers.Dropout(0.3),
      layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.005)),
      layers.Dropout(0.2),
      layers.Dense(1, activation='sigmoid')
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.0015),
      loss='binary_crossentropy',
      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
  )

  # Display model architecture
  print("Model Architecture:")
  model.summary()
  print()

  # Early stopping for better training
  callbacks = [
    EarlyStopping(
       monitor='val_auc', 
       patience=15, 
       restore_best_weights=True, 
       verbose=1,
       mode='max',
       min_delta=0.001
       )
  ]

  # Train the model
  print("\nTraining MLP model...")
  history = model.fit(
      X_train_split, y_train_split,
      validation_data=(X_val, y_val),
      epochs=200,
      batch_size=64,
      callbacks=callbacks,
      verbose=1
  )

  y_pred_proba = model.predict(X_test, verbose=0).ravel()   # shape (N,)
  y_pred = (y_pred_proba >= 0.5).astype(int)
  test_auc = roc_auc_score(y_test, y_pred_proba)

  # Evaluate the model
  test_loss, test_accuracy, test_auc_eval = model.evaluate(X_test, y_test, verbose=0)
  train_loss, train_accuracy, train_auc_eval = model.evaluate(X_train_split, y_train_split, verbose=0)

  print("\n===== Final Results =====")
  print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
  print(f"Testing Accuracy: {test_accuracy:.4f}, Testing Loss: {test_loss:.4f}")
  print(f"Testing AUC: {test_auc:.4f}")

  # Plot training history
  fig = plt.figure(figsize=(18, 10))

  # Plot 1: Confusion Matrix
  ax1 = plt.subplot(2, 3, 1)
  cm = confusion_matrix(y_test, y_pred)
  ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
  ax1.set_ylabel('True Label')
  ax1.set_xlabel('Predicted Label')

  # Plot 2: ROC Curve
  ax2 = plt.subplot(2, 3, 2)
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {test_auc:.4f}')
  ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
  ax2.set_xlabel('False Positive Rate')
  ax2.set_ylabel('True Positive Rate')
  ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  # Plot 3: Accuracy curves
  ax3 = plt.subplot(2, 3, 3)
  ax3.plot(history.history['accuracy'], label='Training', linewidth=2)
  ax3.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
  ax3.set_xlabel('Epoch')
  ax3.set_ylabel('Accuracy')
  ax3.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
  ax3.legend()
  ax3.grid(True, alpha=0.3)

  # Plot 4: AUC curves
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

  # Plot 6: Class distribution
  ax6 = plt.subplot(2, 3, 6)
  unique, counts = np.unique(y_test, return_counts=True)
  ax6.bar(['No Default', 'Default'], counts, color=['green', 'red'], alpha=0.6)
  ax6.set_ylabel('Count')
  ax6.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
  for i, v in enumerate(counts):
      ax6.text(i, v, str(v), ha='center', va='bottom')

  plt.tight_layout()
  plt.show()
