import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATASET_ID = 350

def download_dataset():
  """
  Downloads the Default of Credit Card Clients Dataset from the UCI Machine Learning Repository.
  
  Returns:
    X (pd.DataFrame): Features of the dataset.

    y (pd.Series): Target variable indicating default status.
  """
  default_of_credit_card_clients = fetch_ucirepo(id=DATASET_ID) 
  X = default_of_credit_card_clients.data.features
  y = default_of_credit_card_clients.data.targets
  
  print(f"Dataset shape: {X.shape}")
  print(f"Target shape: {y.shape}")
  print(f"Default rate: {y.values.mean():.2%}")
  
  return X, y

def show_dataset_info(X, y):
  """
  Displays information about the dataset including feature types and missing values.
  
  Args:
    X (pd.DataFrame): Features of the dataset.
    y (pd.Series): Target variable.
  """
  print("\n===== Dataset Information =====")
  print(X.info())

  print("\n===== Target Distribution =====")
  print(y.value_counts(normalize=True))
  print(f"Default rate: {y.values.mean():.2%}")

  print("\n===== Missing Values =====")
  print(f"Total mussing values: {X.isnull().sum().sum()}")

  print("\n===== Statistical Summary =====")
  print(X.describe())


def clean_categorical_features(X):
  """
  Cleans and encodes categorical features in the dataset.
  
  Args:
    X (pd.DataFrame): Features of the dataset.
  
  Returns:
    X_cleaned (pd.DataFrame): Cleaned and encoded features.
  """
  X_cleaned = X.copy()
  
  # clean EDUCATION (0, 5, 6, are mapped to 'others' = 4)
  if 'EDUCATION' in X_cleaned.columns:
    X_cleaned['EDUCATION'] = X_cleaned['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
  
  # clean MARRIAGE (0 is mapped to 'others' = 3)
  if 'MARRIAGE' in X_cleaned.columns:
    X_cleaned['MARRIAGE'] = X_cleaned['MARRIAGE'].replace({0: 3})

  print("\nCategorical features cleaned.")
  return X_cleaned

def get_train_test_split(X, y, test_size=0.2, random_state=35):
  """
  Splits the dataset into training and testing sets.
  
  Args:
    X (pd.DataFrame): Features of the dataset.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

  Returns:
    X_train (pd.DataFrame): Training features.

    X_test (pd.DataFrame): Testing features.

    y_train (pd.Series): Training target.

    y_test (pd.Series): Testing target.
  """
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
  )
  
  print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
  return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
  """
  Scales numerical features using StandardScaler.
  
  Args:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
  
  Returns:
    X_train_scaled (pd.DataFrame): Scaled training features.

    X_test_scaled (pd.DataFrame): Scaled testing features.
  """
  scaler = StandardScaler()

  categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
  numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
  
  # Create copies
  X_train_scaled = X_train.copy()
  X_test_scaled = X_test.copy()
  
  # Fit on training data only, transform both training and test data
  X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
  X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
  
  print("\n===== Feature Scaling Completed =====")
  print(f"Scaled {len(numeric_cols)} numerical features")

  return X_train_scaled, X_test_scaled

def visualize_dataset(X, y, save_path='dataset_overview.png'):
  """
    Create visualizations of the dataset.
    
    Args:
      X: Features DataFrame
      y: Target Series
      save_path: Path to save the visualization (default: 'dataset_overview.png')
  """
  plt.figure(figsize=(12, 6))
  
  # Plot distribution of target variable
  plt.subplot(1, 3, 1)
  y.value_counts().plot(kind='bar', color=["#b5e9a8", "#ee867a"])
  plt.title("Target Distrubution")
  plt.xlabel('Default (0=No, 1=Yes)')
  plt.ylabel('Count')
  plt.xticks(rotation=0)

  # Plot distribution of AGE
  plt.subplot(1, 3, 2)
  X['X5'].hist(bins=30, color='#3498db', edgecolor='black')
  plt.title('Age Distribution')
  plt.xlabel('Age')
  plt.ylabel('Frequency')
  
  # Plot distribution of LIMIT_BAL
  plt.subplot(1, 3, 3)
  X['X1'].hist(bins=50, color='#9b59b6', edgecolor='black')
  plt.title('Credit Limit Distribution')
  plt.xlabel('Credit Limit (NT$)')
  plt.ylabel('Frequency')
  
  plt.tight_layout()
  plt.savefig(save_path, dpi=100, bbox_inches='tight')
  print(f"\nVisualization saved as '{save_path}'")
  plt.close()
  
  print(f"\nDataset visualization saved to {save_path}")

def preprocess_data(scale=True, test_size=0.2, random_state=35, visualize=True, explore=True):
  """
  Main function to preprocess the Default of Credit Card Clients Dataset.
  
  Args:
    scale (bool): Whether to scale numerical features.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    visualize (bool): Whether to create visualizations of the dataset.
    explore (bool): Whether to display dataset information.
  
  Returns:
    X_train (pd.DataFrame): Preprocessed training features.

    X_test (pd.DataFrame): Preprocessed testing features.

    y_train (pd.Series): Training target.

    y_test (pd.Series): Testing target.
  """

  print("=" * 30)
  print("Credit Card Default Dataset Preprocessing")
  print("=" * 30)

  # 1. Download dataset
  X, y = download_dataset()
  
  # 2. Explore dataset
  if explore:
    show_dataset_info(X, y)

  # 3. Clean categorical features
  X_cleaned = clean_categorical_features(X)

  # 4. Split dataset
  X_train, X_test, y_train, y_test = get_train_test_split(X_cleaned, y, test_size, random_state)

  # 5. Scale features
  if scale:
    X_train, X_test = scale_features(X_train, X_test)
  
  # 6. Visualize dataset
  if visualize:
    visualize_dataset(X, y)

  print("\n" + "=" * 30)
  print("Preprocessing Completed")
  print("=" * 30)
  
  
  return X_train, X_test, y_train, y_test


if __name__ == "__main__":  preprocess_data()

