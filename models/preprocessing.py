"""
========================================================================
     STEP 2: Data Preprocessing & Balancing Pipeline
========================================================================

This module handles the full data preparation pipeline:

1. LOAD DATA     -- Read CSV, inspect shape & types
2. CLEAN DATA    -- Handle missing values, remove duplicates
3. SCALE FEATURES -- StandardScaler on Amount & Time columns
4. SPLIT DATA    -- Stratified train/test split (80/20)
5. BALANCE DATA  -- SMOTE oversampling on training set only

WHY SMOTE?
With only ~1.7% fraud, models will be biased toward "not fraud".
SMOTE creates synthetic fraud samples by interpolating between
existing fraud examples, giving the model balanced classes to learn.
We only apply SMOTE to training data -- test data stays unbalanced
to reflect real-world performance.

USAGE:
  from models.preprocessing import load_and_preprocess
  X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "data", "creditcard.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath=None):
    """
    STEP 2a: Load the credit card dataset from CSV.
    """
    if filepath is None:
        filepath = DATA_PATH
    
    print("=" * 60)
    print("[STEP 2a] Loading Dataset")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"   [OK] Loaded {filepath}")
    print(f"   [OK] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   [OK] Columns: {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    print(f"\n   Class Distribution:")
    print(f"      Class 0 (Legitimate): {(df['Class']==0).sum():,}")
    print(f"      Class 1 (Fraud)     : {(df['Class']==1).sum():,}")
    print(f"      Fraud Rate          : {df['Class'].mean()*100:.2f}%")
    
    missing = df.isnull().sum().sum()
    print(f"\n   Missing values: {missing}")
    
    return df


def clean_data(df):
    """
    STEP 2b: Clean the dataset.
    """
    print("\n" + "=" * 60)
    print("[STEP 2b] Cleaning Data")
    print("=" * 60)
    
    initial_rows = len(df)
    
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    print(f"   [OK] Duplicates removed: {dupes_removed}")
    
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
        print("   [OK] Missing values filled with column medians")
    else:
        print("   [OK] No missing values found")
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   [OK] Final shape after cleaning: {df.shape}")
    
    return df


def scale_features(df):
    """
    STEP 2c: Scale the Amount and Time features.
    
    WHY SCALE?
    V1-V28 are already PCA-transformed (zero-centered, unit variance).
    Amount ranges from $0 to $25,000+ and Time from 0 to 172,800 seconds.
    StandardScaler transforms these to mean=0, std=1 so they don't
    dominate the model.
    """
    print("\n" + "=" * 60)
    print("[STEP 2c] Scaling Features (Amount & Time)")
    print("=" * 60)
    
    scaler = StandardScaler()
    
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_Scaled"] = StandardScaler().fit_transform(df[["Time"]])
    df = df.drop(["Amount", "Time"], axis=1)
    
    print(f"   [OK] Amount scaled: mean={df['Amount_Scaled'].mean():.4f}, std={df['Amount_Scaled'].std():.4f}")
    print(f"   [OK] Time scaled  : mean={df['Time_Scaled'].mean():.4f}, std={df['Time_Scaled'].std():.4f}")
    print(f"   [OK] Original Amount & Time columns dropped")
    
    return df, scaler


def split_data(df):
    """
    STEP 2d: Split into train and test sets with stratification.
    
    WHY STRATIFIED SPLIT?
    With only ~1.7% fraud, a random split might put all fraud
    in training or all in test by chance. Stratification ensures
    both sets have the same fraud ratio.
    """
    print("\n" + "=" * 60)
    print("[STEP 2d] Train/Test Split (Stratified)")
    print("=" * 60)
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   [OK] Training set: {X_train.shape[0]:,} samples")
    print(f"      - Legitimate: {(y_train==0).sum():,}")
    print(f"      - Fraud     : {(y_train==1).sum():,}")
    print(f"   [OK] Test set    : {X_test.shape[0]:,} samples")
    print(f"      - Legitimate: {(y_test==0).sum():,}")
    print(f"      - Fraud     : {(y_test==1).sum():,}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    STEP 2e: Balance the training data using SMOTE.
    
    HOW SMOTE WORKS:
    1. Pick a fraud sample
    2. Find its k nearest fraud neighbors
    3. Create a new synthetic sample between them
    4. Repeat until classes are balanced
    
    IMPORTANT: We ONLY balance training data, not test data!
    """
    print("\n" + "=" * 60)
    print("[STEP 2e] Balancing with SMOTE")
    print("=" * 60)
    
    print(f"   Before SMOTE:")
    print(f"      Class 0: {(y_train==0).sum():,}")
    print(f"      Class 1: {(y_train==1).sum():,}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n   After SMOTE:")
    print(f"      Class 0: {(y_resampled==0).sum():,}")
    print(f"      Class 1: {(y_resampled==1).sum():,}")
    print(f"   [OK] Training data is now balanced!")
    
    return X_resampled, y_resampled


def load_and_preprocess(filepath=None, use_smote=True):
    """
    MASTER FUNCTION: Run the complete preprocessing pipeline.
    
    Chains all steps:
    load_data -> clean_data -> scale_features -> split_data -> apply_smote
    """
    print("\n" + "==" * 30)
    print("  COMPLETE PREPROCESSING PIPELINE")
    print("==" * 30)
    
    df = load_data(filepath)
    df = clean_data(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    feature_names = list(X_train.columns)
    
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("[DONE] PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"   * Features: {len(feature_names)}")
    print(f"   * Training samples: {len(X_train):,}")
    print(f"   * Test samples: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess()
    print(f"\n[INFO] Feature list: {features}")
