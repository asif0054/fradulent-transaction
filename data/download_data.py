"""
========================================================================
          STEP 1: Synthetic Credit Card Dataset Generator
========================================================================

WHY SYNTHETIC DATA?
The real Kaggle "Credit Card Fraud Detection" dataset requires API
credentials. This script generates a realistic synthetic dataset
that mirrors its structure exactly.

COLUMNS:
  Time    -- Seconds elapsed from first transaction (0-172800)
  V1-V28  -- 28 PCA-transformed features (anonymized)
  Amount  -- Transaction amount in USD
  Class   -- 0 = Legitimate, 1 = Fraud

The dataset has ~1.7% fraud rate (realistic class imbalance).

USAGE:
  python data/download_data.py
========================================================================
"""

import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
TOTAL_TRANSACTIONS = 10000       # Total number of transactions to generate
FRAUD_RATIO = 0.017              # ~1.7% fraud (matches real-world Kaggle data)
RANDOM_SEED = 42                 # For reproducibility
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "creditcard.csv")


def generate_dataset():
    """
    Generate a synthetic credit card fraud dataset.
    
    STEP-BY-STEP PROCESS:
    1. Calculate how many fraud vs. legitimate transactions
    2. Generate different statistical distributions for each class
       - Fraud: shifted means and larger variance (anomalous behavior)
       - Legitimate: centered around zero with normal variance
    3. Generate realistic Time and Amount columns
    4. Combine, shuffle, and save to CSV
    """
    
    np.random.seed(RANDOM_SEED)
    
    # -- Step 1: Calculate class sizes --
    n_fraud = int(TOTAL_TRANSACTIONS * FRAUD_RATIO)
    n_legit = TOTAL_TRANSACTIONS - n_fraud
    
    print("[INFO] Generating synthetic dataset...")
    print(f"   * Total transactions : {TOTAL_TRANSACTIONS:,}")
    print(f"   * Legitimate (Class 0): {n_legit:,} ({100*(1-FRAUD_RATIO):.1f}%)")
    print(f"   * Fraudulent (Class 1): {n_fraud:,} ({100*FRAUD_RATIO:.1f}%)")
    
    # -- Step 2: Generate PCA features V1-V28 --
    # Legitimate transactions: normal distribution centered near 0
    legit_features = np.random.randn(n_legit, 28) * 1.0
    
    # Fraudulent transactions: shifted means and higher variance
    fraud_means = np.random.uniform(-3, 3, size=28)
    fraud_stds = np.random.uniform(1.5, 3.0, size=28)
    fraud_features = np.random.randn(n_fraud, 28) * fraud_stds + fraud_means
    
    # -- Step 3: Generate Time column --
    legit_time = np.random.uniform(0, 172800, n_legit)
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    
    # -- Step 4: Generate Amount column --
    legit_amount = np.abs(np.random.exponential(scale=88, size=n_legit))
    
    fraud_small = np.random.uniform(0.5, 5, size=n_fraud // 2)
    fraud_large = np.random.uniform(200, 2500, size=n_fraud - n_fraud // 2)
    fraud_amount = np.concatenate([fraud_small, fraud_large])
    np.random.shuffle(fraud_amount)
    
    # -- Step 5: Assemble DataFrames --
    feature_columns = [f"V{i}" for i in range(1, 29)]
    
    df_legit = pd.DataFrame(legit_features, columns=feature_columns)
    df_legit["Time"] = legit_time
    df_legit["Amount"] = legit_amount
    df_legit["Class"] = 0
    
    df_fraud = pd.DataFrame(fraud_features, columns=feature_columns)
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = fraud_amount
    df_fraud["Class"] = 1
    
    # -- Step 6: Combine, shuffle, and reorder columns --
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    column_order = ["Time"] + feature_columns + ["Amount", "Class"]
    df = df[column_order]
    
    # -- Step 7: Save to CSV --
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n[OK] Dataset saved to: {OUTPUT_FILE}")
    print(f"   * Shape: {df.shape}")
    print(f"   * File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print(f"\n[INFO] Column preview:")
    print(f"   {list(df.columns)}")
    print(f"\n[INFO] Class distribution:")
    print(df["Class"].value_counts().to_string())
    
    return df


# -----------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset()
    print("\n[INFO] Sample rows:")
    print(df.head(3).to_string())
