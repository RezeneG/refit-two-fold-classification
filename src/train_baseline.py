"""
Train baseline models (Random Forest, XGBoost) on REFIT data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import argparse
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load preprocessed features and targets."""
    features = pd.read_csv("data/processed/features.csv", index_col=0)
    targets = pd.read_csv("data/processed/targets.csv", index_col=0).squeeze()
    households = pd.read_csv("data/processed/household_ids.csv", index_col=0).squeeze()
    
    return features, targets, households

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier."""
    print("\nTraining Random Forest...")
    
    # Calculate class weights
    class_weights = dict(zip(
        np.unique(y_train),
        1.0 / np.bincount(pd.factorize(y_train)[0])
    ))
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Validation
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"  Validation macro F1: {f1:.4f}")
    
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    print("\nTraining XGBoost...")
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    # Calculate class weights
    class_counts = np.bincount(y_train_enc)
    scale_pos_weight = class_counts[0] / class_counts[1:]  # Binary for detection
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=False
    )
    
    # Validation
    y_pred_enc = model.predict(X_val)
    y_pred = le.inverse_transform(y_pred_enc)
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"  Validation macro F1: {f1:.4f}")
    
    return model, le

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['random_forest', 'xgboost'], required=True)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training Baseline: {args.model}")
    print("=" * 60)
    
    # Load data
    features, targets, households = load_data()
    print(f"\nLoaded {len(features)} samples")
    
    # Time-based split (80/20 per household)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for h in households.unique():
        mask = households == h
        X_h = features[mask]
        y_h = targets[mask]
        
        split_idx = int(0.8 * len(X_h))
        X_train_list.append(X_h.iloc[:split_idx])
        X_val_list.append(X_h.iloc[split_idx:])
        y_train_list.append(y_h.iloc[:split_idx])
        y_val_list.append(y_h.iloc[split_idx:])
    
    X_train = pd.concat(X_train_list)
    X_val = pd.concat(X_val_list)
    y_train = pd.concat(y_train_list)
    y_val = pd.concat(y_val_list)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model
    if args.model == 'random_forest':
        model = train_random_forest(X_train, y_train, X_val, y_val)
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/random_forest.pkl")
        
    elif args.model == 'xgboost':
        model, label_encoder = train_xgboost(X_train, y_train, X_val, y_val)
        # Save model and encoder
        os.makedirs("models", exist_ok=True)
        model.save_model("models/xgboost.json")
        joblib.dump(label_encoder, "models/xgboost_label_encoder.pkl")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
