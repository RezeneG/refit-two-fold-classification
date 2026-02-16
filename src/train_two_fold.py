"""
Train two-fold XGBoost model.
Stage 1: Activity detection (binary: active vs inactive)
Stage 2: Appliance identification (multi-class on active periods only)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load preprocessed features and targets."""
    features = pd.read_csv("data/processed/features.csv", index_col=0)
    targets = pd.read_csv("data/processed/targets.csv", index_col=0).squeeze()
    households = pd.read_csv("data/processed/household_ids.csv", index_col=0).squeeze()
    
    return features, targets, households

def create_stage1_targets(targets):
    """Convert to binary: active (any appliance) vs inactive."""
    return (targets != 'none').astype(int)

def train_stage1(X_train, y_train_bin, X_val, y_val_bin):
    """Train activity detection model."""
    print("\n" + "=" * 40)
    print("Stage 1: Activity Detection")
    print("=" * 40)
    
    # Calculate scale_pos_weight for imbalance
    neg_count = (y_train_bin == 0).sum()
    pos_count = (y_train_bin == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class ratio (neg:pos) = {neg_count}:{pos_count} = {scale_pos_weight:.2f}:1")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)],
        verbose=False
    )
    
    # Evaluate
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_val_bin, y_pred_prob)
    auprc = auc(recall, precision)
    f1 = f1_score(y_val_bin, y_pred)
    
    print(f"Validation AUPRC: {auprc:.4f}")
    print(f"Validation F1: {f1:.4f}")
    
    return model

def train_stage2(X_train_active, y_train_active, X_val_active, y_val_active):
    """Train appliance identification model on active periods only."""
    print("\n" + "=" * 40)
    print("Stage 2: Appliance Identification")
    print("=" * 40)
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_active)
    y_val_enc = le.transform(y_val_active)
    
    print(f"Number of appliance classes: {len(le.classes_)}")
    print(f"Training samples (active only): {len(X_train_active)}")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train_active, y_train_enc,
        eval_set=[(X_val_active, y_val_enc)],
        verbose=False
    )
    
    # Evaluate
    y_pred_enc = model.predict(X_val_active)
    y_pred = le.inverse_transform(y_pred_enc)
    f1 = f1_score(y_val_active, y_pred, average='macro')
    
    print(f"Validation macro F1: {f1:.4f}")
    
    return model, le

def main():
    print("=" * 60)
    print("Training Two-Fold XGBoost")
    print("=" * 60)
    
    # Load data
    features, targets, households = load_data()
    print(f"\nLoaded {len(features)} samples")
    
    # Time-based split
    X_train_list, X_val_list = [], []
    y_train_list, y_val_list = [], []
    
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
    
    # Stage 1: Activity detection
    y_train_bin = create_stage1_targets(y_train)
    y_val_bin = create_stage1_targets(y_val)
    
    stage1_model = train_stage1(X_train, y_train_bin, X_val, y_val_bin)
    
    # Stage 2: Appliance identification (active periods only)
    train_active_mask = y_train != 'none'
    val_active_mask = y_val != 'none'
    
    X_train_active = X_train[train_active_mask]
    y_train_active = y_train[train_active_mask]
    X_val_active = X_val[val_active_mask]
    y_val_active = y_val[val_active_mask]
    
    stage2_model, label_encoder = train_stage2(
        X_train_active, y_train_active,
        X_val_active, y_val_active
    )
    
    # Save models
    os.makedirs("models", exist_ok=True)
    stage1_model.save_model("models/stage1_xgboost.json")
    stage2_model.save_model("models/stage2_xgboost.json")
    joblib.dump(label_encoder, "models/stage2_label_encoder.pkl")
    
    print("\n" + "=" * 60)
    print("✅ Two-fold training complete!")
    print("Models saved to: models/")
    print("=" * 60)

if __name__ == "__main__":
    main()
