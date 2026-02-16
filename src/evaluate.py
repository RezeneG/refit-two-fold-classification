"""
Evaluation script to generate results tables and figures.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                           confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """Load test data (last 20% of each household)."""
    features = pd.read_csv("data/processed/features.csv", index_col=0)
    targets = pd.read_csv("data/processed/targets.csv", index_col=0).squeeze()
    households = pd.read_csv("data/processed/household_ids.csv", index_col=0).squeeze()
    
    # Get test set (last 20% per household)
    X_test_list = []
    y_test_list = []
    
    for h in households.unique():
        mask = households == h
        X_h = features[mask]
        y_h = targets[mask]
        
        split_idx = int(0.8 * len(X_h))
        X_test_list.append(X_h.iloc[split_idx:])
        y_test_list.append(y_h.iloc[split_idx:])
    
    X_test = pd.concat(X_test_list)
    y_test = pd.concat(y_test_list)
    
    return X_test, y_test

def load_models():
    """Load all trained models."""
    models = {}
    
    # Random Forest
    try:
        models['rf'] = joblib.load("models/random_forest.pkl")
        print("✓ Loaded Random Forest")
    except:
        print("✗ Random Forest not found")
        models['rf'] = None
    
    # XGBoost baseline
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/xgboost.json")
        models['xgb'] = xgb_model
        models['xgb_le'] = joblib.load("models/xgboost_label_encoder.pkl")
        print("✓ Loaded XGBoost baseline")
    except:
        print("✗ XGBoost baseline not found")
        models['xgb'] = None
    
    # Two-fold models
    try:
        models['stage1'] = xgb.XGBClassifier()
        models['stage1'].load_model("models/stage1_xgboost.json")
        models['stage2'] = xgb.XGBClassifier()
        models['stage2'].load_model("models/stage2_xgboost.json")
        models['stage2_le'] = joblib.load("models/stage2_label_encoder.pkl")
        print("✓ Loaded two-fold models")
    except:
        print("✗ Two-fold models not found")
        models['stage1'] = None
    
    return models

def evaluate_baseline(model, label_encoder, X_test, y_test, name):
    """Evaluate baseline model."""
    # Encode test labels
    y_test_enc = label_encoder.transform(y_test)
    
    # Predict
    y_pred_enc = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    
    # Metrics
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'name': name,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': y_pred,
        'true': y_test,
        'report': report
    }

def evaluate_two_fold(stage1, stage2, stage2_le, X_test, y_test):
    """Evaluate two-fold model."""
    
    # Stage 1: Activity detection
    y_test_bin = (y_test != 'none').astype(int)
    stage1_pred_prob = stage1.predict_proba(X_test)[:, 1]
    stage1_pred = (stage1_pred_prob > 0.5).astype(int)
    
    # Stage 2: Conditional classification
    final_pred = []
    for i, (pred_active, probs) in enumerate(zip(stage1_pred, stage1_pred_prob)):
        if pred_active == 0:
            final_pred.append('none')
        else:
            # Pass to stage 2
            X_single = X_test.iloc[[i]]
            stage2_pred_enc = stage2.predict(X_single)[0]
            stage2_pred = stage2_le.inverse_transform([stage2_pred_enc])[0]
            final_pred.append(stage2_pred)
    
    final_pred = np.array(final_pred)
    
    # Metrics
    f1_macro = f1_score(y_test, final_pred, average='macro')
    f1_weighted = f1_score(y_test, final_pred, average='weighted')
    
    # Stage 1 metrics
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_test_bin, stage1_pred_prob)
    auprc = auc(recall, precision)
    
    # Stage 2 metrics (on truly active only)
    active_mask = y_test != 'none'
    if active_mask.sum() > 0:
        f1_stage2 = f1_score(
            y_test[active_mask], 
            final_pred[active_mask], 
            average='macro'
        )
    else:
        f1_stage2 = np.nan
    
    # Per-class metrics
    report = classification_report(y_test, final_pred, output_dict=True)
    
    return {
        'name': 'Two-Fold XGBoost',
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auprc_stage1': auprc,
        'f1_stage2': f1_stage2,
        'predictions': final_pred,
        'true': y_test,
        'report': report
    }

def mcnemar_test(y_true, pred1, pred2):
    """McNemar's test for paired comparison."""
    # Create contingency table
    pred1_correct = (pred1 == y_true)
    pred2_correct = (pred2 == y_true)
    
    a = np.sum(pred1_correct & pred2_correct)  # both correct
    b = np.sum(pred1_correct & ~pred2_correct) # only pred1 correct
    c = np.sum(~pred1_correct & pred2_correct) # only pred2 correct
    d = np.sum(~pred1_correct & ~pred2_correct) # both wrong
    
    # McNemar's test (with continuity correction)
    chi2 = (abs(b - c) - 1)**2 / (b + c + 1e-8)
    p_value = 1 - chi2_contingency([[a, b], [c, d]], correction=False)[1]
    
    return chi2, p_value

def create_results_table(results_list):
    """Create main results table."""
    table_data = []
    for r in results_list:
        table_data.append({
            'Model': r['name'],
            'Macro F1': f"{r['f1_macro']:.3f}",
            'Weighted F1': f"{r['f1_weighted']:.3f}"
        })
    
    df = pd.DataFrame(table_data)
    return df

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot confusion matrix for top classes."""
    # Get top 8 classes by frequency
    top_classes = y_true.value_counts().head(8).index.tolist()
    
    # Filter to top classes
    mask = y_true.isin(top_classes)
    y_true_subset = y_true[mask]
    y_pred_subset = y_pred[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=top_classes)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=top_classes, 
                yticklabels=top_classes, cmap='Blues')
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Load data and models
    X_test, y_test = load_data()
    print(f"\nTest samples: {len(X_test)}")
    
    models = load_models()
    
    # Evaluate all models
    results = []
    
    if models['rf'] is not None:
        # For RF, we need to fit label encoder on test classes
        from sklearn.preprocessing import LabelEncoder
        temp_le = LabelEncoder()
        temp_le.fit(y_test)
        # Note: This is a simplification; in practice you'd save encoder from training
        results.append(evaluate_baseline(
            models['rf'], temp_le, X_test, y_test, "Random Forest"
        ))
    
    if models['xgb'] is not None:
        results.append(evaluate_baseline(
            models['xgb'], models['xgb_le'], X_test, y_test, "XGBoost (end-to-end)"
        ))
    
    if models['stage1'] is not None:
        results.append(evaluate_two_fold(
            models['stage1'], models['stage2'], models['stage2_le'], X_test, y_test
        ))
    
    # Create results table
    results_df = create_results_table(results)
    print("\n" + "=" * 40)
    print("Results Summary")
    print("=" * 40)
    print(results_df.to_string(index=False))
    
    # Save table
    results_df.to_csv("results/tables/main_results.csv", index=False)
    
    # McNemar's test
    if len(results) >= 2:
        print("\n" + "=" * 40)
        print("McNemar's Test")
        print("=" * 40)
        
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                chi2, p = mcnemar_test(
                    y_test, 
                    results[i]['predictions'], 
                    results[j]['predictions']
                )
                print(f"{results[i]['name']} vs {results[j]['name']}:")
                print(f"  χ² = {chi2:.2f}, p = {p:.4f}")
                if p < 0.05:
                    print(f"  → Significant difference (p < 0.05)")
                else:
                    print(f"  → No significant difference")
    
    # Per-class metrics for best model
    best_idx = np.argmax([r['f1_macro'] for r in results])
    best_result = results[best_idx]
    
    print("\n" + "=" * 40)
    print(f"Per-Class Performance ({best_result['name']})")
    print("=" * 40)
    
    # Extract per-class metrics from report
    per_class = []
    for class_name, metrics in best_result['report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(metrics, dict):
                per_class.append({
                    'Appliance': class_name,
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-score': f"{metrics['f1-score']:.3f}"
                })
    
    per_class_df = pd.DataFrame(per_class)
    print(per_class_df.to_string(index=False))
    per_class_df.to_csv("results/tables/per_class_performance.csv", index=False)
    
    # Confusion matrix for best model
    plot_confusion_matrix(
        y_test, 
        best_result['predictions'],
        f"Confusion Matrix - {best_result['name']}",
        "results/figures/confusion_matrix.png"
    )
    print("\n✓ Confusion matrix saved to results/figures/")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("Results saved to results/tables/ and results/figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()
