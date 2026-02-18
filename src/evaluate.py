"""
Evaluation script with progress bar and NumPy conversion for two‑fold model.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
from tqdm import tqdm  # <-- added

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    features = pd.read_csv("data/processed/features.csv", index_col=0)
    targets = pd.read_csv("data/processed/targets.csv", index_col=0).squeeze()
    households = pd.read_csv("data/processed/household_ids.csv", index_col=0).squeeze()

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
    models = {}
    try:
        models['rf'] = joblib.load("models/random_forest.pkl")
        models['rf_imputer'] = joblib.load("models/random_forest_imputer.pkl")
        print("✓ Loaded Random Forest + imputer")
    except:
        models['rf'] = None

    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/xgboost.json")
        models['xgb'] = xgb_model
        models['xgb_le'] = joblib.load("models/xgboost_label_encoder.pkl")
        models['xgb_imputer'] = joblib.load("models/xgboost_imputer.pkl")
        print("✓ Loaded XGBoost baseline + imputer")
    except:
        models['xgb'] = None

    try:
        models['stage1'] = xgb.XGBClassifier()
        models['stage1'].load_model("models/stage1_xgboost.json")
        models['stage2'] = xgb.XGBClassifier()
        models['stage2'].load_model("models/stage2_xgboost.json")
        models['stage2_le'] = joblib.load("models/stage2_label_encoder.pkl")
        print("✓ Loaded two-fold models")
    except:
        models['stage1'] = None

    return models

def filter_test_by_known_labels(X, y, known_classes):
    mask = y.isin(known_classes)
    X_f = X[mask]
    y_f = y[mask]
    print(f"  Filtered from {len(y)} to {len(y_f)} samples (dropped {len(y)-len(y_f)})")
    return X_f, y_f

def evaluate_baseline(model, imputer, label_encoder, X_test, y_test, name):
    X_test_imp = imputer.transform(X_test)
    if label_encoder is None:
        y_pred = model.predict(X_test_imp)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            'name': name,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred,
            'true': y_test,
            'report': report
        }
    else:
        known = set(label_encoder.classes_)
        X_f, y_f = filter_test_by_known_labels(X_test_imp, y_test, known)
        if len(X_f) == 0:
            print(f"  WARNING: No test samples with known labels for {name}")
            return None
        y_f_enc = label_encoder.transform(y_f)
        y_pred_enc = model.predict(X_f)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        f1_macro = f1_score(y_f, y_pred, average='macro')
        f1_weighted = f1_score(y_f, y_pred, average='weighted')
        report = classification_report(y_f, y_pred, output_dict=True)
        return {
            'name': name,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': y_pred,
            'true': y_f,
            'report': report
        }

def evaluate_two_fold(stage1, stage2, stage2_le, X_test, y_test):
    y_test_bin = (y_test != 'none').astype(int)
    stage1_pred_prob = stage1.predict_proba(X_test)[:, 1]
    stage1_pred = (stage1_pred_prob > 0.5).astype(int)

    final_pred = []
    valid_indices = []
    known_active = set(stage2_le.classes_)

    # Loop with progress bar
    for i in tqdm(range(len(X_test)), desc="Two‑fold inference"):
        if stage1_pred[i] == 0:
            final_pred.append('none')
            # Keep if true label is 'none' (always known)
            if y_test.iloc[i] == 'none':
                valid_indices.append(i)
        else:
            # Convert single row to numpy array (2D) to avoid XGBoost pandas issues
            X_single = X_test.iloc[[i]].values
            stage2_pred_enc = stage2.predict(X_single)[0]
            stage2_pred = stage2_le.inverse_transform([stage2_pred_enc])[0]
            final_pred.append(stage2_pred)
            # Keep if true label is known (either 'none' or an appliance)
            if y_test.iloc[i] == 'none' or y_test.iloc[i] in known_active:
                valid_indices.append(i)

    final_pred = np.array(final_pred)
    y_valid = y_test.iloc[valid_indices]
    pred_valid = final_pred[valid_indices]

    print(f"  Two‑fold: kept {len(y_valid)} / {len(y_test)} samples")

    if len(y_valid) == 0:
        return None

    f1_macro = f1_score(y_valid, pred_valid, average='macro')
    f1_weighted = f1_score(y_valid, pred_valid, average='weighted')

    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_test_bin, stage1_pred_prob)
    auprc = auc(recall, precision)

    active_mask_valid = (y_valid != 'none')
    if active_mask_valid.sum() > 0:
        f1_stage2 = f1_score(
            y_valid[active_mask_valid],
            pred_valid[active_mask_valid],
            average='macro'
        )
    else:
        f1_stage2 = np.nan

    report = classification_report(y_valid, pred_valid, output_dict=True)

    return {
        'name': 'Two-Fold XGBoost',
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auprc_stage1': auprc,
        'f1_stage2': f1_stage2,
        'predictions': pred_valid,
        'true': y_valid,
        'report': report
    }

def mcnemar_test(y_true, pred1, pred2):
    pred1_correct = (pred1 == y_true)
    pred2_correct = (pred2 == y_true)
    a = np.sum(pred1_correct & pred2_correct)
    b = np.sum(pred1_correct & ~pred2_correct)
    c = np.sum(~pred1_correct & pred2_correct)
    d = np.sum(~pred1_correct & ~pred2_correct)
    chi2 = (abs(b - c) - 1)**2 / (b + c + 1e-8)
    p_value = 1 - chi2_contingency([[a, b], [c, d]], correction=False)[1]
    return chi2, p_value

def create_results_table(results_list):
    table_data = []
    for r in results_list:
        if r is not None:
            table_data.append({
                'Model': r['name'],
                'Macro F1': f"{r['f1_macro']:.3f}",
                'Weighted F1': f"{r['f1_weighted']:.3f}"
            })
    return pd.DataFrame(table_data)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    top_classes = y_true.value_counts().head(8).index.tolist()
    mask = y_true.isin(top_classes)
    y_true_subset = y_true[mask]
    y_pred_subset = y_pred[mask]
    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=top_classes)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
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

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    X_test, y_test = load_data()
    print(f"\nTest samples: {len(X_test)}")

    models = load_models()

    results = []

    # Random Forest
    if models['rf'] is not None:
        print("\nEvaluating Random Forest...")
        res = evaluate_baseline(
            models['rf'], models['rf_imputer'], None,
            X_test, y_test, "Random Forest"
        )
        results.append(res)

    # XGBoost baseline
    if models['xgb'] is not None:
        print("\nEvaluating XGBoost baseline...")
        res = evaluate_baseline(
            models['xgb'], models['xgb_imputer'], models['xgb_le'],
            X_test, y_test, "XGBoost (end-to-end)"
        )
        if res is not None:
            results.append(res)

    # Two-fold
    if models['stage1'] is not None:
        print("\nEvaluating Two-Fold...")
        res = evaluate_two_fold(
            models['stage1'], models['stage2'], models['stage2_le'],
            X_test, y_test
        )
        if res is not None:
            results.append(res)

    if not results:
        print("No valid results obtained.")
        return

    # Results table
    results_df = create_results_table(results)
    print("\n" + "=" * 40)
    print("Results Summary")
    print("=" * 40)
    print(results_df.to_string(index=False))
    results_df.to_csv("results/tables/main_results.csv", index=False)

    # Per-class metrics for best model
    best_idx = np.argmax([r['f1_macro'] for r in results if r is not None])
    best_result = results[best_idx]

    print("\n" + "=" * 40)
    print(f"Per-Class Performance ({best_result['name']})")
    print("=" * 40)

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
        best_result['true'],
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