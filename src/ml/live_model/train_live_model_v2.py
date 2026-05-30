"""
Train the live-native classifier on the v2 (artifact-free) dataset.

Built-in safeguard: before reporting performance, print a per-feature
pos-vs-neg mean comparison so any source artifact (a feature that's
suspiciously separated) is visible. We got burned once by a placeholder
gas-price leak that gave a fake F1=1.0; this makes that failure mode loud.
"""
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib

from config.logging_config import get_logger
from src.ml.tabular.feature_engineer import TransactionFeatures

logger = get_logger(__name__)
PROCESSED = Path("data/processed")
MODELS = Path("data/models")
NAMES = TransactionFeatures.FEATURE_NAMES


def _leakage_report(X, y):
    """Print per-feature pos/neg means + standardized separation."""
    pos, neg = X[y == 1], X[y == 0]
    print("\n=== Leakage / separation check (per feature) ===")
    print(f"{'feature':<22}{'pos_mean':>12}{'neg_mean':>12}{'|sep|':>8}")
    seps = []
    for i, n in enumerate(NAMES):
        pm, nm = pos[:, i].mean(), neg[:, i].mean()
        sd = X[:, i].std() + 1e-9
        sep = abs(pm - nm) / sd
        seps.append((n, sep))
        flag = "  <-- suspicious" if sep > 2.5 else ""
        print(f"{n:<22}{pm:>12.3f}{nm:>12.3f}{sep:>8.2f}{flag}")
    # A single feature with huge separation is the classic leak signature
    worst = max(seps, key=lambda s: s[1])
    if worst[1] > 2.5:
        print(f"\nWARNING: '{worst[0]}' separates classes very strongly "
              f"(|sep|={worst[1]:.1f}). Verify it's real signal, not an artifact.")


def train():
    X = np.load(PROCESSED / "live_X_v2.npy")
    y = np.load(PROCESSED / "live_y_v2.npy")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    _leakage_report(X, y)

    # Drop dead features (always-zero graph slots in this offline build):
    # in_degree(13), out_degree(14), cluster_risk(15) are set elsewhere live.
    keep = [i for i in range(X.shape[1]) if i not in (13, 14, 15)]
    Xk = X[:, keep]
    kept_names = [NAMES[i] for i in keep]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xk)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)

    # Honest estimate first: 5-fold CV
    cv = cross_val_score(clf, Xs, y, cv=5, scoring="f1")
    print(f"\n5-fold CV F1: {np.round(cv, 3)}  mean={cv.mean():.3f} std={cv.std():.3f}")

    # Fit final on a train/test split for reported metrics
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.25, stratify=y, random_state=42)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "cv_f1_mean": round(float(cv.mean()), 4),
        "cv_f1_std": round(float(cv.std()), 4),
        "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "features_used": kept_names,
    }
    coefs = clf.coef_[0]
    metrics["top_features"] = sorted(
        [{"feature": kept_names[i], "coef": round(float(coefs[i]), 4)}
         for i in range(len(coefs))],
        key=lambda d: abs(d["coef"]), reverse=True)[:8]

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS / "live_model.joblib")
    joblib.dump(scaler, MODELS / "live_scaler.joblib")
    json.dump({"keep_indices": keep}, open(MODELS / "live_model_features.json", "w"))
    json.dump(metrics, open(MODELS / "live_model_metrics.json", "w"), indent=2)
    return metrics


def main():
    m = train()
    print("\n=== Live Model (v2, artifact-checked) ===")
    print(f"  CV F1:     {m['cv_f1_mean']} ± {m['cv_f1_std']}")
    print(f"  Test F1:   {m['f1']}  P={m['precision']} R={m['recall']} AUC={m['roc_auc']}")
    print(f"  pos/neg:   {m['n_pos']}/{m['n_neg']}")
    print("  Top features:")
    for f in m["top_features"][:6]:
        print(f"    {f['feature']}: {f['coef']}")


if __name__ == "__main__":
    main()
