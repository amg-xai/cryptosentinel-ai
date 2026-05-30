"""
Train a live-native classifier on the 16 live-feature space.

WHY this exists:
  The Elliptic-trained models can't score live EVM transactions (feature
  space mismatch -> collapsed scores). This model is trained DIRECTLY on
  the 16 features the live pipeline computes, using weak labels from public
  threat-intel lists (OFAC sanctions + MEW darklist) for positives and
  random recent mainnet traffic for negatives.

  It's weak supervision: labels are incomplete and negatives are assumed-
  legit. But the feature space matches production exactly, so unlike the
  padded Elliptic models, this one actually discriminates on live data.

Pipeline: StandardScaler -> LogisticRegression (class_weight balanced).
LogReg chosen for calibrated probabilities + interpretable coefficients
(you can explain WHICH features drive a flag), and it's robust on small
weakly-labeled sets where a heavy model would overfit.
"""
import json
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
import joblib

from config.logging_config import get_logger
from src.ml.tabular.feature_engineer import TransactionFeatures

logger = get_logger(__name__)

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


def train() -> dict:
    X = np.load(PROCESSED / "live_X.npy")
    y = np.load(PROCESSED / "live_y.npy")

    # Replace any inf/nan from feature engineering
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
    )
    clf.fit(X_tr_s, y_tr)

    proba = clf.predict_proba(X_te_s)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "f1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_pos_total": int(y.sum()),
        "n_neg_total": int((1 - y).sum()),
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS / "live_model.joblib")
    joblib.dump(scaler, MODELS / "live_scaler.joblib")

    # Save feature importances (logreg coefficients) for explainability
    feat_names = TransactionFeatures.FEATURE_NAMES
    coefs = clf.coef_[0]
    importance = sorted(
        [{"feature": feat_names[i] if i < len(feat_names) else f"f{i}",
          "coef": round(float(coefs[i]), 4)}
         for i in range(len(coefs))],
        key=lambda d: abs(d["coef"]), reverse=True,
    )
    metrics["top_features"] = importance[:8]

    with open(MODELS / "live_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("live_model_trained", **{k: v for k, v in metrics.items()
                                         if k != "top_features"})
    return metrics


def main():
    m = train()
    print("=== Live Model (16-feature space) ===")
    print(f"  F1:        {m['f1']}")
    print(f"  Precision: {m['precision']}")
    print(f"  Recall:    {m['recall']}")
    print(f"  ROC-AUC:   {m['roc_auc']}")
    print(f"  Train/Test: {m['n_train']}/{m['n_test']}  "
          f"(pos {m['n_pos_total']} / neg {m['n_neg_total']})")
    print("  Top features by |coef|:")
    for f in m["top_features"][:6]:
        print(f"    {f['feature']}: {f['coef']}")


if __name__ == "__main__":
    main()
