"""
Generate the drift-detection baseline from the GNN's scores on the
held-out Elliptic test set. Run once after training.
Usage: python -m src.monitoring.generate_drift_baseline
"""
import numpy as np
from src.backtesting.backtester import Backtester
from src.monitoring.drift_detector import save_baseline


def main():
    bt = Backtester()
    if not bt.load():
        print("Could not load model/data — train the GNN first.")
        return
    scores, _ = bt._gnn_scores_on_test()
    save_baseline(np.asarray(scores))
    print(f"Baseline saved: {len(scores)} scores, "
          f"mean={scores.mean():.4f}, std={scores.std():.4f}")


if __name__ == "__main__":
    main()
