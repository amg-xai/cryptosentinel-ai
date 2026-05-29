"""
Run the backtest and print a readable report.
Usage: python -m src.backtesting.run_backtest
"""
from src.backtesting.backtester import Backtester


def main():
    bt = Backtester()
    print("Loading GNN model + Elliptic test set...")
    if not bt.load():
        print("FAILED to load — are models trained and data/processed/ present?")
        return

    print("Running backtest...\n")
    results = bt.run(production_threshold=0.5)

    print(f"Test samples:       {results['test_samples']}")
    print(f"Illicit in test:    {results['illicit_in_test']}")
    print()

    prod = results["production_metrics"]
    c = prod["confusion"]
    print(f"=== At production threshold {prod['threshold']} ===")
    print(f"  Precision: {prod['precision']}")
    print(f"  Recall:    {prod['recall']}")
    print(f"  F1:        {prod['f1']}")
    print(f"  Accuracy:  {prod['accuracy']}")
    print(f"  Confusion: TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}")
    print()

    best = results["best_threshold"]
    print(f"=== Best operating point (max F1) ===")
    print(f"  Threshold: {best['threshold']}")
    print(f"  Precision: {best['precision']}")
    print(f"  Recall:    {best['recall']}")
    print(f"  F1:        {best['f1']}")
    print()
    print("Results saved to data/models/backtest_results.json")


if __name__ == "__main__":
    main()
