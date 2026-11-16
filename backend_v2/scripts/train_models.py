"""CLI script to train baseline and candidate ML models, calibrate probabilities, and register them."""

import argparse
from pathlib import Path

from app.ml.loader import load_dataset_manifest, load_split
from app.ml.models import CalibrationMethod, ModelType, train_model
from app.ml.registry import ModelRegistry
from app.ml.validation import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register ML models for DeltaBot 2.0")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to versioned dataset directory containing train.jsonl and manifest.json",
    )
    parser.add_argument(
        "--output-registry",
        type=Path,
        default=Path("model-registry"),
        help="Path to model registry directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "lightgbm", "xgboost"],
        choices=["logistic_regression", "random_forest", "lightgbm", "xgboost"],
        help="Model types to train",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Probability calibration method",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.40,
        help="Confidence threshold for LONG/SHORT signals",
    )
    parser.add_argument(
        "--target-asset",
        type=str,
        default=None,
        choices=["BTC", "ETH"],
        help="Optional asset filter",
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.dataset_dir)
    dataset_version = manifest["dataset_version"]
    feature_schema = manifest.get("feature_schema_version", "candle-market-features-v1")

    print("=" * 70)
    print(f"DELTABOT 2.0 ML TRAINING PIPELINE")
    print(f"Dataset: {args.dataset_dir} ({dataset_version})")
    print(f"Schema:  {feature_schema}")
    print(f"Models:  {args.models}")
    print("=" * 70)

    print("\nLoading training and validation splits...")
    train_split = load_split(args.dataset_dir, "train", target_asset=args.target_asset)
    val_split = load_split(args.dataset_dir, "validation", target_asset=args.target_asset)

    print(f"Train samples: {len(train_split.y)}, Features: {len(train_split.feature_names)}")
    print(f"Val samples:   {len(val_split.y)}")

    registry = ModelRegistry(args.output_registry)
    results = []

    for model_type in args.models:
        print(f"\n--- Training {model_type} ---")
        artifact = train_model(
            model_type=model_type,  # type: ignore
            X_train=train_split.X,
            y_train=train_split.y,
            X_val=val_split.X,
            y_val=val_split.y,
            calibration=args.calibration,  # type: ignore
            feature_names=train_split.feature_names,
        )

        eval_result = evaluate_model(
            artifact,
            val_split.X,
            val_split.y,
            val_split.net_returns_bps,
            val_split.sample_metadata,
            confidence_threshold=args.confidence_threshold,
        )

        model_id = registry.register_model(
            artifact,
            dataset_version=dataset_version,
            feature_schema_version=feature_schema,
            evaluation_metrics=eval_result,
            description=f"{model_type} trained on {dataset_version} with {args.calibration} calibration",
        )

        clf = eval_result["classification"]
        trd = eval_result["trading"]
        print(f"Model ID:        {model_id}")
        print(f"Accuracy:        {clf['accuracy']:.4f} | F1 Macro: {clf['f1_macro']:.4f} | MCC: {clf['mcc']:.4f}")
        print(f"Brier Score:     {clf['brier_score']:.4f}")
        print(f"Trades:          {trd['total_trades']} | Win Rate: {trd['win_rate']*100:.1f}%")
        print(f"Expectancy:      {trd['net_expectancy_bps']:+.2f} bps/trade")
        print(f"Profit Factor:   {trd['profit_factor']:.2f}")
        print(f"Max Drawdown:    {trd['max_drawdown_bps']:.1f} bps")
        print(f"Net P&L:         {trd['net_pnl_bps']:+.1f} bps")
        print(f"Acceptance Gate: {'PASSED' if eval_result['passes_acceptance'] else 'FAILED'}")

        results.append(
            {
                "model_id": model_id,
                "model_type": model_type,
                "accuracy": clf["accuracy"],
                "f1": clf["f1_macro"],
                "expectancy": trd["net_expectancy_bps"],
                "profit_factor": trd["profit_factor"],
                "passed": eval_result["passes_acceptance"],
            }
        )

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION SUMMARY")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Expectancy':<14} {'Profit Factor':<14} {'Accepted'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model_type']:<20} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {r['expectancy']:<+14.2f} {r['profit_factor']:<14.2f} {'YES' if r['passed'] else 'NO'}"
        )
    print("=" * 70)
    print("Models registered in 'model-registry/'. None are automatically promoted.")


if __name__ == "__main__":
    main()
