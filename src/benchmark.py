import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure repository root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml

from src.data_loader import (
    extract_feature_subsets,
    load_dataset,
    prepare_train_test_split,
)
from src.metrics import (
    calculate_classification_metrics,
    evaluate_guardrails,
    measure_inference_performance,
    measure_memory_overhead,
)
from src.models import build_model


def setup_logger(log_file: str, log_level: str = "INFO") -> logging.Logger:
    """Sets up dual file and console logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("BenchmarkLogger")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML experiment configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(config: Dict[str, Any], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Runs systematic benchmark across model and feature set combinations."""
    data_cfg = config.get("data", {})
    feat_cfg = config.get("feature_selection", {})
    models_cfg = config.get("models", {})
    bench_cfg = config.get("benchmark", {})
    guardrail_cfg = config.get("guardrails", {})
    out_cfg = config.get("output", {})

    logger.info("Starting IoT DDoS Classification Benchmark Pipeline")
    logger.info(f"Target dataset sample: {data_cfg.get('sample_path')}")

    # 1. Load data
    X, y = load_dataset(
        file_path=data_cfg.get("sample_path"),
        target_column=data_cfg.get("target_column", "label"),
        binary_classification=data_cfg.get("binary_classification", True)
    )
    logger.info(f"Loaded dataset: {len(X)} samples, {X.shape[1]} raw features")

    # 2. Extract feature subsets
    feature_subsets = extract_feature_subsets(
        X=X,
        top_10_features=feat_cfg.get("top_10_features", [])
    )

    models_to_test = [
        ("RandomForest", "random_forest", models_cfg.get("random_forest", {})),
        ("LightGBM", "lightgbm", models_cfg.get("lightgbm", {}))
    ]

    feature_conditions = [
        ("Full_46_Features", "full"),
        ("Top_10_Features", "top_10")
    ]

    results_records: List[Dict[str, Any]] = []

    for model_display_name, model_key, model_params in models_to_test:
        for feat_name, feat_key in feature_conditions:
            condition_id = f"{model_display_name}__{feat_name}"
            logger.info(f"--> Executing condition: {condition_id}")

            X_subset = feature_subsets[feat_key]
            X_train, X_test, y_train, y_test = prepare_train_test_split(
                X=X_subset,
                y=y,
                test_size=data_cfg.get("test_size", 0.2),
                random_state=config.get("experiment", {}).get("seed", 42),
                stratify=data_cfg.get("stratify", False)
            )

            # Build model
            model = build_model(
                model_name=model_key,
                params=model_params,
                sample_count=len(X_train)
            )

            # Measure memory & fit
            mem_metrics = measure_memory_overhead(model, X_train, y_train, X_test)

            # Evaluate predictions
            y_pred = model.predict(X_test)
            class_metrics = calculate_classification_metrics(y_test, y_pred)

            # Measure inference performance
            perf_metrics = measure_inference_performance(
                model=model,
                X_test=X_test,
                batch_sizes=bench_cfg.get("batch_sizes", [1, 16]),
                iterations=bench_cfg.get("iterations", 10),
                warmup_runs=bench_cfg.get("warmup_runs", 3)
            )

            single_sample_lat = perf_metrics.get("batch_1", {}).get("per_sample_latency_ms", 0.0)

            # Evaluate guardrails
            guard_status = evaluate_guardrails(
                f1_score_val=class_metrics["f1_macro"],
                ram_mb_val=mem_metrics["rss_memory_mb"],
                latency_ms_val=single_sample_lat,
                config_guardrails=guardrail_cfg
            )

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "model": model_display_name,
                "feature_set": feat_name,
                "feature_count": X_subset.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": class_metrics["accuracy"],
                "f1_macro": class_metrics["f1_macro"],
                "precision_macro": class_metrics["precision_macro"],
                "recall_macro": class_metrics["recall_macro"],
                "latency_batch1_ms": single_sample_lat,
                "throughput_samples_sec": perf_metrics.get("batch_1", {}).get("throughput_samples_sec", 0.0),
                "ram_rss_mb": mem_metrics["rss_memory_mb"],
                "ram_traced_peak_mb": mem_metrics["traced_peak_mb"],
                "guardrail_passed": guard_status["all_passed"],
                "performance_details": perf_metrics,
                "guardrail_details": guard_status
            }
            results_records.append(record)

            logger.info(
                f"Result: F1={record['f1_macro']} | Latency={record['latency_batch1_ms']} ms | "
                f"RAM={record['ram_rss_mb']} MB | Guardrails Passed={record['guardrail_passed']}"
            )

    # 3. Export structured results
    out_dir = out_cfg.get("output_dir", "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = out_cfg.get("results_json", os.path.join(out_dir, "benchmark_results.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    logger.info(f"Saved benchmark JSON results to: {json_path}")

    csv_path = out_cfg.get("results_csv", os.path.join(out_dir, "benchmark_results.csv"))
    flat_records = []
    for r in results_records:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
        flat_records.append(flat)
    df_results = pd.DataFrame(flat_records)
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark CSV summary to: {csv_path}")

    # 4. Print Summary Table
    logger.info("======================== BENCHMARK SUMMARY ========================")
    summary_cols = ["condition_id", "feature_count", "f1_macro", "latency_batch1_ms", "ram_rss_mb", "guardrail_passed"]
    logger.info("\n" + df_results[summary_cols].to_string(index=False))
    logger.info("===================================================================")

    return results_records


def main():
    parser = argparse.ArgumentParser(description="IoT Edge DDoS Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log_file = config.get("output", {}).get("log_file", "results/benchmark.log")
    logger = setup_logger(log_file=log_file)

    try:
        run_experiment(config=config, logger=logger)
    except Exception as e:
        logger.exception(f"Fatal error during benchmark run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
