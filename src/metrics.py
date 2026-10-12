import time
import tracemalloc
from typing import Any, Dict, List
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Computes standard evaluation metrics."""
    # Zero division handled safely for edge cases
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4)
    }


def measure_inference_performance(
    model: Any,
    X_test,
    batch_sizes: List[int],
    iterations: int = 10,
    warmup_runs: int = 3
) -> Dict[str, Any]:
    """Measures inference latency and throughput across batch sizes."""
    results = {}

    n_samples = len(X_test)

    # Warmup runs
    warmup_slice = X_test.iloc[:min(1, n_samples)] if hasattr(X_test, "iloc") else X_test[:min(1, n_samples)]
    for _ in range(warmup_runs):
        _ = model.predict(warmup_slice)

    for batch_size in batch_sizes:
        batch_slice = X_test.iloc[:min(batch_size, n_samples)] if hasattr(X_test, "iloc") else X_test[:min(batch_size, n_samples)]
        actual_batch_size = len(batch_slice)

        latencies_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model.predict(batch_slice)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

        avg_latency = float(np.mean(latencies_ms))
        p99_latency = float(np.percentile(latencies_ms, 99))
        per_sample_latency = avg_latency / max(1, actual_batch_size)
        throughput = (actual_batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

        results[f"batch_{batch_size}"] = {
            "avg_latency_ms": round(avg_latency, 4),
            "p99_latency_ms": round(p99_latency, 4),
            "per_sample_latency_ms": round(per_sample_latency, 4),
            "throughput_samples_sec": round(throughput, 2)
        }

    return results


def measure_memory_overhead(model: Any, X_train, y_train, X_test) -> Dict[str, float]:
    """Measures peak memory usage during training and inference."""
    process = psutil.Process()
    base_rss = process.memory_info().rss / (1024 * 1024)

    tracemalloc.start()
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    post_rss = process.memory_info().rss / (1024 * 1024)

    return {
        "traced_peak_mb": round(peak_mem / (1024 * 1024), 4),
        "rss_memory_mb": round(post_rss, 2),
        "rss_delta_mb": round(max(0.0, post_rss - base_rss), 2)
    }


def evaluate_guardrails(
    f1_score_val: float,
    ram_mb_val: float,
    latency_ms_val: float,
    config_guardrails: Dict[str, float]
) -> Dict[str, Any]:
    """Checks whether measured metrics meet predefined safety guardrails."""
    f1_min = config_guardrails.get("f1_min_threshold", 0.95)
    max_ram = config_guardrails.get("max_ram_limit_mb", 256.0)
    max_lat = config_guardrails.get("max_latency_limit_ms", 15.0)

    f1_pass = bool(f1_score_val >= f1_min)
    ram_pass = bool(ram_mb_val <= max_ram)
    latency_pass = bool(latency_ms_val <= max_lat)

    return {
        "f1_guardrail": {"threshold": f1_min, "measured": f1_score_val, "passed": f1_pass},
        "ram_guardrail": {"limit_mb": max_ram, "measured": ram_mb_val, "passed": ram_pass},
        "latency_guardrail": {"limit_ms": max_lat, "measured": latency_ms_val, "passed": latency_pass},
        "all_passed": f1_pass and ram_pass and latency_pass
    }
