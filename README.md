# Comparative Analysis of Machine Learning Algorithms (Random Forest vs LightGBM) for Real-Time Anomaly and DDoS Attack Detection in Resource-Constrained IoT Networks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)
[![Benchmark](https://img.shields.io/badge/Dataset-CICIoT2023-orange.svg)](https://www.unb.ca/cic/datasets/iot-dataset-2023.html)

## Overview & Academic Background

The rapid proliferation of Internet of Things (IoT) devices in smart industrial and edge computing environments has drastically magnified the attack surface for distributed denial-of-service (DDoS) events. Traditional signature-based Intrusion Detection Systems (IDS) cannot keep pace with zero-day anomalies, while deep learning architectures (e.g., CNN-LSTM) introduce unacceptable memory and computational overhead for resource-constrained edge gateways.

This empirical research investigates the operational trade-offs between two prominent tree ensemble paradigms: **Random Forest** (bagging) and **LightGBM** (histogram-based gradient boosting). We evaluate how feature dimensionality reduction (full 46 features vs. top 10 flow features) impacts packet classification latency, memory footprint, and detection robustness using the benchmark **CICIoT2023** dataset.

## Authors

* **Bauyrzhan Balkybek** — Author & Developer ([GitHub: @Bauyrjan18](https://github.com/Bauyrjan18))
* Course: *Methods of Research (IT)*

---

## Research Questions & Hypotheses

### Research Question 1 (RQ1 — Detection Accuracy)
> *How do Random Forest and LightGBM compare in detection accuracy ($F_1$-score, Precision, Recall) when classifying DDoS traffic anomalies on resource-constrained IoT edge gateways?*

* **Null Hypothesis ($H_{0,1}$):** LightGBM exhibits no statistically significant advantage in $F_1$-score over Random Forest ($F_1^{\text{LightGBM}} \le F_1^{\text{RandomForest}}$).
* **Alternative Hypothesis ($H_{1,1}$):** LightGBM achieves a statistically superior $F_1$-score over Random Forest during high-rate IoT DDoS traffic ($F_1^{\text{LightGBM}} > F_1^{\text{RandomForest}}$), maintaining $F_1 \ge 0.95$.

### Research Question 2 (RQ2 — Dimensionality Reduction & Efficiency)
> *To what extent does feature dimensionality reduction affect per-packet processing latency and RAM usage in LightGBM versus Random Forest during real-time IoT attack classification?*

* **Null Hypothesis ($H_{0,2}$):** Feature reduction to the top 10 informative features does not reduce inference latency or RAM consumption in LightGBM more effectively than in Random Forest, or causes unacceptable accuracy degradation ($F_1 < 0.95$).
* **Alternative Hypothesis ($H_{1,2}$):** Reducing the feature space to the top 10 features reduces per-sample inference latency by $\ge 30\%$ and cuts RAM usage by $\ge 25\%$ in LightGBM relative to Random Forest while preserving $F_1 \ge 0.95$.

---

## Repository Structure

```text
.
├── .gitignore                     # Strict VCS ignore rules for Python, caches, and models
├── LICENSE                        # Open-source MIT License
├── README.md                      # Academic project documentation and execution guide
├── METHODOLOGY.md                 # Formal Research Methodology Passport (Card)
├── Methodology_Card.pdf           # Printable PDF Methodology Passport Card
├── Dockerfile                     # Reproducible container specification with exact pins
├── requirements.txt               # Fully pinned Python dependencies
├── environment.yml                # Conda environment specification
├── configs/
│   └── config.yaml                # Experiment parameters, feature lists, and thresholds
├── data/
│   └── sample/
│       └── ciciot2023_sample.csv  # Micro-sample of CICIoT2023 for pipeline validation
├── src/
│   ├── __init__.py                # Package initialization
│   ├── benchmark.py               # Main experiment runner and latency/RAM profiler
│   ├── data_loader.py             # Dataset ingestion, binary labeling, and feature splits
│   ├── models.py                  # Model instantiators for Random Forest and LightGBM
│   └── metrics.py                 # High-resolution timing, memory tracking, and guardrails
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebook for sample exploration
├── docs/
│   ├── METHODOLOGY.md             # Methodology card documentation copy
│   ├── methodology_card.html      # Styled HTML source for PDF export
│   └── Methodology_Card.pdf       # Archival PDF export
└── results/
    ├── benchmark_results.json     # Detailed latency percentiles and guardrail outputs
    ├── benchmark_results.csv      # Flat tabular benchmark summary
    └── benchmark.log              # Timestamped execution log
```

---

## Experimental Matrix & Planned Benchmarks

| Condition ID | Model Architecture | Feature Subset | Input Dimension | Primary Metric Target | Guardrail Thresholds | Baseline Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RF__Full` | Random Forest | Full Features | 46 features | Macro $F_1$, Latency | $F_1 \ge 0.95$, $\text{RAM} \le 256$ MB | **Control Baseline** |
| `RF__Top10` | Random Forest | Selected Flow Features | 10 features | Macro $F_1$, Latency | $F_1 \ge 0.95$, $\text{RAM} \le 256$ MB | Optimization Variant |
| `LGBM__Full` | LightGBM Classifier | Full Features | 46 features | Macro $F_1$, Latency | $F_1 \ge 0.95$, $\text{RAM} \le 256$ MB | Experimental Variant |
| `LGBM__Top10`| LightGBM Classifier | Selected Flow Features | 10 features | Macro $F_1$, Latency | $F_1 \ge 0.95$, $\text{RAM} \le 256$ MB | Experimental Variant |

### Selected Top-10 Flow Features
1. `flow_duration`
2. `Header_Length`
3. `Rate`
4. `Srate`
5. `Tot sum`
6. `Number`
7. `Magnitue`
8. `Radius`
9. `Variance`
10. `ack_count`

---

## System Requirements

* **Operating System:** Linux, macOS, or Windows 10/11
* **Python Runtime:** Python `>= 3.12`
* **Container Runtime:** Docker Engine 24+ (optional for containerized execution)
* **Memory & Storage:** $\ge 512$ MB RAM, $\approx 300$ MB free disk space

---

## Quickstart Guide

The experiment pipeline is designed to execute deterministically in **2–3 commands**.

### Option A: Running via Python Virtual Environment (Recommended)

```bash
# 1. Clone repository and navigate to root
git clone https://github.com/Bauyrjan18/Method.git
cd Method

# 2. Create virtual environment and install pinned dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the benchmark harness on sample data
python src/benchmark.py --config configs/config.yaml
```

### Option B: Running via Docker Container

```bash
# 1. Build the container image
docker build -t iot-ids-benchmark .

# 2. Run benchmark pipeline
docker run --rm -v $(pwd)/results:/app/results iot-ids-benchmark
```

---

## Sample Execution Output

Upon completion, the harness generates console logs and writes structured artifacts:

```text
======================== BENCHMARK SUMMARY ========================
                  condition_id  feature_count  f1_macro  latency_batch1_ms  ram_rss_mb  guardrail_passed
RandomForest__Full_46_Features             46       1.0            11.7295      159.11              True
 RandomForest__Top_10_Features             10       1.0            11.3202      156.76              True
    LightGBM__Full_46_Features             46       1.0             1.7220      159.92              True
     LightGBM__Top_10_Features             10       1.0             1.4785      160.24              True
===================================================================
```

* Detailed JSON report: `results/benchmark_results.json`
* Tabular CSV summary: `results/benchmark_results.csv`
* Execution log: `results/benchmark.log`

---

## Citation & License

This project is licensed under the terms of the [MIT License](LICENSE).

```bibtex
@misc{balkybek2026iotbenchmark,
  author = {Balkybek, Bauyrzhan},
  title = {Comparative Analysis of Machine Learning Algorithms (Random Forest vs LightGBM) for Real-Time Anomaly and DDoS Attack Detection in Resource-Constrained IoT Networks},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Bauyrjan18/Method}}
}
```
