# Fraud Detection MLOps Pipeline

**Continual Learning with Plasticity-Stability Evaluation for Credit Card Fraud Detection**

Student: Syed Ibrahim Ali (i221872)
Course: MLOps — Track II (Technical Research)

---

## Research Contribution

This project implements and extends the work of Lebichot et al. (2024) on catastrophic forgetting in fraud detection models. The key research gap: existing fraud detection literature studies concept drift and continual learning *in isolation* but no production MLOps pipeline exists that:

1. Automatically detects distribution shift via dual-signal drift detection (PSI + KS)
2. Compares continual learning strategies using a rigorous Plasticity-Stability Matrix
3. Exposes all metrics in real-time through a production-grade observability stack

**Our contribution:** A fully operational MLOps pipeline that closes this gap, demonstrating that Experience Replay reduces catastrophic forgetting by **45.6%** compared to naive fine-tuning while maintaining plasticity.

---

## System Architecture

```
Kaggle ULB Dataset
       │
       ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Data Ingest │───▶│  Drift Detector  │───▶│  Continual Learner  │
│  (8 windows) │    │  PSI + KS test   │    │  FT / ER / SW       │
└─────────────┘    └──────────────────┘    └─────────────────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
                   ┌─────────────┐    ┌───────────────┐
                   │  MLflow     │    │  FastAPI       │
                   │  Tracking   │    │  /predict      │
                   └─────────────┘    │  /metrics      │
                                      └───────┬───────┘
                                              │
                                    ┌─────────▼────────┐
                                    │   Prometheus      │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼────────┐
                                    │     Grafana       │
                                    │  11-panel dashboard│
                                    └──────────────────┘

GitHub Actions CI/CD ──── test → build → (retrain on drift)
```

---

## Project Structure

```
mlops_project/
├── src/
│   ├── train.py              # Base model training (XGBoost + SMOTE)
│   ├── serve.py              # FastAPI inference server
│   ├── continual_train.py    # CL strategy comparison pipeline
│   ├── plasticity.py         # Plasticity-Stability Matrix computation
│   └── drift_trigger.py      # PSI + KS drift detection
├── tests/
│   └── test_serve.py         # API unit tests (CI-safe, no real model needed)
├── data/
│   ├── model.json            # Trained XGBoost model artifact
│   ├── feature_names.pkl     # Feature name list
│   └── *_plasticity_matrix.csv  # CL evaluation results
├── monitoring/
│   ├── prometheus.yml        # Scrape config
│   └── grafana/
│       ├── provisioning/     # Auto-provisioned datasource + dashboard
│       └── dashboards/fraud_pipeline.json
├── paper/
│   └── main.tex              # IEEE-format research paper (LaTeX)
├── docker-compose.yml        # Full stack orchestration
├── Dockerfile                # Fraud detection API image
├── Dockerfile.mlflow         # MLflow tracking server image
├── requirements.txt
└── .github/workflows/ci.yml  # CI/CD pipeline
```

---

## Experimental Results

### Base Model (XGBoost + SMOTE, trained on full dataset)

| Metric        | Value  |
|---------------|--------|
| ROC-AUC       | 0.9769 |
| F1-Score      | 0.7404 |
| Fraud Recall  | 0.8878 |
| Precision     | 0.6276 |

### Drift Detection (8 temporal windows)

Drift confirmed at **Window 4 → 5** (fraud rate spike from 0.10% → 0.60%):
- PSI fired on 100% of features (threshold 0.2)
- KS test fired on 100% of features (p < 0.05)

### Continual Learning Strategy Comparison

| Strategy           | Plasticity | Stability | Forgetting |
|--------------------|-----------|-----------|------------|
| Fine-Tuning        | 0.748     | 0.681     | 0.134      |
| **Experience Replay** | **0.775** | **0.752** | **0.073** |
| Sliding Window     | 0.761     | 0.729     | 0.098      |

Experience Replay achieves the best balance: highest plasticity (learns new patterns) + highest stability (retains old knowledge) + lowest forgetting.

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Kaggle API credentials (for dataset download)

### 1. Download Dataset

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

### 2. Train Base Model

```bash
pip install -r requirements.txt
python src/train.py --data data/creditcard.csv
```

### 3. Run Continual Learning Evaluation

```bash
python src/continual_train.py --data data/creditcard.csv
```

### 4. Start Full Stack

```bash
docker compose up -d
```

Services:
- **Fraud API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 5. Test Prediction API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
```

### 6. Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## API Endpoints

| Endpoint              | Method | Description                              |
|-----------------------|--------|------------------------------------------|
| `/health`             | GET    | Health check + model version             |
| `/predict`            | POST   | Fraud prediction (30 features)           |
| `/metrics`            | GET    | Prometheus metrics scrape endpoint       |
| `/update_cl_metrics`  | POST   | Push CL scores to Prometheus (internal)  |

---

## Monitoring

The Grafana dashboard (`fraud_pipeline.json`) includes 11 panels:

- Total Predictions, Fraud Rate (rolling 100), Latency p99
- Drift Detected (red/green indicator), PSI Ratio, KS Ratio
- Plasticity / Stability / Forgetting bars per CL strategy
- Prediction volume timeseries, Latency timeseries

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **test** — runs `pytest tests/ -v` on every push/PR to main
2. **build** — builds Docker image + smoke tests `/health` endpoint
3. **retrain** — triggered manually (`workflow_dispatch`) or via `repository_dispatch` (drift alert); downloads fresh data, retrains, commits updated model artifact

---

## Technology Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Model            | XGBoost 2.1.x (native Booster API)  |
| Imbalance        | SMOTE (imbalanced-learn 0.13)       |
| Experiment Track | MLflow 3.x (SQLite backend)         |
| Serving          | FastAPI + Uvicorn                   |
| Monitoring       | Prometheus + Grafana                |
| Containerization | Docker + Docker Compose             |
| CI/CD            | GitHub Actions                      |
| Drift Detection  | PSI + KS (scipy)                    |

---

## Compile the Paper

```bash
cd paper/
pdflatex main.tex
pdflatex main.tex   # run twice for cross-references
```

Requires: `texlive-full` or equivalent with `IEEEtran`, `pgfplots`, `tikz` packages.

---

## References

This work builds on and extends:

- Lebichot et al. (2024) — *Catastrophic forgetting in credit card fraud detection*, Expert Systems with Applications
- Dal Pozzolo et al. (2017) — *Credit card fraud detection: A realistic modeling*, IEEE TNNLS
- Chen & Guestrin (2016) — *XGBoost: A scalable tree boosting system*, KDD

Full bibliography in `paper/main.tex` (14 references).
