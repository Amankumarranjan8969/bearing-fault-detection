# Bearing Fault Detection

**Physics-Informed AI Bearing Fault Detection, Digital Twin & Predictive Maintenance Platform**

`Bearing Fault Detection` is an end-to-end industrial condition monitoring and fault diagnosis platform that combines analytical bearing kinematics ($BPFO, BPFI, BSF, FTF$), Hilbert envelope demodulation, classical machine learning ensembles (`Random Forest`, `LightGBM`, `XGBoost`, `SVM`), Physics-Informed Neural Networks (`PINN` + Domain-Adversarial `DANN`), Mahalanobis + Kalman Digital Twin degradation tracking, and a Hybrid BM25 + Dense Vector RAG Engineering Assistant.

---

## Key Capabilities

* **Real Multi-Dataset Benchmark Support & Explicit Provenance**:
  * **CWRU 12k Drive-End Bearing Benchmark** (SKF 6205-2RS JEM, $f_s = 12,000\text{ Hz}$, 1,082 windows across 22 recordings).
  * **NASA IMS Run-to-Failure Dataset** (Rexnord ZA-2115 double-row bearing, $f_s = 20,000\text{ Hz}$, 984 run-to-failure files).
  * **Physics-Informed Analytical Simulator** (`GET /simulate`) for controlled fault-severity wear replay, explicitly badged as `Physics Simulation` vs. `Recorded Data` (`GET /dataset/sample`).
* **Zero-Leakage Group Splitting**:
  * `preprocessing/splitting.py` enforces bearing-level group splits (`assert_no_leakage`) with fault-diameter prefix representation so no recording crosses `train`/`val`/`test`.
* **Physics-Informed Signal Processing & Feature Engineering**:
  * Butterworth bandpass filtering (`10–5500 Hz`), Hilbert resonance envelope demodulation (`1500–5000 Hz`), and 44–58 time, spectral, envelope, motor-current, and kinematic ratio features (`phys_BPFO_ratio`, `phys_BPFI_ratio`, `phys_BSF_ratio`, `phys_FTF_ratio`).
* **Classical ML & Deep Learning / PINN Models**:
  * **Tabular Ensemble (`models/ml/ml_bundle.joblib`)**: Random Forest / LightGBM (`val_macro_f1 = 0.9881`, `test_macro_f1 = 1.0000`).
  * **NASA IMS Run-to-Failure Model (`models/ml/nasa_ims_bundle.joblib`)**: Random Forest (`test_acc = 97.0%`, `test_macro_f1 = 0.9664`).
  * **Deep Learning & PINN Checkpoints (`models/dl/*.pt`)**: `cnn.pt`, `multimodal.pt`, `pi.pt`, `da.pt`, `pi_da.pt` (`source_test_f1 = 1.0000`), and `nasa_ims_cnn.pt`.
* **Digital Twin & Health Prognostics**:
  * Mahalanobis distance `HealthIndex` (0–100) smoothed by a scalar Kalman filter (`ScalarKalman`), least-squares degradation slope (`fit_trend`), indicative steps-to-threshold, and multi-factor ISO-aligned alert levels (`normal`, `warning`, `degradation`, `critical`).
* **Hybrid BM25 + Dense RAG Engineering Assistant**:
  * Combines Okapi BM25 ($k_1=1.5, b=0.75$) lexical retrieval with Dense TF-IDF/SentenceTransformer embeddings via Reciprocal Rank Fusion (RRF) and a Levenshtein domain typo normalizer.
* **Enterprise Role-Based Access Control (RBAC)**:
  * **Super Admin** (`superadmin` / `SuperAdmin@2026!`): Full Master Control (Digital Twin schematic, Knowledge Base ingestion, live model switching, threshold calibration, telemetry purge, compliance audit export).
  * **Plant Admin** (`admin` / `Admin@2026!`): Plant reliability operations, telemetry analysis, wear replay.
  * **Field Operator** (`operator` / `Operator@2026!`): Field monitoring, signal FFT/Envelope inspection, diagnosis, and RAG assistant.

---

## Project Structure

```text
bearing-digital-twin/
├── configs/                # Central YAML configuration & bearing geometries (SKF 6205, ZA-2115)
├── data/                   # Raw (CWRU, NASA IMS), processed (.npz windows), features (.csv), metadata
├── preprocessing/          # Signal validation, Butterworth filtering, segmentation, group splitting
├── physics/                # Bearing geometry, BPFO/BPFI/BSF/FTF equations, Hilbert envelope evidence
├── feature_engineering/    # Time, FFT, envelope, motor current, and dataset builder
├── ml/                     # Classical ML models, tuning, feature selection, bundle training
├── deep_learning/          # 1D-CNN, CNN-GRU/LSTM, Multimodal CNN, PyTorch training loop
├── physics_informed/       # Physics-informed spectral alignment & consistency loss
├── domain_adaptation/      # Gradient Reversal Layer (DANN) & CORAL loss
├── health/                 # Mahalanobis HealthIndex, trend fitting, alert rules
├── digital_twin/           # Per-bearing ScalarKalman state estimator & history buffer
├── explainability/         # Local SHAP / importance-weighted deviation attributions
├── rag/                    # Hybrid BM25 + Dense Retriever, Typo Normalizer, HumanAssistant, KB docs
├── evaluation/             # Metrics, calibration, robustness, ML/DL ablation studies
├── backend/                # FastAPI application (main.py) & services (inference, auth, db)
├── frontend/dashboard/     # Production PWA interface (index.html, dashboard.css, tokens.css, sw.js)
├── notebooks/              # Complete standalone backend ML/DL pipeline (bearing.ipynb)
├── reports/                # Audit report, test verification results, and changelog
└── tests/                  # Automated pytest suite (core physics, DL smoke, RAG, API, RBAC)
```

---

## Quick Start

### 1. One-Click Launch (Windows)
Double-click `launch_bearing_fault_detection.bat` (or `launch_bearing_twin.bat`), or run in PowerShell:
```powershell
.\launch_bearing_fault_detection.ps1
```

### 2. Manual CLI Start
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Then open **http://localhost:8000** in your browser.

---

## Training & Evaluation Commands

```bash
# 1. Prepare datasets (synthetic, CWRU, Paderborn, or NASA IMS)
python -m scripts.prepare_data --dataset cwru --raw-dir data/raw/cwru --geometry cwru_6205

# 2. Train classical ML models
python -m ml.train_ml --features data/features/cwru_features.csv

# 3. Train Deep Learning / Physics-Informed Domain-Adaptive models
python -m deep_learning.train_dl --model pi_da --source data/processed/cwru_windows.npz --target data/processed/synthetic_windows.npz --epochs 15

# 4. Run Ablation Study
python -m evaluation.ablation --features data/features/cwru_features.csv

# 5. Run Full Automated Test Suite
python -m pytest -v
```
