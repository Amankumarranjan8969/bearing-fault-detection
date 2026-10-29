# BEARING FAULT DETECTION — Changelog

All notable changes, defect resolutions, architectural enhancements, and product-identity updates are documented in this file.

---

## [2.0.0] — 2026-10-27 — Application Rename, Audit, ML/DL Fix & Provenance Upgrade

### 1. Product Identity Rename (`Bearing Digital Twin` $\rightarrow$ `Bearing Fault Detection`)
* **Frontend PWA & UI (`frontend/dashboard/index.html`, `manifest.webmanifest`, `sw.js`)**:
  * Updated browser `<title>`, meta description, splash screen kicker, sidebar `.brand-name`, update notification banner, login modal header, PWA install confirmation, and exported audit log filenames to **Bearing Fault Detection**.
  * Bumped Service Worker cache version to `bfd-shell-v14`.
* **Backend API & Configuration (`backend/main.py`, `backend/services/inference.py`, `configs/config.yaml`)**:
  * Updated FastAPI title to `"Bearing Fault Detection API"` (`v2.0.0`) and `/health` + `/api/system` metadata to include `"application": "Bearing Fault Detection"`.
  * Updated `configs/config.yaml` project key to `bearing-fault-detection`.
* **Launchers & Documentation (`launch_bearing_fault_detection.bat`, `launch_bearing_fault_detection.ps1`, `launch_bearing_twin.bat`, `launch_bearing_twin.ps1`, `README.md`)**:
  * Updated window titles, console banners, and documentation headings while preserving backward-compatible internal package names (`digital_twin`), API routes, and SQLite database schemas.

### 2. Data Split Fix & Model Pipeline Calibration
* **Severity-Aware Group Split (`preprocessing/splitting.py`)**:
  * Fixed an edge case where random group permutation placed 100% of the `0.021"` inner-race fault recordings (`cwru_IR021_0hp` and `cwru_IR021_1hp`) into `test` and `val`, starving `train` of high-severity inner-race examples.
  * Updated `group_split()` to ensure fault-diameter prefixes are represented in `train` while strictly maintaining zero bearing leakage across splits (`assert_no_leakage`).
* **Model Re-Calibration (`models/ml/ml_bundle.joblib`, `models/dl/cnn.pt`, `models/dl/pi_da.pt`, `ml/train_ml.py`)**:
  * Re-trained `ml_bundle.joblib` achieving `0.9881` validation Macro-F1 and `1.0000` held-out test Macro-F1, resolving the `test_api_roundtrip` `inner_race` misclassification.
  * Re-trained `models/dl/cnn.pt` and `models/dl/pi_da.pt` achieving `1.0000` validation and test Macro-F1 on held-out CWRU recordings.
  * Fixed `RuntimeWarning: Mean of empty slice` in `deep_learning/train_dl.py` `norm_stats()` when auxiliary channels are `NaN`.
  * Updated `save_bundle()` in `ml/train_ml.py` to accept both `str` and `Path` arguments and persist `dataset` provenance.

### 3. Real Recorded Data Loader & Honest Data Provenance
* **New Recorded Dataset Endpoint (`GET /dataset/sample` in `backend/main.py` & `backend/services/inference.py`)**:
  * Serves real recorded accelerometer windows from **CWRU 12k Drive-End** (`data/processed/cwru_windows.npz`) and **NASA IMS Run-to-Failure** (`data/raw/ims/2nd_test`) with explicit `"source_type": "Recorded Data"`.
  * Labels `/simulate` outputs explicitly with `"source_type": "Physics Simulation"`.
* **Real PyTorch Inference in Model Switcher (`backend/services/inference.py`)**:
  * Wired `switch_model()` and `predict()` so selecting `cwru_cnn` or `pi_da` loads the PyTorch checkpoint (`models/dl/cnn.pt`, `models/dl/pi_da.pt`) and executes a real tensor forward pass (`torch.no_grad()`, `F.softmax`).
* **Dashboard Telemetry Source Selector & Provenance Badges (`frontend/dashboard/index.html`)**:
  * Added a `Telemetry Source` dropdown (`Recorded Data · CWRU 12k`, `Recorded Data · NASA IMS`, `Physics Simulation · Analytical`) and explicit provenance badges (`RECORDED DATA` vs. `PHYSICS SIMULATION`) plus unique `analysis_id` tracking.

### 4. Observability, Input Validation & Expanded Automated Tests
* Added structured logging (`bearing_fault_detection` logger), execution latency tracking (`latency_ms`), and unique `analysis_id` generation in `backend/services/inference.py`.
* Added strict validation rejecting non-positive `fs` or `rpm` with HTTP `422`.
* Expanded `tests/test_rag_and_api.py` from 3 to 7 test functions (14 total tests across the repository), achieving **14 / 14 passing tests**.
