# BEARING FAULT DETECTION — Automated Test & Model Validation Results

**Execution Environment:** Windows (`win32`), Python `3.10.10`, `pytest-9.1.1`, PyTorch `CUDA` / CPU  
**Command Executed:** `python -m pytest -v`  
**Overall Result:** **14 / 14 PASSED (100% Pass Rate in 30.54s)**

---

## 1. Automated Pytest Suite Results

```text
============================= test session starts =============================
platform win32 -- Python 3.10.10, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\amank\Downloads\FULL WORK FINAL\bearing-digital-twin-version-2\bearing-digital-twin
collected 14 items

tests/test_core.py::test_cwru_characteristic_frequencies PASSED          [  7%]
tests/test_core.py::test_segment_shape PASSED                            [ 14%]
tests/test_core.py::test_group_split_has_no_leakage PASSED               [ 21%]
tests/test_core.py::test_validation_flags_bad_signals PASSED             [ 28%]
tests/test_core.py::test_physics_evidence_finds_outer_race_peak PASSED   [ 35%]
tests/test_core.py::test_health_index_orders_healthy_above_shifted PASSED [ 42%]
tests/test_dl_smoke.py::test_all_models_train_two_epochs PASSED          [ 50%]
tests/test_rag_and_api.py::test_retriever_finds_frequency_note PASSED    [ 57%]
tests/test_rag_and_api.py::test_template_answer_never_changes_prediction PASSED [ 64%]
tests/test_rag_and_api.py::test_hybrid_rag_and_typo_tolerance PASSED     [ 71%]
tests/test_rag_and_api.py::test_api_roundtrip PASSED                     [ 78%]
tests/test_rag_and_api.py::test_real_dataset_samples_cwru_and_ims PASSED [ 85%]
tests/test_rag_and_api.py::test_invalid_inputs_rejected PASSED           [ 92%]
tests/test_rag_and_api.py::test_rbac_and_model_switching PASSED          [100%]

============================= 14 passed in 30.54s =============================
```

---

## 2. Detailed Test Coverage Matrix

| Test Case | Module / Scope | What Is Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_cwru_characteristic_frequencies` | `physics/characteristic_frequency.py` | Verifies SKF 6205 analytical $BPFO$, $BPFI$, $BSF$, $FTF$ at 1,797 RPM against published benchmark values. | `PASSED` |
| `test_segment_shape` | `preprocessing/segmentation.py` | Verifies sliding-window segmentation (`8192` samples, `50%` overlap) output tensor dimensions. | `PASSED` |
| `test_group_split_has_no_leakage` | `preprocessing/splitting.py` | Verifies bearing-level group splitting and `assert_no_leakage` enforcement. | `PASSED` |
| `test_validation_flags_bad_signals` | `preprocessing/cleaning.py` | Negative test: verifies rejection of `NaN`/`Inf`, flatline constant signals, and short recordings. | `PASSED` |
| `test_physics_evidence_finds_outer_race_peak` | `physics/physics_features.py` | Verifies Hilbert envelope demodulation detects outer-race $BPFO$ peak and `evidence_supports` returns `True`. | `PASSED` |
| `test_health_index_orders_healthy_above_shifted` | `health/health_index.py` | Verifies Mahalanobis `HealthIndex` assigns ~95–100 to healthy baseline and monotonic lower scores to degraded windows. | `PASSED` |
| `test_all_models_train_two_epochs` | `deep_learning/train_dl.py` | Trains and evaluates `cnn`, `cnn_gru`, `multimodal`, and `pi_da` (with target domain adaptation) end-to-end. | `PASSED` |
| `test_retriever_finds_frequency_note` | `rag/retriever.py` | Verifies dense retrieval finds `characteristic_frequencies.md` for kinematic queries. | `PASSED` |
| `test_template_answer_never_changes_prediction` | `rag/llm.py` | Verifies RAG assistant grounds explanation in active prediction without altering fault class or confidence. | `PASSED` |
| `test_hybrid_rag_and_typo_tolerance` | `rag/hybrid_retriever.py`, `rag/human_assistant.py` | Verifies BM25 + Dense RRF retrieval and Levenshtein typo normalization (`"wat is bpfi and why is my bearin noisy?"`). | `PASSED` |
| `test_api_roundtrip` | `backend/main.py` | End-to-end `/simulate` + `/predict` roundtrip across all 4 fault classes (`healthy`, `inner_race`, `outer_race`, `ball`), plus `/fft`, `/physics`, `/bearing`, `/history`, and `/chat`. | `PASSED` |
| `test_real_dataset_samples_cwru_and_ims` | `backend/main.py`, `backend/services/inference.py` | Loads genuine recorded sensor windows from CWRU 12k and NASA IMS 20k via `/dataset/sample`, runs real model inference via `/predict`, and verifies provenance metadata (`Recorded Data`). | `PASSED` |
| `test_invalid_inputs_rejected` | `backend/main.py` | Negative API tests: verifies HTTP `422` for short windows (`< 2048` samples) and negative RPM, and HTTP `404` for unknown bearing IDs. | `PASSED` |
| `test_rbac_and_model_switching` | `backend/services/auth.py`, `backend/main.py` | Verifies product identity (`Bearing Fault Detection`), HTTP `401` on invalid credentials, HTTP `403` when `operator` accesses Super Admin routes, and `200` when `superadmin` switches models (`cwru_cnn`, `random_forest`), calibrates thresholds, and queries the audit trail. | `PASSED` |

---

## 3. Validated ML & DL Model Performance Summary

| Model Artifact | Architecture | Dataset & Geometry | Validation Macro-F1 | Held-Out Test Macro-F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/ml/ml_bundle.joblib` | Random Forest (`30` selected features) | CWRU 12k Drive-End + Physics Kinematic Windows (`cwru_6205`) | `0.9881` | `1.0000` | `VERIFIED` |
| `models/ml/nasa_ims_bundle.joblib` | Random Forest + Mahalanobis `HealthIndex` | NASA IMS Run-to-Failure 2nd Test (`ims_zA2115`, `20 kHz`) | `0.9680` | `0.9664` (`97.0%` Acc) | `VERIFIED` |
| `models/dl/cnn.pt` | 1D Convolutional Neural Network | CWRU 12k Drive-End (`cwru_windows.npz`, `8192` samples) | `1.0000` | `1.0000` | `VERIFIED` |
| `models/dl/pi_da.pt` | Physics-Informed Domain-Adaptive Net (`PINN` + `GRL`) | Source: CWRU 12k $\rightarrow$ Target: Synthetic (`8192` samples) | `1.0000` | `1.0000` | `VERIFIED` |
| `models/dl/nasa_ims_cnn.pt` | Deep 1D-CNN (`PyTorch CUDA`) | NASA IMS Run-to-Failure 2nd Test (`8192` samples) | `0.9520` | `0.9485` | `VERIFIED` |
