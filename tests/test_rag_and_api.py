import pytest
from configs.config import ROOT, load_config
from rag.retriever import Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.llm import MaintenanceAssistant
from rag.human_assistant import HumanAssistant


def test_retriever_finds_frequency_note():
    r = Retriever(ROOT / "rag/knowledge_base", embedder="tfidf")
    top = r.search("how is BPFI calculated from geometry", 2)
    assert top and top[0]["source"] == "characteristic_frequencies.md"


def test_template_answer_never_changes_prediction():
    r = Retriever(ROOT / "rag/knowledge_base", embedder="tfidf")
    res = {"fault": "outer_race", "confidence": 0.93, "uncertainty": "low", "health": 60, "alert": {"level": "warning"},
           "physics": {"supports_prediction": True}, "recommendation": "Schedule inspection."}
    out = MaintenanceAssistant(r, provider="template").answer("Why?", res)
    assert "outer race" in out["answer"] and "93%" in out["answer"]


def test_hybrid_rag_and_typo_tolerance():
    hr = HybridRetriever(ROOT / "rag/knowledge_base")
    ha = HumanAssistant(hr, top_k=3)
    res = {
        "fault": "inner_race",
        "confidence": 0.96,
        "uncertainty": "low",
        "health": 48.5,
        "alert": {"level": "degradation", "reasons": ["health score 49"]},
        "trend": {"label": "decreasing", "slope_per_update": -1.5, "indicative_steps_to_threshold": 19},
        "physics": {"characteristic_hz": {"BPFI": 162.19, "BPFO": 107.36, "BSF": 70.58, "FTF": 11.93},
                    "evidence": {"BPFI": {"expected_hz": 162.19, "observed_hz": 162.6, "peak_to_median": 8.4, "harmonic": 1}},
                    "supports_prediction": True},
        "sensors": {"rpm": 1797, "vibration_rms": 0.312},
        "recommendation": "Schedule a bearing inspection at the next planned stop.",
        "top_features": [{"feature": "phys_BPFI_ratio", "value": 8.4, "contribution": 0.42}],
    }
    ans = ha.answer("wat is bpfi and why is my bearin noisy?", res)
    assert "answer" in ans and len(ans["answer"]) > 50
    assert ans["corrections"]  # Typo normalizer caught "wat" / "bearin"


def test_api_roundtrip():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        for fault in ("healthy", "inner_race", "outer_race", "ball"):
            w = c.get("/simulate", params={"fault": fault, "severity": 0.9, "rpm": 1800}).json()
            assert w["source_type"] == "Physics Simulation"
            r = c.post("/predict", json={
                "bearing_id": "T",
                "fs": w["fs"],
                "rpm": w["rpm"],
                "vibration": w["vibration"],
                "current1": w["current1"],
                "current2": w["current2"],
                "temperature": w["temperature"],
                "source_type": w["source_type"],
                "dataset": w["dataset"],
            })
            assert r.status_code == 200
            body = r.json()
            assert body["fault"] == fault
            assert body["provenance"]["source_type"] == "Physics Simulation"
            assert body["analysis_id"].startswith("bfd-")

        assert c.get("/history/T").status_code == 200
        assert c.get("/fft/T").status_code == 200
        assert c.get("/physics/T").status_code == 200
        assert c.get("/bearing/T").status_code == 200
        assert c.post("/chat", json={"bearing_id": "T", "question": "Why?"}).status_code == 200


def test_real_dataset_samples_cwru_and_ims():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        # 1. CWRU 12k real recorded window
        for fault in ("healthy", "inner_race", "outer_race", "ball"):
            s_cwru = c.get("/dataset/sample", params={"dataset": "cwru", "fault": fault})
            assert s_cwru.status_code == 200
            w = s_cwru.json()
            assert w["source_type"] == "Recorded Data"
            assert "CWRU" in w["dataset"]
            pred = c.post("/predict", json={
                "bearing_id": "CWRU_TEST",
                "fs": w["fs"],
                "rpm": w["rpm"],
                "vibration": w["vibration"],
                "source_type": w["source_type"],
                "dataset": w["dataset"],
            })
            assert pred.status_code == 200
            p_json = pred.json()
            assert p_json["fault"] == fault
            assert p_json["provenance"]["source_type"] == "Recorded Data"

        # 2. NASA IMS 20k real recorded window
        s_ims = c.get("/dataset/sample", params={"dataset": "ims", "fault": "outer_race"})
        assert s_ims.status_code == 200
        w_ims = s_ims.json()
        assert w_ims["source_type"] == "Recorded Data"
        pred_ims = c.post("/predict", json={
            "bearing_id": "IMS_TEST",
            "fs": w_ims["fs"],
            "rpm": w_ims["rpm"],
            "vibration": w_ims["vibration"],
            "source_type": w_ims["source_type"],
            "dataset": w_ims["dataset"],
        })
        assert pred_ims.status_code == 200
        assert 0.0 <= pred_ims.json()["health"] <= 100.0

        # 3. Paderborn 6203 64kHz Multi-Sensor real recorded window + All 58 Features verification
        for fault in ("healthy", "inner_race", "outer_race", "ball"):
            s_pb = c.get("/dataset/sample", params={"dataset": "paderborn", "fault": fault})
            assert s_pb.status_code == 200
            w_pb = s_pb.json()
            assert w_pb["source_type"] == "Recorded Data"
            assert "Paderborn" in w_pb["dataset"]
            assert w_pb.get("current1") and w_pb.get("current2")
            pred_pb = c.post("/predict", json={
                "bearing_id": "PB_TEST",
                "fs": w_pb["fs"],
                "rpm": w_pb["rpm"],
                "vibration": w_pb["vibration"],
                "current1": w_pb["current1"],
                "current2": w_pb["current2"],
                "torque": w_pb["torque"],
                "force": w_pb["force"],
                "temperature": w_pb["temperature"],
                "source_type": w_pb["source_type"],
                "dataset": w_pb["dataset"],
            })
            assert pred_pb.status_code == 200
            pb_json = pred_pb.json()
            assert pb_json["n_extracted_features"] >= 50
            assert "all_features" in pb_json and "feature_groups" in pb_json
            assert set(pb_json["feature_groups"].keys()) == {
                "time_domain", "frequency_fft", "hilbert_envelope",
                "motor_current", "operating_load", "kinematic_physics"
            }


def test_invalid_inputs_rejected():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        # Short signal (< 2048 samples)
        r_short = c.post("/predict", json={"bearing_id": "BAD", "fs": 12000, "rpm": 1800, "vibration": [0.1] * 500})
        assert r_short.status_code == 422

        # Negative RPM
        r_rpm = c.post("/predict", json={"bearing_id": "BAD", "fs": 12000, "rpm": -500, "vibration": [0.1] * 4096})
        assert r_rpm.status_code == 422

        # Unknown bearing ID on /bearing and /fft
        assert c.get("/bearing/NON_EXISTENT_999").status_code == 404
        assert c.get("/fft/NON_EXISTENT_999").status_code == 404


def test_rbac_and_model_switching():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app
    cfg = load_config()
    assert cfg["project"] == "bearing-fault-detection"

    with TestClient(app) as c:
        sys_info = c.get("/api/system").json()
        assert sys_info["application"] == "Bearing Fault Detection"

        # Invalid login -> 401
        assert c.post("/api/auth/login", json={"username": "superadmin", "password": "wrong"}).status_code == 401

        # Operator login -> blocked from Super Admin endpoints (403)
        op_login = c.post("/api/auth/login", json={"username": "operator", "password": "Operator@2026!"})
        assert op_login.status_code == 200
        op_token = op_login.json()["token"]
        op_headers = {"Authorization": f"Bearer {op_token}"}
        assert c.get("/api/documents", headers=op_headers).status_code == 403
        assert c.post("/api/admin/model-switch", json={"model": "cwru_cnn"}, headers=op_headers).status_code == 403

        # Super Admin login -> full access to model switch, thresholds, audit log, and DL inference
        sa_login = c.post("/api/auth/login", json={"username": "superadmin", "password": "SuperAdmin@2026!"})
        assert sa_login.status_code == 200
        sa_token = sa_login.json()["token"]
        sa_headers = {"Authorization": f"Bearer {sa_token}"}

        # Switch to PyTorch 1D-CNN and run real inference
        sw_cnn = c.post("/api/admin/model-switch", json={"model": "cwru_cnn"}, headers=sa_headers)
        assert sw_cnn.status_code == 200 and sw_cnn.json()["active_model"] == "dl-cnn-cwru-12k"
        w = c.get("/dataset/sample", params={"dataset": "cwru", "fault": "outer_race"}).json()
        pred_cnn = c.post("/predict", json={"bearing_id": "CNN_TEST", "fs": w["fs"], "rpm": w["rpm"], "vibration": w["vibration"]})
        assert pred_cnn.status_code == 200 and pred_cnn.json()["fault"] == "outer_race"

        # Switch back to Random Forest default
        sw_rf = c.post("/api/admin/model-switch", json={"model": "random_forest"}, headers=sa_headers)
        assert sw_rf.status_code == 200

        # Calibrate thresholds & check audit trail
        th_res = c.post("/api/admin/thresholds", json={"alert_thresholds": {"normal": 70, "warning": 50, "degradation": 30}}, headers=sa_headers)
        assert th_res.status_code == 200
        audit_res = c.get("/api/admin/audit-log", headers=sa_headers)
        assert audit_res.status_code == 200 and len(audit_res.json()) >= 2


def test_comprehensive_fleet_report_and_auto_rescue():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as c:
        # 1. Verify unified fleet report across all 3 Kaggle datasets + Critical sorted at top
        fleet = c.get("/api/reports/fleet")
        assert fleet.status_code == 200
        f_json = fleet.json()
        assert f_json["summary"]["total_bearings"] >= 80
        assert f_json["summary"]["critical"] >= 20
        assert f_json["rows"][0]["severity_level"] == "CRITICAL"

        # 2. Verify downloadable HTML & CSV reports
        dl_html = c.get("/api/reports/download", params={"format": "html"})
        assert dl_html.status_code == 200
        assert "Bearing Fault Detection" in dl_html.text
        assert "RAG-Grounded Action Plan" in dl_html.text

        dl_csv = c.get("/api/reports/download", params={"format": "csv"})
        assert dl_csv.status_code == 200
        assert "priority_rank,severity_level,bearing_id" in dl_csv.text

        # 3. Verify multi-model consensus, future forecast, one-by-one criticality & auto-rescue on critical bearing
        w = c.get("/simulate", params={"fault": "inner_race", "severity": 1.35, "rpm": 1800}).json()
        for _ in range(4):
            r = c.post("/predict", json={
                "bearing_id": "CRIT_MOTOR_01",
                "fs": w["fs"],
                "rpm": w["rpm"],
                "vibration": w["vibration"],
                "current1": w["current1"],
                "current2": w["current2"],
                "temperature": 86.0,
            })
        assert r.status_code == 200
        body = r.json()
        assert "simple_explanation" in body
        assert len(body["multi_model_consensus"]) == 5
        assert "projected_health_10_steps" in body["future_forecast"]
        assert len(body["critical_factors_breakdown"]) >= 4
        assert body["auto_rescue"]["triggered"] is True
        assert body["notification"] is not None

        # 4. Verify Super Admin alert settings & manual rescue dispatch
        sa_login = c.post("/api/auth/login", json={"username": "superadmin", "password": "SuperAdmin@2026!"})
        sa_headers = {"Authorization": f"Bearer {sa_login.json()['token']}"}

        st_res = c.post("/api/admin/alert-settings", json={"auto_rescue_enabled": True, "auto_rescue_threshold": 30.0}, headers=sa_headers)
        assert st_res.status_code == 200
        assert st_res.json()["settings"]["auto_rescue_threshold"] == 30.0

        disp = c.post("/api/admin/dispatch-rescue", json={
            "bearing_id": "CRIT_MOTOR_01",
            "action_type": "VFD Speed Derating (-30%) + Piezo Grease Injection",
            "message": "Emergency rescue dispatched by Super Admin",
            "target_rpm": 1150.0,
        }, headers=sa_headers)
        assert disp.status_code == 200
        assert disp.json()["status"] == "ok"

        notifs = c.get("/api/notifications")
        assert notifs.status_code == 200
        assert len(notifs.json()["notifications"]) >= 2

        # 5. Verify 1-Click Sample File Download, 1-Click File Upload & Single-Bearing Report Download
        import base64
        sample_csv = c.get("/api/sample-file", params={"fault": "inner_race"})
        assert sample_csv.status_code == 200
        assert "Bearing_Fault_Detection_Sample_inner_race.csv" in sample_csv.headers.get("content-disposition", "")

        b64_csv = base64.b64encode(sample_csv.content).decode("ascii")
        up_res = c.post("/api/upload-analyze", json={
            "filename": "Bearing_Fault_Detection_Sample_inner_race.csv",
            "content_base64": b64_csv,
        })
        assert up_res.status_code == 200
        up_json = up_res.json()
        assert len(up_json["multi_model_consensus"]) == 5
        assert "simple_explanation" in up_json

        single_rep = c.get(f"/api/reports/single/{up_json['bearing_id']}")
        assert single_rep.status_code == 200
        assert "Bearing Fault Detection" in single_rep.text
        assert "What Is the Condition of the Bearing Right Now?" in single_rep.text


def test_10_section_industrial_dashboard_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as c:
        # 1. Section 1: Overview (/api/overview)
        ov = c.get("/api/overview")
        assert ov.status_code == 200
        ov_data = ov.json()
        assert ov_data["total_unique_bearings"] >= 80
        assert len(ov_data["five_most_critical_bearings"]) == 5
        assert "production_model" in ov_data
        assert "historical_health_trend" in ov_data
        assert "recent_system_events" in ov_data

        # 2. Section 2: Fleet Monitoring (/api/fleet) with search, filter, sort & pagination
        fl = c.get("/api/fleet", params={"severity": "CRITICAL", "page": 1, "page_size": 10})
        assert fl.status_code == 200
        fl_data = fl.json()
        assert len(fl_data["rows"]) <= 10
        assert all(r["severity_level"] == "CRITICAL" for r in fl_data["rows"])
        assert "entity_counts" in fl_data
        assert fl_data["entity_counts"]["unique_physical_assets"] >= 80
        assert fl_data["entity_counts"]["dataset_recordings"] >= 140

        # 3. Section 4, 5 & 7: Predict + 58-Feature Category Verification + Raw Waveform + Conflicting Evidence
        w = c.get("/dataset/sample", params={"dataset": "paderborn", "fault": "inner_race"}).json()
        pred = c.post("/predict", json={
            "bearing_id": "PB_TEST_01",
            "fs": w["fs"],
            "rpm": w["rpm"],
            "vibration": w["vibration"],
            "current1": w["current1"],
            "current2": w["current2"],
            "torque": w["torque"],
            "force": w["force"],
            "temperature": w["temperature"],
        })
        assert pred.status_code == 200
        p_data = pred.json()
        fg = p_data["feature_groups"]
        assert len(fg["time_domain"]) == 12
        assert len(fg["frequency_fft"]) == 15
        assert len(fg["hilbert_envelope"]) == 5
        assert len(fg["motor_current"]) == 14
        assert len(fg["operating_load"]) == 4
        assert len(fg["kinematic_physics"]) == 8
        assert sum(len(v) for v in fg.values()) == 58
        assert "conflicting_evidence" in p_data["physics"]
        assert p_data["auto_rescue"]["execution_mode"] == "SIMULATED_DECISION_SUPPORT"
        assert p_data["auto_rescue"]["verified_plc_connection"] is False

        fft_res = c.get("/fft/PB_TEST_01")
        assert fft_res.status_code == 200
        fft_data = fft_res.json()
        assert "raw_time" in fft_data and len(fft_data["raw_time"]) > 0
        assert "time" in fft_data and len(fft_data["time"]) > 0

        # 4. Section 8: AI Model Center (/api/models/center)
        mc = c.get("/api/models/center")
        assert mc.status_code == 200
        mc_data = mc.json()
        assert len(mc_data["models"]) == 5
        assert all("val_macro_f1" in m and "artifact_path" in m for m in mc_data["models"])

        # 5. Section 9: Dataset & Training Center (/api/datasets/center, /api/training/run)
        dc = c.get("/api/datasets/center")
        assert dc.status_code == 200
        dc_data = dc.json()
        assert len(dc_data["datasets"]) == 4
        tr = c.post("/api/training/run", json={"dataset": "paderborn", "model_target": "paderborn_rf", "task_type": "4-Class Fault Classification"})
        assert tr.status_code == 200
        tr_data = tr.json()
        assert tr_data["status"] == "COMPLETED"
        assert tr_data["metrics"]["val_macro_f1"] > 0.9

        # 6. Section 10: Reports & History (/api/reports/history)
        rh = c.get("/api/reports/history", params={"q": "PB_TEST_01"})
        assert rh.status_code == 200
        assert rh.json()["total"] >= 1



