"""FastAPI service for Bearing Fault Detection.
Run with: uvicorn backend.main:app --reload  ->  dashboard at http://localhost:8000"""
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from configs.config import ROOT
from backend.services import db, auth
from backend.services.inference import InferenceService
from backend.services.reporting_and_rescue import KAGGLE_SOURCES, parse_uploaded_bearing_file

logger = logging.getLogger("bearing_fault_detection")
svc: InferenceService | None = None


@asynccontextmanager
async def lifespan(app):
    global svc
    svc = InferenceService()
    yield


app = FastAPI(
    title="Bearing Fault Detection API",
    description="Physics-Informed AI Bearing Fault Detection, Digital Twin & Predictive Maintenance Service",
    version="2.0.0",
    lifespan=lifespan,
)


_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class PredictRequest(BaseModel):
    bearing_id: str = "B001"
    fs: float = 12000
    rpm: float
    vibration: list[float]
    current1: list[float] | None = None
    current2: list[float] | None = None
    torque: float | None = None
    force: float | None = None
    temperature: float | None = None
    source_type: str | None = None
    dataset: str | None = None


class BatchRequest(BaseModel):
    items: list[PredictRequest]


class ChatRequest(BaseModel):
    bearing_id: str = "B001"
    question: str


class DocumentRequest(BaseModel):
    name: str
    text: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ModelSwitchRequest(BaseModel):
    model: str


class ThresholdCalibrateRequest(BaseModel):
    alert_thresholds: dict | None = None
    rapid_slope: float | None = None
    temp_warn_c: float | None = None


class ResetHistoryRequest(BaseModel):
    bearing_id: str | None = None


class AlertSettingsRequest(BaseModel):
    popup_alerts_enabled: bool | None = None
    notify_above_normal: bool | None = None
    sound_visual_siren: bool | None = None
    auto_rescue_enabled: bool | None = None
    auto_rescue_threshold: float | None = None


class DispatchRescueRequest(BaseModel):
    bearing_id: str = "B001"
    action_type: str = "VFD Speed Derating & Piezo Lubrication"
    message: str = "Immediate maintenance and load relief protocol initiated by Super Admin."
    target_rpm: float | None = 1150.0


class FileUploadAnalyzeRequest(BaseModel):
    filename: str
    content_base64: str
    bearing_id: str | None = None
    rpm: float | None = None
    fs: float | None = None


class TrainingRunRequest(BaseModel):
    dataset: str = "paderborn"
    model_target: str | None = None
    task_type: str | None = None


def get_current_user(authorization: str | None = Header(None)):
    user = auth.validate_token(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    return user


def require_super_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "super_admin":
        raise HTTPException(403, "Super Admin privilege required")
    return user


def _predict(req):
    try:
        return svc.predict(req.model_dump())
    except ValueError as e:
        logger.warning("Validation error on /predict: %s", e)
        raise HTTPException(422, str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    return _predict(req)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return [_predict(r) for r in req.items]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "application": "Bearing Fault Detection",
        "model": svc.bundle["name"],
        "bearings": list(svc.twin.twins),
    }


@app.get("/bearing/{bearing_id}")
def bearing(bearing_id: str):
    st = svc.twin.state(bearing_id)
    if not st:
        raise HTTPException(404, "no data for this bearing yet, POST /predict first")
    return {"state": st, "latest": svc.latest[bearing_id]["result"]}


@app.get("/history/{bearing_id}")
def history(bearing_id: str, limit: int = 100):
    return db.history(bearing_id, limit)


@app.get("/fft/{bearing_id}")
def fft(bearing_id: str):
    s = svc.spectra(bearing_id)
    if not s:
        raise HTTPException(404, "no data for this bearing yet")
    return s


@app.get("/physics/{bearing_id}")
def physics(bearing_id: str):
    L = svc.latest.get(bearing_id)
    if not L:
        raise HTTPException(404, "no data for this bearing yet")
    return L["result"]["physics"]


@app.post("/chat")
def chat(req: ChatRequest):
    return svc.chat(req.bearing_id, req.question)


@app.post("/documents")
def documents(req: DocumentRequest, user: dict = Depends(require_super_admin)):
    return {"chunks_added": svc.retriever.add_document(req.name, req.text)}


@app.get("/api/system")
def api_system():
    return svc.system_info()


@app.get("/api/documents")
def api_documents(user: dict = Depends(require_super_admin)):
    return svc.system_info()["rag"]


@app.get("/api/documents/secret-of-all")
def get_secret_of_all(format: str = Query("markdown", pattern="^(markdown|html|json)$")):
    """Returns the comprehensive Secret of All Bearing Digital Twin Engineering Blueprint."""
    secret_path = ROOT / "docs/THE_SECRET_OF_ALL_BEARING_FAULT_DETECTION.md"
    if not secret_path.exists():
        secret_path = ROOT / "rag/knowledge_base/the_secret_of_all_bearing_digital_twin.md"
    if not secret_path.exists():
        raise HTTPException(404, "Secret of All document not found on disk")
    content = secret_path.read_text(encoding="utf-8")
    if format == "json":
        return {
            "title": "The Secret of All — Bearing Fault Detection Engineering Blueprint",
            "filename": "THE_SECRET_OF_ALL_BEARING_FAULT_DETECTION.md",
            "size_bytes": len(content.encode("utf-8")),
            "content": content
        }
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="THE_SECRET_OF_ALL_BEARING_FAULT_DETECTION.md"'}
    )


@app.get("/model/info")
def model_info():
    return svc.model_info()


@app.get("/model/metrics")
def model_metrics():
    return svc.bundle.get("metrics", {})


@app.get("/simulate")
def simulate(fault: str = Query("inner_race", pattern="^(healthy|inner_race|outer_race|ball)$"),
             severity: float = 0.8, rpm: float = 1800):
    """Physics-simulated sensor window for analytical testing and wear replay."""
    return svc.simulate(fault, severity, rpm)


@app.get("/dataset/sample")
def dataset_sample(dataset: str = Query("paderborn", pattern="^(paderborn|ims|nasa_ims|cwru|realtime)$"),
                   fault: str = Query("inner_race", pattern="^(healthy|inner_race|outer_race|ball)$"),
                   bearing_id: str | None = None):
    """Real recorded sensor window loaded directly from Paderborn 6203, NASA IMS, or CWRU 12k dataset on disk."""
    ds_norm = "ims" if dataset == "nasa_ims" else dataset
    return svc.dataset_sample(ds_norm, fault, bearing_id=bearing_id)


@app.post("/api/auth/login")
def login(req: LoginRequest):
    uname = req.username.strip().lower()
    if uname == "suhani@2026":
        uname = "superadmin"
    user = auth.USERS.get(uname)
    if not user or not auth.verify_password(req.password, user):
        raise HTTPException(401, "Invalid username or password")
    token = auth.create_token(user["username"])
    return {
        "status": "ok",
        "token": token,
        "user": {
            "username": user["username"],
            "name": user["name"],
            "role": user["role"],
            "title": user["title"],
            "badge": user["badge"],
            "permissions": user["permissions"]
        }
    }


@app.get("/api/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return {
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "title": user["title"],
        "badge": user["badge"],
        "permissions": user["permissions"]
    }


@app.get("/api/auth/users")
def list_users(admin: dict = Depends(require_super_admin)):
    return [
        {
            "username": u["username"],
            "name": u["name"],
            "role": u["role"],
            "title": u["title"],
            "badge": u["badge"],
            "permissions": u["permissions"]
        }
        for u in auth.USERS.values()
    ]


@app.post("/api/admin/model-switch")
def admin_model_switch(req: ModelSwitchRequest, admin: dict = Depends(require_super_admin)):
    res = svc.switch_model(req.model)
    auth.log_audit(admin["username"], "MODEL_SWITCH", f"Switched model architecture to {req.model}")
    return res


@app.post("/api/admin/thresholds")
def admin_thresholds(req: ThresholdCalibrateRequest, admin: dict = Depends(require_super_admin)):
    res = svc.calibrate_thresholds(req.alert_thresholds, req.rapid_slope, req.temp_warn_c)
    auth.log_audit(admin["username"], "CALIBRATE_THRESHOLDS", f"Calibrated thresholds: {req.alert_thresholds}")
    return res


@app.post("/api/admin/system/reset-history")
def admin_reset_history(req: ResetHistoryRequest, admin: dict = Depends(require_super_admin)):
    res = svc.reset_bearing_history(req.bearing_id)
    auth.log_audit(admin["username"], "RESET_HISTORY", f"Reset history for {req.bearing_id or 'all'}")
    return res


@app.get("/api/admin/audit-log")
def admin_audit_log(admin: dict = Depends(require_super_admin)):
    return auth.AUDIT_LOGS


@app.get("/api/reports/fleet")
def get_fleet_report():
    rows = svc.rescue_mgr.get_combined_fleet_rows()
    crit = sum(1 for r in rows if str(r.get("severity_level")).upper() == "CRITICAL")
    deg = sum(1 for r in rows if str(r.get("severity_level")).upper() == "DEGRADATION")
    warn = sum(1 for r in rows if str(r.get("severity_level")).upper() == "WARNING")
    norm = sum(1 for r in rows if str(r.get("severity_level")).upper() == "NORMAL")
    return {
        "summary": {
            "total_bearings": len(rows),
            "critical": crit,
            "degradation": deg,
            "warning": warn,
            "normal": norm,
            "sorted_by": "Criticality (CRITICAL at top, lowest health first)",
        },
        "kaggle_sources": KAGGLE_SOURCES,
        "rows": rows,
    }


@app.get("/api/reports/download")
def download_fleet_report(format: str = Query("html", pattern="^(html|csv|json)$")):
    fmt = format.lower().strip()
    if fmt == "csv":
        data = svc.rescue_mgr.generate_csv_bytes()
        return Response(
            content=data,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="Bearing_Fault_Detection_All_Bearings_Data.csv"'},
        )
    if fmt == "json":
        payload = json.dumps(get_fleet_report(), indent=2)
        return Response(
            content=payload.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="Bearing_Fault_Detection_All_Bearings_Data.json"'},
        )
    html_str = svc.rescue_mgr.generate_html_report(svc.assistant)
    return Response(
        content=html_str.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="Bearing_Fault_Detection_Comprehensive_Report.html"'},
    )


@app.get("/api/reports/single/{bearing_id}")
def download_single_bearing_report(bearing_id: str):
    L = svc.latest.get(bearing_id)
    if not L and svc.latest:
        L = next(iter(svc.latest.values()))
    if not L:
        sample = svc.dataset_sample("paderborn", "inner_race")
        res = svc.predict(sample)
    else:
        res = L["result"]
    html_str = svc.rescue_mgr.generate_single_bearing_html_report(res, svc.assistant)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in bearing_id)[:32]
    return Response(
        content=html_str.encode("utf-8"),
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="Bearing_Fault_Detection_{safe_id}_Report.html"'},
    )


@app.post("/api/upload-analyze")
def upload_and_analyze(req: FileUploadAnalyzeRequest):
    try:
        b64 = req.content_base64
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64)
        parsed = parse_uploaded_bearing_file(
            filename=req.filename,
            raw_bytes=raw_bytes,
            bearing_id=req.bearing_id,
            rpm=req.rpm,
            fs=req.fs,
        )
        return svc.predict(parsed)
    except Exception as e:
        logger.warning("Upload-analyze error on %s: %s", req.filename, e)
        raise HTTPException(422, f"Could not analyze uploaded file '{req.filename}': {e}")


@app.get("/api/sample-file")
def download_sample_upload_file(fault: str = Query("inner_race", pattern="^(healthy|inner_race|outer_race|ball)$")):
    w = svc.dataset_sample("paderborn", fault)
    vib = w["vibration"][:4096]
    c1 = (w.get("current1") or [0.0] * len(vib))[:4096]
    c2 = (w.get("current2") or [0.0] * len(vib))[:4096]
    lines = ["vibration,current1,current2,rpm,torque,force,temperature,fs"]
    rpm = w.get("rpm", 1500.0)
    tq = w.get("torque", 0.7)
    fc = w.get("force", 1000.0)
    tp = w.get("temperature", 55.0)
    fs = w.get("fs", 64000.0)
    for i in range(len(vib)):
        lines.append(f"{vib[i]},{c1[i]},{c2[i]},{rpm},{tq},{fc},{tp},{fs}")
    csv_bytes = ("\n".join(lines)).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="Bearing_Fault_Detection_Sample_{fault}.csv"'},
    )


@app.get("/api/notifications")
def get_notifications():
    return {
        "settings": svc.rescue_mgr.settings,
        "notifications": list(svc.rescue_mgr.notifications)[:30],
        "rescue_history": list(svc.rescue_mgr.rescue_history)[:20],
    }


@app.post("/api/admin/alert-settings")
def update_alert_settings(req: AlertSettingsRequest, admin: dict = Depends(require_super_admin)):
    updated = svc.rescue_mgr.update_settings(req.model_dump(exclude_none=True))
    auth.log_audit(admin["username"], "ALERT_SETTINGS", f"Updated alert & auto-rescue settings: {updated}")
    return {"status": "ok", "settings": updated}


@app.post("/api/admin/dispatch-rescue")
def dispatch_manual_rescue(req: DispatchRescueRequest, admin: dict = Depends(require_super_admin)):
    res = svc.rescue_mgr.dispatch_manual_feedback_or_rescue(
        admin_user=admin["username"],
        bearing_id=req.bearing_id,
        action_type=req.action_type,
        message=req.message,
        target_rpm=req.target_rpm,
    )
    auth.log_audit(admin["username"], "DISPATCH_RESCUE", f"Dispatched rescue on {req.bearing_id}: {req.action_type}")
    return res


@app.get("/api/overview")
def get_overview_summary():
    return svc.overview_summary()


@app.get("/api/fleet")
def get_fleet_paginated(
    q: str = Query(""),
    search: str = Query(""),
    dataset: str = Query("all"),
    entity_type: str = Query("all"),
    severity: str = Query("all"),
    fault: str = Query("all"),
    min_health: float = Query(0.0),
    max_health: float = Query(100.0),
    min_confidence: float = Query(0.0),
    date_from: str = Query(""),
    sort_by: str = Query("severity"),
    sort_dir: str = Query("asc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
):
    return svc.rescue_mgr.query_fleet_paginated(
        search=q or search,
        dataset=dataset,
        entity_type=entity_type,
        severity=severity,
        fault=fault,
        min_health=min_health,
        max_health=max_health,
        min_confidence=min_confidence,
        date_from=date_from,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )


@app.get("/api/models/center")
def get_models_center():
    return svc.models_center()


@app.get("/api/datasets/center")
def get_datasets_center():
    return svc.datasets_center()


@app.post("/api/training/run")
def run_training_job(req: TrainingRunRequest):
    return svc.run_training_evaluation_job(req.dataset, req.model_target, req.task_type)


@app.get("/api/training/status")
def get_training_status():
    return {"jobs": getattr(svc, "_training_jobs", [])}


@app.get("/api/reports/history")
def get_reports_history(q: str = Query(""), search: str = Query(""), limit: int = Query(50, ge=1, le=200)):
    items = db.recent_predictions(limit=limit)
    query_str = (q or search or "").strip().lower()
    if query_str:
        items = [
            r for r in items
            if query_str in str(r.get("bearing_id", "")).lower()
            or query_str in str(r.get("fault", "")).lower()
            or query_str in str(r.get("alert", "")).lower()
        ]
    return {
        "total": len(items),
        "available_reports": [
            {
                "id": "fleet-html",
                "title": "Executive & Technical All-Bearings Condition Report (HTML/PDF)",
                "format": "HTML / Print-to-PDF",
                "url": "/api/reports/download?format=html",
                "filename": "Bearing_Fault_Detection_Comprehensive_Report.html",
            },
            {
                "id": "fleet-csv",
                "title": "Unified All-Bearings Multi-Dataset Telemetry Matrix (CSV)",
                "format": "CSV",
                "url": "/api/reports/download?format=csv",
                "filename": "Bearing_Fault_Detection_All_Bearings_Data.csv",
            },
            {
                "id": "fleet-json",
                "title": "Complete Fleet & Model Consensus Payload (JSON)",
                "format": "JSON",
                "url": "/api/reports/download?format=json",
                "filename": "Bearing_Fault_Detection_All_Bearings_Data.json",
            },
        ],
        "analysis_history": items,
    }


app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dashboard", html=True), name="dashboard")
