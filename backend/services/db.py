"""SQLite persistence (swap the connection for PostgreSQL in production)."""
import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from configs.config import ROOT

logger = logging.getLogger("bearing_fault_detection")

# Detect read-only serverless environment (Vercel / AWS Lambda)
IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("LAMBDA_TASK_ROOT"))

if IS_SERVERLESS:
    DEFAULT_DB = "/tmp/bdt.sqlite3"
else:
    DEFAULT_DB = str(ROOT / "data/bdt.sqlite3")

DB_PATH = os.getenv("BDT_DB", DEFAULT_DB)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, role TEXT);
CREATE TABLE IF NOT EXISTS machines(id TEXT PRIMARY KEY, name TEXT, location TEXT);
CREATE TABLE IF NOT EXISTS bearings(id TEXT PRIMARY KEY, machine_id TEXT, geometry_key TEXT, installed TEXT);
CREATE TABLE IF NOT EXISTS sensor_readings(id INTEGER PRIMARY KEY, bearing_id TEXT, ts REAL, rpm REAL, temperature REAL, vibration_rms REAL);
CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, bearing_id TEXT, ts REAL, fault TEXT, confidence REAL, health REAL, alert TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, bearing_id TEXT, ts REAL, level TEXT, reasons TEXT);
CREATE TABLE IF NOT EXISTS model_versions(id INTEGER PRIMARY KEY, name TEXT, version TEXT, info TEXT);
CREATE TABLE IF NOT EXISTS chat(id INTEGER PRIMARY KEY, bearing_id TEXT, ts REAL, question TEXT, answer TEXT);
CREATE TABLE IF NOT EXISTS maintenance(id INTEGER PRIMARY KEY, bearing_id TEXT, ts REAL, note TEXT);
"""


def _conn():
    try:
        p = Path(DB_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        logger.warning("Could not connect to sqlite at %s (%s). Falling back to in-memory DB.", DB_PATH, e)
        return sqlite3.connect(":memory:")


def init():
    try:
        with closing(_conn()) as c:
            c.executescript(SCHEMA)
            c.commit()
    except Exception as e:
        logger.warning("DB init skipped: %s", e)


def _run(sql, args=()):
    try:
        with closing(_conn()) as c:
            cur = c.execute(sql, args)
            c.commit()
            return cur
    except Exception as e:
        logger.error("DB query error: %s", e)
        return None


def save_prediction(r):
    ts = time.time()
    _run("INSERT INTO predictions(bearing_id,ts,fault,confidence,health,alert,payload) VALUES(?,?,?,?,?,?,?)",
         (r["bearing_id"], ts, r["fault"], r["confidence"], r["health"], r["alert"]["level"],
          json.dumps({k: v for k, v in r.items() if k not in ("spectra",)}, default=str)))
    s = r["sensors"]
    _run("INSERT INTO sensor_readings(bearing_id,ts,rpm,temperature,vibration_rms) VALUES(?,?,?,?,?)",
         (r["bearing_id"], ts, s.get("rpm"), s.get("temperature"), s.get("vibration_rms")))
    if r.get("alert", {}).get("level") != "normal":
        _run("INSERT INTO alerts(bearing_id,ts,level,reasons) VALUES(?,?,?,?)",
             (r["bearing_id"], ts, r["alert"]["level"], "; ".join(r["alert"].get("reasons", []))))


def history(bearing_id, limit=100):
    try:
        with closing(_conn()) as c:
            rows = c.execute("SELECT ts,fault,confidence,health,alert FROM predictions WHERE bearing_id=? "
                             "ORDER BY id DESC LIMIT ?", (bearing_id, limit)).fetchall()
        return [dict(zip(("ts", "fault", "confidence", "health", "alert"), r)) for r in reversed(rows)]
    except Exception as e:
        logger.warning("DB history fetch failed: %s", e)
        return []


def recent_predictions(limit=50):
    try:
        with closing(_conn()) as c:
            rows = c.execute("SELECT id,bearing_id,ts,fault,confidence,health,alert FROM predictions "
                             "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(("id", "bearing_id", "ts", "fault", "confidence", "health", "alert"), r)) for r in rows]
    except Exception as e:
        logger.warning("DB recent predictions fetch failed: %s", e)
        return []


def save_chat(bearing_id, q, a):
    _run("INSERT INTO chat(bearing_id,ts,question,answer) VALUES(?,?,?,?)", (bearing_id, time.time(), q, a))


def add_maintenance(bearing_id, note):
    _run("INSERT INTO maintenance(bearing_id,ts,note) VALUES(?,?,?)", (bearing_id, time.time(), note))


def clear_history(bearing_id=None):
    if bearing_id:
        _run("DELETE FROM predictions WHERE bearing_id=?", (bearing_id,))
        _run("DELETE FROM sensor_readings WHERE bearing_id=?", (bearing_id,))
        _run("DELETE FROM alerts WHERE bearing_id=?", (bearing_id,))
    else:
        _run("DELETE FROM predictions")
        _run("DELETE FROM sensor_readings")
        _run("DELETE FROM alerts")