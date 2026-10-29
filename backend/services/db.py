"""SQLite persistence (swap the connection for PostgreSQL in production)."""
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from configs.config import ROOT

DB_PATH = os.getenv("BDT_DB", str(ROOT / "data/bdt.sqlite3"))
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
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init():
    with closing(_conn()) as c:
        c.executescript(SCHEMA)
        c.commit()


def _run(sql, args=()):
    with closing(_conn()) as c:
        cur = c.execute(sql, args)
        c.commit()
        return cur


def save_prediction(r):
    ts = time.time()
    _run("INSERT INTO predictions(bearing_id,ts,fault,confidence,health,alert,payload) VALUES(?,?,?,?,?,?,?)",
         (r["bearing_id"], ts, r["fault"], r["confidence"], r["health"], r["alert"]["level"],
          json.dumps({k: v for k, v in r.items() if k not in ("spectra",)}, default=str)))
    s = r["sensors"]
    _run("INSERT INTO sensor_readings(bearing_id,ts,rpm,temperature,vibration_rms) VALUES(?,?,?,?,?)",
         (r["bearing_id"], ts, s.get("rpm"), s.get("temperature"), s.get("vibration_rms")))
    if r["alert"]["level"] != "normal":
        _run("INSERT INTO alerts(bearing_id,ts,level,reasons) VALUES(?,?,?,?)",
             (r["bearing_id"], ts, r["alert"]["level"], "; ".join(r["alert"]["reasons"])))


def history(bearing_id, limit=100):
    with closing(_conn()) as c:
        rows = c.execute("SELECT ts,fault,confidence,health,alert FROM predictions WHERE bearing_id=? "
                         "ORDER BY id DESC LIMIT ?", (bearing_id, limit)).fetchall()
    return [dict(zip(("ts", "fault", "confidence", "health", "alert"), r)) for r in reversed(rows)]


def recent_predictions(limit=50):
    with closing(_conn()) as c:
        rows = c.execute("SELECT id,bearing_id,ts,fault,confidence,health,alert FROM predictions "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(("id", "bearing_id", "ts", "fault", "confidence", "health", "alert"), r)) for r in rows]


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

