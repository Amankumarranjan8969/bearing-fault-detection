"""Tiny JSON model registry: name, version, dataset, features, params, val/test scores, date."""
import json
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(__file__).resolve().parent / "registry.json"


def _load():
    return json.loads(PATH.read_text()) if PATH.exists() else []


def register(name, kind, dataset, features, params, val_score, test_score, path, extra=None):
    items = _load()
    version = 1 + sum(1 for i in items if i["name"] == name)
    entry = {"name": name, "version": f"v{version}", "kind": kind, "dataset": dataset,
             "n_features": len(features) if features is not None else None, "params": params,
             "val_score": val_score, "test_score": test_score, "path": str(path),
             "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    items.append(entry)
    PATH.write_text(json.dumps(items, indent=2, default=str))
    return entry


def latest(name=None):
    items = [i for i in _load() if name is None or i["name"] == name]
    return items[-1] if items else None
