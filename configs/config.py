from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)
