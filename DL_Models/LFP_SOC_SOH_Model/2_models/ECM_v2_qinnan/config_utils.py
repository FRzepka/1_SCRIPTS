import yaml
from pathlib import Path

def load_config():
    SCRIPT_DIR = Path(__file__).resolve().parent
    path = SCRIPT_DIR / "config.yaml"
    with open(path,"r") as f:
         cfg=yaml.safe_load(f)
    return cfg