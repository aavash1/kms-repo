#src/core/config.py
import yaml
from pathlib import Path

def load_ocr_config():
    config_path = Path(__file__).parent.parent / "configs" / "ocr_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)