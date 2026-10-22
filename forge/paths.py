"""Project paths and small shared helpers (hashing, seeding, config loading)."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ART / "data"
WORK = DATA / "work"
SHARDS = DATA / "shards"
MODELS = ART / "models"
REPORTS = ART / "reports"
CONFIGS = ROOT / "configs"
STRUCT_DIR = WORK / "structures"
FEFF_DIR = WORK / "feff"

for p in (DATA, WORK, SHARDS, MODELS, REPORTS, STRUCT_DIR, FEFF_DIR):
    p.mkdir(parents=True, exist_ok=True)


def load_env():
    """Read KEY=VALUE pairs from .env (gitignored) without ever printing values."""
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
