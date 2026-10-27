"""Shared validation and portable, atomic session output."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


API_ROOT = next(p for p in Path(__file__).resolve().parents
                if p.name == "api" and (p / "src/ELMiRA").is_dir())
LOG_ROOT = API_ROOT / "logs/active_vision"


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".pending_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("T_wc must be a finite rigid camera-optical-to-world transform")
    return matrix


def positive(value, label):
    value = finite(value, label)
    if value <= 0:
        raise ValueError(label + " must be positive")
    return value
