"""Shared utilities: logging, seeding, timing, JSON IO."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed python/numpy (and tensorflow, if importable) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: F401 - optional heavy dependency

        tf.random.set_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[{label}] took {elapsed:,.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
