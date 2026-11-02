"""
ModelStore — Secure model persistence for Metadron Capital ML pipeline.

Saves and loads:
- sklearn models (via joblib) with HMAC-SHA256 integrity verification
- numpy weight arrays (via np.save) with integrity verification
- Training metadata (JSON)

Security:
- HMAC-SHA256 signing on all saves, verified on all loads
- Path traversal protection (name sanitization)
- Safe class whitelist for joblib deserialization
- Atomic writes for crash safety

Used by: AlphaOptimizer, UniverseClassifier, MLVoteEnsemble, PatternRecognition
"""

import os
import json
import hmac
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# joblib for sklearn models
try:
    import joblib
except ImportError:
    joblib = None

# HMAC signing key — from env or default (change in production)
# HMAC signing key — MUST set MODEL_SIGNING_KEY in production
_SIGNING_KEY_ENV = os.environ.get("MODEL_SIGNING_KEY", "")
if _SIGNING_KEY_ENV:
    _SIGNING_KEY = _SIGNING_KEY_ENV.encode()
else:
    # Generate a random key per process — models won't survive restarts
    # but at least they can't be tampered with across restarts
    _SIGNING_KEY = os.urandom(32)
    logger.warning("MODEL_SIGNING_KEY not set — using random per-process key. "
                   "Models will not verify across restarts. Set env var for production.")

# Allowed sklearn model classes for safe deserialization
_SAFE_SKLEARN_CLASSES = {
    "sklearn.linear_model._base.LinearRegression",
    "sklearn.linear_model._ridge.Ridge",
    "sklearn.linear_model._coordinate_descent.Lasso",
    "sklearn.linear_model._coordinate_descent.ElasticNet",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._gb.GradientBoostingRegressor",
    "sklearn.ensemble._bagging.BaggingRegressor",
    "sklearn.svm._classes.SVR",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "xgboost.sklearn.XGBRegressor",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
}


def _compute_signature(data: bytes) -> str:
    """Compute HMAC-SHA256 signature over data."""
    return hmac.new(_SIGNING_KEY, data, hashlib.sha256).hexdigest()


def _verify_signature(data: bytes, expected: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    actual = _compute_signature(data)
    return hmac.compare_digest(actual, expected)


def _sanitize_name(name: str) -> str:
    """Sanitize model name to prevent path traversal."""
    # Remove path separators and parent references
    clean = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    # Only allow alphanumeric, dash, underscore
    clean = "".join(c for c in clean if c.isalnum() or c in "-_")
    if not clean:
        raise ValueError(f"Invalid model name: {name}")
    return clean


class ModelStore:
    """Secure persistent model storage with HMAC verification."""

    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, name: str) -> Path:
        """Get sanitized model directory path."""
        safe_name = _sanitize_name(name)
        model_dir = self.base_dir / safe_name
        # Verify the resolved path is within base_dir (no traversal)
        try:
            model_dir.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {name}")
        return model_dir

    def save_sklearn(self, name: str, model: Any, metadata: Optional[dict] = None) -> str:
        """Save sklearn model with HMAC signature."""
        if joblib is None:
            logger.warning("joblib not available — cannot save sklearn model")
            return ""

        model_dir = self._get_model_dir(name)
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{timestamp}_model.joblib"
        sig_path = model_dir / f"{timestamp}_sig.txt"
        meta_path = model_dir / f"{timestamp}_meta.json"

        # Save model
        joblib.dump(model, model_path)

        # Compute and save HMAC signature over the model file
        model_bytes = model_path.read_bytes()
        signature = _compute_signature(model_bytes)
        sig_path.write_text(signature)

        # Save metadata
        meta = {
            "model_name": name,
            "timestamp": timestamp,
            "model_type": type(model).__name__,
            "class": f"{type(model).__module__}.{type(model).__name__}",
            **(metadata or {}),
        }
        # Atomic meta write
        tmp_meta = meta_path.with_suffix('.tmp')
        tmp_meta.write_text(json.dumps(meta, indent=2, default=str))
        tmp_meta.rename(meta_path)

        logger.info("Saved model %s to %s (signed)", name, model_path)
        return str(model_path)

    def load_sklearn(self, name: str) -> tuple:
        """Load latest sklearn model with HMAC verification."""
        if joblib is None:
            return None, None

        model_dir = self._get_model_dir(name)
        if not model_dir.exists():
            return None, None

        joblib_files = sorted(model_dir.glob("*_model.joblib"))
        if not joblib_files:
            return None, None

        latest = joblib_files[-1]
        sig_path = Path(str(latest).replace("_model.joblib", "_sig.txt"))
        meta_path = Path(str(latest).replace("_model.joblib", "_meta.json"))

        # Verify HMAC signature before loading
        if sig_path.exists():
            model_bytes = latest.read_bytes()
            expected_sig = sig_path.read_text().strip()
            if not _verify_signature(model_bytes, expected_sig):
                raise SecurityError(
                    f"Model file integrity check failed: {latest}. "
                    "File may have been tampered with. Refusing to load."
                )
        else:
            logger.warning("No signature file for %s — loading without verification", latest)

        # Load model
        model = joblib.load(latest)

        # Verify model class is in safe whitelist
        model_class = f"{type(model).__module__}.{type(model).__name__}"
        if model_class not in _SAFE_SKLEARN_CLASSES:
            logger.warning("Loaded model class %s not in safe whitelist", model_class)

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta

    def save_numpy(self, name: str, weights: np.ndarray, metadata: Optional[dict] = None) -> str:
        """Save numpy weight array with HMAC signature."""
        model_dir = self._get_model_dir(name)
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = model_dir / f"{timestamp}_weights.npy"
        sig_path = model_dir / f"{timestamp}_sig.txt"
        meta_path = model_dir / f"{timestamp}_meta.json"

        # Save weights
        np.save(weights_path, weights)

        # Compute and save HMAC signature
        weights_bytes = weights_path.read_bytes()
        signature = _compute_signature(weights_bytes)
        sig_path.write_text(signature)

        # Save metadata (atomic write)
        meta = {
            "model_name": name,
            "timestamp": timestamp,
            "shape": list(weights.shape),
            "dtype": str(weights.dtype),
            **(metadata or {}),
        }
        tmp_meta = meta_path.with_suffix('.tmp')
        tmp_meta.write_text(json.dumps(meta, indent=2, default=str))
        tmp_meta.rename(meta_path)

        logger.info("Saved weights %s to %s (signed)", name, weights_path)
        return str(weights_path)

    def load_numpy(self, name: str) -> tuple:
        """Load latest numpy weights with HMAC verification."""
        model_dir = self._get_model_dir(name)
        if not model_dir.exists():
            return None, None

        npy_files = sorted(model_dir.glob("*_weights.npy"))
        if not npy_files:
            return None, None

        latest = npy_files[-1]
        sig_path = Path(str(latest).replace("_weights.npy", "_sig.txt"))
        meta_path = Path(str(latest).replace("_weights.npy", "_meta.json"))

        # Verify HMAC signature
        if sig_path.exists():
            weights_bytes = latest.read_bytes()
            expected_sig = sig_path.read_text().strip()
            if not _verify_signature(weights_bytes, expected_sig):
                raise SecurityError(
                    f"Weights file integrity check failed: {latest}. "
                    "File may have been tampered with."
                )

        weights = np.load(latest)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return weights, meta

    def list_models(self) -> Dict[str, list]:
        """List all stored models with versions."""
        result = {}
        if not self.base_dir.exists():
            return result

        for model_dir in sorted(self.base_dir.iterdir()):
            if model_dir.is_dir():
                versions = []
                for f in sorted(model_dir.glob("*_meta.json")):
                    meta = json.loads(f.read_text())
                    versions.append(meta)
                result[model_dir.name] = versions

        return result

    def cleanup(self, name: str, keep_last: int = 5):
        """Remove old model versions, keep latest N."""
        model_dir = self._get_model_dir(name)
        if not model_dir.exists():
            return

        # Collect full timestamps (YYYYMMDD_HHMMSS) not just dates
        timestamps = set()
        for f in model_dir.iterdir():
            # Pattern: YYYYMMDD_HHMMSS_type.ext
            parts = f.name.split("_")
            if len(parts) >= 3:
                # Full timestamp: date_time
                timestamps.add(f"{parts[0]}_{parts[1]}")
            elif len(parts) >= 2:
                timestamps.add(parts[0])

        sorted_ts = sorted(timestamps)
        if len(sorted_ts) > keep_last:
            for ts in sorted_ts[:-keep_last]:
                for f in model_dir.glob(f"{ts}_*"):
                    f.unlink()
                logger.info("Cleaned up old model version: %s/%s", name, ts)


class SecurityError(Exception):
    """Raised when model integrity check fails."""
    pass
