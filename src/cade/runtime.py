from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from cade.autoencoder import Autoencoder, ContrastiveAE

if TYPE_CHECKING:
    from keras.models import Model


@dataclass(slots=True)
class CadeDetectionOutput:
    """Output from CADE drift detection on a batch or chunk."""

    row_flags: np.ndarray
    scores: np.ndarray
    closest_classes: np.ndarray
    chunk_drift: bool


class CadeRuntimeDetector:
    """Runtime-facing CADE detector for library use inside FIRCE."""

    def __init__(  # noqa: C901
        self,
        dims: list[int],
        margin: float = 10.0,
        mad_threshold: float = 3.5,
        min_drift_ratio: float = 0.05,
        min_drift_count: int = 1,
        cae_lambda_1: float = 1e-1,
        similar_ratio: float = 0.25,
        batch_size: int = 64,
        epochs: int = 250,
        lr: float = 1e-3,
        display_interval: int = 10,
        weights_path: str | None = None,
        device: str = "/CPU:0",
        *,
        force_retrain: bool = False,
    ) -> None:
        if len(dims) < 2:
            raise ValueError(
                "dims must contain at least input and latent dimensions."
            )
        if any(v <= 0 for v in dims):
            raise ValueError("All dims entries must be positive.")
        if margin < 0:
            raise ValueError("margin must be non-negative.")
        if mad_threshold < 0:
            raise ValueError("mad_threshold must be non-negative.")
        if not (0.0 <= min_drift_ratio <= 1.0):
            raise ValueError("min_drift_ratio must be in [0, 1].")
        if min_drift_count < 1:
            raise ValueError("min_drift_count must be >= 1.")
        if cae_lambda_1 < 0:
            raise ValueError("cae_lambda_1 must be non-negative.")
        if not (0.0 <= similar_ratio <= 1.0):
            raise ValueError("similar_ratio must be in [0, 1].")
        if batch_size < 4 or batch_size % 4 != 0:
            raise ValueError("batch_size must be a multiple of 4 and >= 4.")
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if lr <= 0:
            raise ValueError("lr must be > 0.")
        if display_interval < 1:
            raise ValueError("display_interval must be >= 1.")

        self.dims = dims
        self.margin = margin
        self.mad_threshold = mad_threshold
        self.min_drift_ratio = min_drift_ratio
        self.min_drift_count = min_drift_count

        self.cae_lambda_1 = cae_lambda_1
        self.similar_ratio = similar_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.display_interval = display_interval
        self.force_retrain = force_retrain
        self.device = device
        if weights_path is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="cade_runtime_"))
            self.weights_path = str(tmp_dir / "cae.weights.h5")
        else:
            self.weights_path = weights_path

        self._is_fitted = False

        # Filled during fit()
        self.classes_: np.ndarray | None = None
        self.class_to_idx_: dict[Any, int] | None = None
        self.centroids_: np.ndarray | None = None
        self.median_distances_: np.ndarray | None = None
        self.mad_distances_: np.ndarray | None = None
        self.encoder_: Model | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether detector state has been initialized by fit()."""
        return self._is_fitted

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the detector on training data.

        Steps:
            1. normalize labels to contiguous integer indices
            2. train/load contrastive autoencoder
            3. build/load encoder
            4. encode train data into latent space
            5. compute per-class centroids
            6. compute per-class median and MAD statistics
        """
        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train)

        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same length.")
        if x_train.shape[1] != self.dims[0]:
            raise ValueError(
                f"x_train has {x_train.shape[1]} features, "
                f"but dims[0] is {self.dims[0]}."
            )

        classes = np.unique(y_train)
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        y_encoded = np.asarray([class_to_idx[v]
                               for v in y_train], dtype=np.int32)

        if len(classes) < 2:
            raise ValueError(
                "CADE requires at least 2 classes in training data."
            )

        self.classes_ = classes
        self.class_to_idx_ = class_to_idx

        # Train contrastive AE
        optimizer = tf.keras.optimizers.Adam
        cae = ContrastiveAE(self.dims, optimizer, self.lr)
        if self.force_retrain and Path(self.weights_path).exists():
            Path(self.weights_path).unlink()

        with tf.device(self.device):
            cae.train(
                x_train=x_train,
                y_train=y_encoded,
                lambda_1=self.cae_lambda_1,
                batch_size=self.batch_size,
                epochs=self.epochs,
                similar_ratio=self.similar_ratio,
                margin=self.margin,
                weights_save_name=self.weights_path,
                display_interval=self.display_interval,
            )

            self.encoder_ = self._build_encoder(self.weights_path)
            z_train = self._encode(x_train)

        # Per-class latent groups
        z_family = [z_train[y_encoded == idx] for idx in range(len(classes))]
        if any(len(z) == 0 for z in z_family):
            raise RuntimeError(
                "One or more classes had no latent samples after encoding.")

        # Per-class centroids
        centroids = np.asarray(
            [np.mean(z_group, axis=0) for z_group in z_family],
            dtype=np.float64,
        )

        # Per-class distance distributions
        distance_lists: list[np.ndarray] = []
        median_distances: list[float] = []
        mad_distances: list[float] = []

        for idx, z_group in enumerate(z_family):
            dists = np.linalg.norm(
                z_group - centroids[idx], axis=1).astype(np.float64)
            distance_lists.append(dists)

            med = float(np.median(dists))
            mad = 1.4826 * float(np.median(np.abs(dists - med)))

            # Prevent division-by-zero / unstable anomaly scores
            mad = max(mad, 1e-12)

            median_distances.append(med)
            mad_distances.append(mad)

        self.centroids_ = centroids
        self.median_distances_ = np.asarray(median_distances, dtype=np.float64)
        self.mad_distances_ = np.asarray(mad_distances, dtype=np.float64)

        self._is_fitted = True

    def detect(self, x: np.ndarray) -> CadeDetectionOutput:
        """
        Detect drift on a batch of samples.

        For each sample:
            1. embed into latent space
            2. compute distance to each class centroid
            3. compute MAD-normalized anomaly score to each class
            4. score is the minimum anomaly score across classes
            5. closest class is the nearest centroid
            6. drift if min anomaly score > mad_threshold
        """
        self._require_fitted()

        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if x.shape[1] != self.dims[0]:
            raise ValueError(
                f"x has {x.shape[1]} features, but dims[0] is {self.dims[0]}."
            )

        if self.centroids_ is None:
            raise RuntimeError(
                "centroids_ missing; detector is not properly fitted.")
        if self.median_distances_ is None or self.mad_distances_ is None:
            raise RuntimeError(
                "distance statistics missing; detector is not properly fitted.")

        with tf.device(self.device):
            z = self._encode(x)  # (n_samples, latent_dim)

        # Distances to all centroids: (n_samples, n_classes)
        distances = np.linalg.norm(
            z[:, None, :] - self.centroids_[None, :, :],
            axis=2,
        ).astype(np.float64)

        # CADE anomaly score per class:
        # |distance - class_median_distance| / class_mad
        anomalies = np.abs(
            distances - self.median_distances_[None, :]
        ) / self.mad_distances_[None, :]

        closest_classes = np.argmin(distances, axis=1).astype(np.int32)
        min_scores = np.min(anomalies, axis=1).astype(np.float64)
        row_flags = (min_scores > self.mad_threshold)

        drift_count = int(row_flags.sum())
        drift_ratio = float(row_flags.mean()) if len(row_flags) else 0.0
        chunk_drift = (
            drift_count >= self.min_drift_count
            or drift_ratio >= self.min_drift_ratio
        )

        return CadeDetectionOutput(
            row_flags=row_flags,
            scores=min_scores,
            closest_classes=closest_classes,
            chunk_drift=chunk_drift,
        )

    def _build_encoder(self, weights_path: str) -> Model:
        """Build an encoder model and load trained weights."""
        ae = Autoencoder(self.dims)
        _ae_model, encoder = ae.build()
        encoder.load_weights(weights_path)
        return encoder

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode features into latent space."""
        if self.encoder_ is None:
            raise RuntimeError("encoder_ is not initialized.")
        z = self.encoder_.predict(x, verbose=str(0))
        return np.asarray(z, dtype=np.float32)

    def _require_fitted(self) -> None:
        """Raise if detect() is called before fit()."""
        if not self._is_fitted:
            raise RuntimeError(
                "CadeRuntimeDetector must be fitted before detect()."
            )
