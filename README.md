# CADE-FIRCE

Modernized CADE for concept drift detection in reusable Python workflows.

This package adapts the original CADE codebase from the USENIX Security 2021 paper into a library-oriented Python package for integration into other systems. In particular, it provides a runtime detector API that can be used inside streaming pipelines, evaluation frameworks, and drift monitoring components rather than only through the original experimental scripts.

## What this fork adds

This fork keeps the core CADE idea intact while updating the codebase for modern Python packaging and programmatic use.

Key changes include:

- packaging through `pyproject.toml`
- modern dependency management with `uv`
- a runtime-facing detector class, `CadeRuntimeDetector`
- a clearer fit and detect workflow for integration into other projects
- improved validation and runtime checks for detector configuration and input data shapes

The main entry point for integration is `cade.runtime.CadeRuntimeDetector`, which exposes a direct library API for training on reference data and then scoring incoming batches for drift.

## Background

CADE, short for Contrastive Autoencoder for Drift Detection and Explanation, was introduced in:

Limin Yang, Wenbo Guo, Qingying Hao, Arridhana Ciptadi, Ali Ahmadzadeh, Xinyu Xing, and Gang Wang.  
**CADE: Detecting and Explaining Concept Drift Samples for Security Applications.**  
USENIX Security 2021. 

The original work targets a specific form of concept drift in security settings, especially cases where new samples no longer align well with previously learned class structure. This fork focuses on making that detector easier to embed in downstream systems.

If you build on this package in a project or publication, please cite the original CADE paper.

```bibtex
@inproceedings{yang2021cade,
  title={$\{$CADE$\}$: Detecting and explaining concept drift samples for security applications},
  author={Yang, Limin and Guo, Wenbo and Hao, Qingying and Ciptadi, Arridhana and Ahmadzadeh, Ali and Xing, Xinyu and Wang, Gang},
  booktitle={30th USENIX Security Symposium (USENIX Security 21)},
  pages={2327--2344},
  year={2021}
}
```

## Installation

This project uses `uv` for environment and dependency management.

Clone the repository, then sync dependencies:

```bash
uv sync
```

For development dependencies:

```bash
uv sync --group dev
```

For scripting helpers:

```bash
uv sync --group scripting
```

To install all configured dependency groups:

```bash
uv sync --all-groups
```

## Development workflow

Common development commands:

```bash
uv lock
uv sync
uv run pytest -q
uv run pytest --cov=cade --cov-report=term-missing --cov-report=xml
uv run ruff format .
uv run ruff check .
uv run ruff check . --fix
uv build
uv run twine check dist/*
uv run deptry .
```

If you use the included `Makefile`, these commands are wrapped in targets such as `make sync`, `make test`, `make lint`, and `make build`.

## Runtime drift detection

The primary integration surface is `CadeRuntimeDetector`.

It is designed for the common pattern:

1. Fit the detector on known reference data
2. Encode incoming samples into CADE's latent space
3. Measure distance to learned class centroids
4. Compute robust anomaly scores using per-class median and MAD statistics
5. Flag row-level drift and summarize chunk-level drift status 

### Detector behavior

After `fit`, the detector stores:

* the observed training classes
* a label-to-index mapping
* latent centroids for each class
* per-class median distances
* per-class MAD-scaled distance statistics
* the trained encoder model  

During `detect(x)`, the detector:

* validates the input batch
* encodes each row into latent space
* computes distance from each encoded row to every class centroid
* converts those distances into anomaly scores
* marks a row as drifted if its minimum anomaly score exceeds `mad_threshold`
* marks the chunk as drifted if drift count or drift ratio exceeds configured thresholds 

This makes the detector useful both for per-row inspection and for higher-level monitoring decisions.

## Basic example

A minimal runtime example looks like this:

```python
from __future__ import annotations

import numpy as np

from cade.runtime import CadeRuntimeDetector

X_train = np.random.rand(1000, 32).astype(np.float32)
y_train = np.random.randint(0, 3, size=1000)

X_chunk = np.random.rand(128, 32).astype(np.float32)

detector = CadeRuntimeDetector(
    dims=[32, 64, 16],
    margin=10.0,
    mad_threshold=3.5,
    min_drift_ratio=0.05,
    min_drift_count=1,
    batch_size=64,
    epochs=25,
    lr=1e-3,
)

detector.fit(X_train, y_train)
out = detector.detect(X_chunk)

print("Chunk drift:", out.chunk_drift)
print("Drifted rows:", int(out.row_flags.sum()))
print("Scores shape:", out.scores.shape)
```

The returned object contains:

* `row_flags`: boolean drift flags for each row
* `scores`: per-row anomaly scores
* `closest_classes`: nearest learned class for each row
* `chunk_drift`: overall chunk-level drift decision  

## Integration example in a monitoring pipeline

One intended use of this package is wrapping the runtime detector inside a project-specific monitoring interface. For example, a monitoring component can fit CADE on training data and then translate CADE output into a framework-specific drift result object:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cade.runtime import CadeRuntimeDetector

from firce.drift_monitor.base import DriftDetectionResult

from .cade_config import CadeMonitorConfig

if TYPE_CHECKING:
    from firce.utils.config import SimulationConfig
    from firce.utils.perf_stats import PerformanceStats


class CadeDriftMonitor:
    def __init__(self, config: SimulationConfig) -> None:
        cade_cfg = CadeMonitorConfig(**config.monitor_kwargs)

        self._detector = CadeRuntimeDetector(
            dims=cade_cfg.dims,
            margin=cade_cfg.margin,
            mad_threshold=cade_cfg.mad_threshold,
            min_drift_ratio=cade_cfg.min_drift_ratio,
            min_drift_count=cade_cfg.min_drift_count,
            batch_size=cade_cfg.batch_size,
            epochs=cade_cfg.epochs,
            lr=cade_cfg.lr,
            cae_lambda_1=cade_cfg.cae_lambda_1,
            similar_ratio=cade_cfg.similar_ratio,
            display_interval=cade_cfg.display_interval,
            force_retrain=cade_cfg.force_retrain,
            weights_path=cade_cfg.weights_path,
            device=cade_cfg.device,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perf_stats: PerformanceStats | None = None,
    ) -> None:
        self._detector.fit(X_train, y_train)

    def detect(self, X: np.ndarray) -> DriftDetectionResult:
        out = self._detector.detect(X)
        row_flags = np.asarray(out.row_flags, dtype=bool).reshape(-1)
        scores = np.asarray(out.scores, dtype=float).reshape(-1)

        return DriftDetectionResult(
            row_flags=row_flags,
            chunk_drift=bool(row_flags.any()),
            scores=scores,
            metadata={
                "drift_count": int(row_flags.sum()),
                "chunk_size": int(len(row_flags)),
                "drift_ratio": float(row_flags.mean()) if len(row_flags) else 0.0,
            },
        )
```

This pattern is useful when CADE is one detector among several, or when a larger framework expects a standard drift-monitor interface.

## API notes

### `CadeRuntimeDetector(...)`

Important configuration parameters include:

* `dims`: network dimensions, including input and latent dimensions
* `margin`: contrastive margin used during training
* `mad_threshold`: row-level anomaly threshold
* `min_drift_ratio`: chunk-level ratio threshold
* `min_drift_count`: chunk-level count threshold
* `batch_size`: training batch size
* `epochs`: number of training epochs
* `lr`: optimizer learning rate
* `cae_lambda_1`: CAE training weight
* `similar_ratio`: ratio used for similar-pair construction
* `display_interval`: training log interval
* `weights_path`: optional saved weights path
* `device`: TensorFlow device string such as `/CPU:0`
* `force_retrain`: whether to discard an existing weights file before training 

### `fit(x_train, y_train)`

Fits the detector on labeled reference data. Input requirements:

* `x_train` must be a 2D array
* `y_train` must be a 1D array
* lengths must match
* `x_train.shape[1]` must equal `dims[0]`
* at least two classes must be present in training data 

### `detect(x)`

Scores a batch for drift. Input requirements:

* `x` must be a 2D array
* `x.shape[1]` must equal `dims[0]`
* the detector must already be fitted 

## When to use this package

This package is a good fit when you need:

* a drift detector that can be embedded directly into Python systems
* row-level drift flags and continuous anomaly scores
* chunk-level drift decisions based on configurable thresholds
* a detector that learns class structure in a latent space rather than relying only on raw-feature distances

It is especially useful in workflows where training data represents known classes and incoming data may contain new or shifted patterns that no longer fit those learned latent distributions.

## Project status

This package is a maintained downstream adaptation of the original CADE research code. It is intended to make CADE easier to use in modern Python environments and in integration-heavy projects such as evaluation pipelines, security tooling, and drift monitoring frameworks.

It should not be treated as the official upstream release.

## Attribution

This package is derived from the original CADE codebase and research work by:

* Limin Yang
* Wenbo Guo
* Qingying Hao
* Arridhana Ciptadi
* Ali Ahmadzadeh
* Xinyu Xing
* Gang Wang 

If you use this fork, please credit both:

1. the original CADE paper for the research contribution
2. this package or repository for packaging and runtime integration work, where appropriate

## License

This repository retains the original CADE licensing terms.

For ethical considerations, the code and data are covered by a modified BSD 3-Clause style license that restricts use to non-commercial scientific research and non-commercial education. Commercial use is prohibited. 

Please review the `LICENSE` file before redistribution or use.

## Repository links

* Upstream CADE research repository: [`whyisyoung/CADE`](https://github.com/whyisyoung/CADE)
* This repository: [`DFAIR-LAB-Augusta/CADE_FIRCE`](https://github.com/DFAIR-LAB-Augusta/CADE_FIRCE)
