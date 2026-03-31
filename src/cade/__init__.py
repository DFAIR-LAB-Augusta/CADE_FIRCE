"""CADE package."""

from __future__ import annotations

__all__ = ['CadeDetectionOutput', 'CadeRuntimeDetector']


def __getattr__(name: str) -> None:
    if name in {'CadeDetectionOutput', 'CadeRuntimeDetector'}:
        try:
            from cade.runtime import (
                CadeDetectionOutput,
                CadeRuntimeDetector,
            )
        except ModuleNotFoundError as exc:
            if exc.name in {'tensorflow', 'keras'}:
                raise ModuleNotFoundError(
                    "CADE runtime requires the optional 'tf' dependencies. "
                    'Install them with: uv sync --extra tf --group dev'
                ) from exc
            raise

        return {
            'CadeDetectionOutput': CadeDetectionOutput,
            'CadeRuntimeDetector': CadeRuntimeDetector,
        }[name]

    raise AttributeError(f"module 'cade' has no attribute {name!r}")
