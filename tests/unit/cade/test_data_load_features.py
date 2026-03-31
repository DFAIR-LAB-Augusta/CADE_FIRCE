from pathlib import Path

import numpy as np
import pytest

from cade.data import load_features


@pytest.mark.unit
def test_load_features_maps_known_and_unknown_labels_for_drebin(
    tmp_path: Path,
) -> None:
    dataset_name = 'drebin_sample'
    npz_path = tmp_path / f'{dataset_name}.npz'

    x_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y_train = np.array([10, 20])
    x_test = np.array([[1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    y_test = np.array([20, 999])

    np.savez_compressed(
        npz_path,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
    )

    x_train_out, y_train_out, x_test_out, y_test_out = load_features(
        dataset=dataset_name,
        newfamily=123,
        folder=str(tmp_path),
    )

    assert np.array_equal(x_train_out, x_train)
    assert np.array_equal(x_test_out, x_test)

    # Training labels are normalized to contiguous ints
    assert np.array_equal(y_train_out, np.array([0, 1], dtype=np.int32))

    # Known test label maps to encoded label; unknown becomes Drebin persistent label 7
    assert np.array_equal(y_test_out, np.array([1, 7], dtype=np.int32))
    assert y_train_out.dtype == np.int32
    assert y_test_out.dtype == np.int32


@pytest.mark.unit
@pytest.mark.fast
def test_load_features_exits_for_unsupported_dataset(tmp_path: Path) -> None:
    dataset_name = 'unsupported_dataset'
    np.savez_compressed(
        tmp_path / f'{dataset_name}.npz',
        X_train=np.array([[1.0]], dtype=np.float32),
        y_train=np.array([1]),
        X_test=np.array([[1.0]], dtype=np.float32),
        y_test=np.array([1]),
    )

    with pytest.raises(SystemExit) as exc_info:
        load_features(dataset=dataset_name, newfamily=99, folder=str(tmp_path))

    assert exc_info.value.code == -4
