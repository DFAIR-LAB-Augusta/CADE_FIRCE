from unittest.mock import Mock

import pytest

import cade.data as data
from cade.utils import SimConfig


@pytest.fixture
def simconfig_factory():  # noqa: ANN201
    def _factory(**overrides):  # noqa: ANN202
        base = {
            'data': 'drebin_case',
            'classifier': 'mlp',
            'stage': 'detect',
            'pure_ae': 0,
            'quiet': 1,
            'cae_hidden': '512-128-32',
            'cae_batch_size': 64,
            'cae_lr': 0.001,
            'cae_epochs': 250,
            'cae_lambda_1': 0.1,
            'similar_ratio': 0.25,
            'margin': 10.0,
            'display_interval': 10,
            'mad_threshold': 3.5,
            'exp_method': 'distance_mm1',
            'exp_lambda_1': 0.001,
            'mlp_retrain': 0,
            'mlp_hidden': '100-30',
            'mlp_batch_size': 32,
            'mlp_lr': 0.001,
            'mlp_epochs': 50,
            'mlp_dropout': 0.2,
            'newfamily_label': 7,
            'tree': 100,
            'rf_retrain': 0,
        }
        base.update(overrides)
        return SimConfig(**base)

    return _factory


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_dataset_calls_prepare_drebin_data(
    monkeypatch: pytest.MonkeyPatch,
    simconfig_factory,  # noqa: ANN001
) -> None:
    config = simconfig_factory(data='drebin_experiment', newfamily_label=5)
    mock = Mock()

    monkeypatch.setattr(data, 'prepare_drebin_data', mock)

    data.prepare_dataset(config)

    mock.assert_called_once_with('drebin_experiment', newfamily=5)


@pytest.mark.unit
@pytest.mark.fast
def test_prepare_dataset_rejects_non_drebin(simconfig_factory) -> None:  # noqa: ANN001
    config = simconfig_factory(data='IDS2018')

    with pytest.raises(ValueError, match='Should only be run in drebin'):
        data.prepare_dataset(config)
