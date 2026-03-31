import pytest

from cade.utils import get_model_dims


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    ('model_name', 'input_dim', 'hidden', 'output_dim', 'expected'),
    [
        pytest.param(
            'MLP',
            10,
            '100-30',
            3,
            [10, 100, 30, 3],
            id='mlp-two-hidden-layers',
        ),
        pytest.param(
            'Contrastive AE',
            20,
            '512-128-32',
            8,
            [20, 512, 128, 32, 8],
            id='cae-three-hidden-layers',
        ),
        pytest.param(
            'MLP',
            4,
            '16',
            2,
            [4, 16, 2],
            id='single-hidden-layer',
        ),
    ],
)
def test_get_model_dims_parses_hidden_layers(
    model_name: str,
    input_dim: int,
    hidden: str,
    output_dim: int,
    expected: list[int],
) -> None:
    assert get_model_dims(model_name, input_dim, hidden, output_dim) == expected


@pytest.mark.unit
@pytest.mark.fast
def test_get_model_dims_exits_on_invalid_hidden_spec() -> None:
    with pytest.raises(SystemExit) as exc_info:
        get_model_dims('MLP', 10, '100-bad-30', 2)

    assert exc_info.value.code == -1
