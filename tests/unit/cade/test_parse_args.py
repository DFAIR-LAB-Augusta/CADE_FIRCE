import pytest

from cade.utils import SimConfig, parse_args


@pytest.mark.unit
@pytest.mark.fast
def test_parse_args_builds_simconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'sys.argv',
        [
            'prog',
            '--data',
            'drebin_new_7',
            '--classifier',
            'mlp',
            '--stage',
            'detect',
        ],
    )

    config = parse_args()

    assert isinstance(config, SimConfig)
    assert config.data == 'drebin_new_7'
    assert config.classifier == 'mlp'
    assert config.stage == 'detect'
    assert config.tree == 100


@pytest.mark.unit
@pytest.mark.fast
def test_parse_args_rejects_negative_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'sys.argv',
        [
            'prog',
            '--data',
            'drebin_new_7',
            '--tree',
            '-1',
        ],
    )

    with pytest.raises(ValueError, match='cannot be negative'):
        parse_args()
