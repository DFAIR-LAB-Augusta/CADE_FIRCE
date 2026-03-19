import pytest


@pytest.mark.unit
@pytest.mark.fast
def test_unknown_attribute_raises_attribute_error() -> None:
    import cade

    with pytest.raises(AttributeError, match="has no attribute"):
        cade.does_not_exist
