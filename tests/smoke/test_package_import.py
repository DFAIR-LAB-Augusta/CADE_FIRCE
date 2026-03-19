import pytest


@pytest.mark.smoke_sanity
def test_import_cade_package() -> None:
    import cade

    assert cade is not None
    assert cade.__name__ == "cade"
