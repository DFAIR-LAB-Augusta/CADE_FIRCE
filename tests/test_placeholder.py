def test_syntax_and_imports() -> None:
    """
    A simple smoke test to ensure the environment is set up correctly
    and the cade package can be imported.
    """
    import cade
    assert cade is not None
    assert cade.__name__ == 'cade'
