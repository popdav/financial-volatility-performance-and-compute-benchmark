"""Package import tests."""


def test_package_import() -> None:
    """The top-level package is importable."""
    import volatility_benchmark

    assert volatility_benchmark.__doc__
