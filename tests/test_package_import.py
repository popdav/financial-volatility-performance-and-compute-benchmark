"""Package import tests."""


def test_package_import() -> None:
    """The top-level package is importable."""
    import financial_volatility

    assert financial_volatility.__doc__
