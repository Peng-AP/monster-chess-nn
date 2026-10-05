"""Shared pytest configuration."""


def pytest_configure(config):
    # CI (a fresh checkout) deselects these with -m "not local_artifacts"; on the
    # training machine they run like any other test.
    config.addinivalue_line(
        "markers",
        "local_artifacts: reads files that exist only on the training machine (model weights, "
        "benchmark receipts and journals, the uncommitted play.ipynb)")
