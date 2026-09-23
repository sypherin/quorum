import pytest

from quorum import serve


@pytest.fixture(autouse=True)
def _fresh_judgment_cache():
    """The serve cache is process-global; a hit left by one test would hide
    the upstream call another test asserts on."""
    serve._cache.clear()
    yield
    serve._cache.clear()
