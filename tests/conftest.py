import pytest


@pytest.fixture(params=["the", "quick", "brown", "fox"])
def sample_args(request):
    return request.param
