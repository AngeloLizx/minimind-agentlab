import pytest

from calculator import divide


def test_divide_zero_has_domain_error():
    with pytest.raises(ValueError, match="zero"):
        divide(10, 0)
