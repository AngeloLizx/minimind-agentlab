import pytest

from calculator import add


def test_add_rejects_boolean_operands():
    with pytest.raises(TypeError, match="boolean"):
        add(True, 2)
    with pytest.raises(TypeError, match="boolean"):
        add(2, False)
