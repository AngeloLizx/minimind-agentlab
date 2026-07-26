from calculator import add, divide, multiply, subtract


def test_arithmetic():
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2
    assert multiply(4, 3) == 12
    assert divide(8, 2) == 4
