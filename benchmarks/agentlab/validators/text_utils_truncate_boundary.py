from text_utils import truncate


def test_truncate_does_not_expand_short_or_equal_text():
    assert truncate("abc", 3) == "abc"
    assert truncate("ab", 3) == "ab"
