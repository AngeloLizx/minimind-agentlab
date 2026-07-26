from text_utils import slugify


def test_empty_slug_uses_stable_fallback():
    assert slugify("   ") == "untitled"
