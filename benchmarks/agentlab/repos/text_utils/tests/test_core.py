from text_utils import slugify, title_case, truncate


def test_slugify_words():
    assert slugify("Hello Agent Lab") == "hello-agent-lab"


def test_title_case():
    assert title_case("mini mind") == "Mini Mind"


def test_truncate_long_text():
    assert truncate("abcdef", 3) == "abc..."
