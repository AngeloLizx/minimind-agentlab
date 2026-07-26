def slugify(value):
    return value.strip().lower().replace(" ", "-")


def title_case(value):
    return " ".join(word.capitalize() for word in value.split())


def truncate(text, limit):
    return text[:limit] + "..."
