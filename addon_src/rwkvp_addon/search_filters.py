from __future__ import annotations


def required_search_with_filter(base_search: str, extra_search: str | None = None) -> str:
    """Return a search that requires both the base query and the user filter.

    The extra search is grouped so a user-provided OR clause cannot escape the
    source deck / active-card constraints.
    """
    base = str(base_search or "").strip()
    extra = str(extra_search or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{base} ({extra})"


def candidate_filter_count_label(count: int) -> str:
    suffix = "card" if int(count) == 1 else "cards"
    return f"{int(count)} {suffix}"
