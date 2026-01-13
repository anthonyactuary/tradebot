from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any


async def paginate(
    fetch_page: Callable[[str | None], Any],
    *,
    cursor_field: str = "cursor",
    items_field: str = "markets",
) -> AsyncIterator[dict[str, Any]]:
    """Generic cursor-based pagination.

    `fetch_page(cursor)` must return a dict containing:
    - items_field: list
    - cursor_field: str | None
    """

    cursor: str | None = None
    while True:
        page = await fetch_page(cursor)
        for item in page.get(items_field) or []:
            yield item
        cursor = page.get(cursor_field)
        if not cursor:
            return
