"""Typed Guided Setup blocks; interactive markup is owned by Svelte."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any


class _ProseValidator(HTMLParser):
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"button", "input", "select", "textarea", "form", "script", "iframe"}:
            raise ValueError(f"Guided Setup prose cannot contain interactive <{tag}> markup")
        if any(name.casefold().startswith("on") for name, _value in attrs):
            raise ValueError("Guided Setup prose cannot contain event handlers")


@dataclass(frozen=True)
class SetupContent:
    """Compose trusted prose with separately typed controls, without HTML injection."""

    blocks: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def prose(cls, html: str) -> SetupContent:
        _ProseValidator().feed(html)
        return cls(({"kind": "prose", "html": html},)) if html else cls()

    @classmethod
    def coerce(cls, value: SetupContent | str) -> SetupContent:
        return value if isinstance(value, cls) else cls.prose(value)

    @classmethod
    def join(cls, values: Iterable[SetupContent | str]) -> SetupContent:
        result = cls()
        for value in values:
            result += value
        return result

    @classmethod
    def results(cls, rows: Iterable[Mapping[str, Any]]) -> SetupContent:
        return cls(({"kind": "results", "rows": list(rows)},))

    @classmethod
    def report(cls, payload: Mapping[str, Any]) -> SetupContent:
        return cls(({
            "kind": "report", "subtitle": payload["subtitle"],
            "sections": payload["sections"], "notes": payload["notes"],
            "label": payload["card_label"],
        },))

    @classmethod
    def faq_link(cls) -> SetupContent:
        return cls(({"kind": "faq-link"},))

    def __add__(self, other: SetupContent | str) -> SetupContent:
        return SetupContent(self.blocks + self.coerce(other).blocks)

    def __radd__(self, other: SetupContent | str) -> SetupContent:
        return SetupContent(self.coerce(other).blocks + self.blocks)

    def payload(self) -> list[Mapping[str, Any]]:
        return list(self.blocks)
