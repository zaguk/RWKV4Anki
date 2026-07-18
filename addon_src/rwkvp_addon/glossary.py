from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    definition: str
    phrases: tuple[str, ...]


# Keep user-facing definitions here so Settings and Guided Setup explain the
# same term in the same way. Phrases are deliberately specific: a dotted term
# should answer a likely question, not decorate every occurrence of a common
# word such as "model" or "state".
GLOSSARY_TERMS: Mapping[str, GlossaryEntry] = {
    "state-building": GlossaryEntry(
        "Processing your Anki review history to create or update RWKV's memory of it.",
        ("state building", "state-building"),
    ),
    "review-history": GlossaryEntry(
        "Anki's record of your past answers, including when and how you rated each card.",
        ("review history",),
    ),
    "rwkv-state": GlossaryEntry(
        "RWKV's memory of your review history, used to make future predictions.",
        ("RWKV state",),
    ),
    "checkpoint": GlossaryEntry(
        "RWKV state saved to disk so the full review history does not need to be "
        "processed for every operation.",
        ("checkpoints", "checkpoint"),
    ),
    "retrievability": GlossaryEntry(
        "The estimated probability that you would recall a card at a particular time.",
        ("retrievability",),
    ),
    "rwkv-immediate": GlossaryEntry(
        "RWKV's estimate of how likely you are to recall a card if Anki showed it now.",
        ("RWKV Immediate",),
    ),
    "forgetting-curve": GlossaryEntry(
        "A prediction of how a card's retrievability changes over time after a review.",
        (
            "RWKV Forgetting Curves",
            "RWKV Forgetting Curve",
            "forgetting curves",
            "forgetting curve",
        ),
    ),
    "live-session": GlossaryEntry(
        "An RWKV study mode that updates predictions after every answer and uses them "
        "to choose what to show next.",
        ("Live Sessions", "Live Session"),
    ),
    "fsrs": GlossaryEntry(
        "Anki's built-in memory model and scheduler. RWKV4Anki can compare its "
        "predictions with FSRS.",
        ("FSRS",),
    ),
    "cpu-fast": GlossaryEntry(
        "RWKV's optimized CPU mode for everyday state building and card prediction.",
        ("CPU Fast",),
    ),
    "gpu": GlossaryEntry(
        "A compatible graphics processor that can accelerate large groups of reviews "
        "or predictions. RWKV tests support separately for each task.",
        ("GPU",),
    ),
    "stability": GlossaryEntry(
        "An estimate of how long a card's memory lasts. Greater stability generally "
        "means its retrievability falls more slowly.",
        ("stability",),
    ),
    "log-loss": GlossaryEntry(
        "An accuracy score that penalizes confident wrong predictions. Lower is better.",
        ("LogLoss", "log-loss", "log loss"),
    ),
    "rmse-bins": GlossaryEntry(
        "A calibration score comparing predicted and observed recall rates in groups "
        "of reviews. Lower is better.",
        ("RMSE(bins)",),
    ),
    "rwkv-model": GlossaryEntry(
        "A trained version of RWKV. Bundled models learned from different subsets of "
        "the benchmark review data.",
        ("Underlying Model", "RWKV models", "RWKV model"),
    ),
    "deleted-history": GlossaryEntry(
        "Past reviews for cards that are no longer in your collection. They can provide "
        "context for current-card predictions but cannot be scored themselves.",
        ("deleted-card history", "deleted history"),
    ),
    "desired-retention": GlossaryEntry(
        "The target probability of recalling a card when it is reviewed.",
        ("desired retention",),
    ),
    "oracle": GlossaryEntry(
        "The slower reference implementation, mainly useful for comparison and testing.",
        ("Oracle",),
    ),
    "batch-size": GlossaryEntry(
        "The number of card predictions RWKV processes together in one operation.",
        ("batch sizes", "batch size"),
    ),
    "rwkv-current-version": GlossaryEntry(
        "The bundled RWKV-P model was trained with review timing measured at the end "
        "of each answer, while Live Session predicts before the next answer begins. "
        "Rechecking the same card immediately can therefore give the model an "
        "unreliable near-zero elapsed time.",
        ("current version of RWKV",),
    ),
}

_GLOSSARY_PHRASE_KEYS = {
    phrase.casefold(): key
    for key, entry in GLOSSARY_TERMS.items()
    for phrase in entry.phrases
}
_GLOSSARY_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(phrase)
        for phrase in sorted(_GLOSSARY_PHRASE_KEYS, key=len, reverse=True)
    )
    + r")(?!\w)",
    re.IGNORECASE,
)


class GlossaryRenderer:
    """Escape prose and annotate the first occurrence of each known term."""

    def __init__(self, namespace: str) -> None:
        self._namespace = re.sub(
            r"[^a-z0-9]+", "-", namespace.casefold()
        ).strip("-")
        self._rendered_keys: set[str] = set()

    def render(self, text: str) -> str:
        rendered: list[str] = []
        cursor = 0
        for match in _GLOSSARY_PATTERN.finditer(text):
            key = _GLOSSARY_PHRASE_KEYS[match.group(0).casefold()]
            if key in self._rendered_keys:
                continue
            entry = GLOSSARY_TERMS[key]
            tooltip_id = f"glossary-{self._namespace}-{key}"
            rendered.append(html.escape(text[cursor : match.start()]))
            rendered.append(
                f'<dfn class="glossary-term" tabindex="0" '
                f'data-glossary-key="{html.escape(key, quote=True)}" '
                f'aria-describedby="{html.escape(tooltip_id, quote=True)}" '
                'aria-expanded="false">'
                f"{html.escape(match.group(0))}"
                f'<span class="glossary-popover" id="{html.escape(tooltip_id, quote=True)}" '
                f'role="tooltip">{html.escape(entry.definition)}</span>'
                "</dfn>"
            )
            self._rendered_keys.add(key)
            cursor = match.end()
        rendered.append(html.escape(text[cursor:]))
        return "".join(rendered)
