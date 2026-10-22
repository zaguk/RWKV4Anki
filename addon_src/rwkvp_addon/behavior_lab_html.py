# ruff: noqa: E501
from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .behavior_lab import (
    BehaviorLabExperiment,
    BehaviorLabResult,
    cohort_delta_summary,
    observation_deltas,
)
from .svelte_ui import render_svelte_document

_GRAPH_WIDTH = 920
_GRAPH_HEIGHT = 330
_GRAPH_LEFT = 58
_GRAPH_RIGHT = 20
_GRAPH_TOP = 18
_GRAPH_BOTTOM = 42


def render_behavior_lab_editor(
    experiment: BehaviorLabExperiment,
    *,
    saved_experiments: Mapping[str, BehaviorLabExperiment] | None = None,
    message: str = "",
    generation: int = 1,
    is_dark: bool = False,
) -> str:
    return render_svelte_document(
        "behavior-lab-editor",
        {
            "experiment": experiment.to_dict(),
            "saved": {
                str(name): value.to_dict() for name, value in (saved_experiments or {}).items()
            },
            "message": message,
        },
        title="RWKV Behavior Lab",
        title_id="behavior-lab-title",
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-behavior-lab rwkv-behavior-lab--editor",
    )


def render_behavior_lab_results(
    result: BehaviorLabResult,
    *,
    generation: int = 1,
    is_dark: bool = False,
) -> str:
    sections = [
        (
            "Focal observations",
            ("Scenario", "Observation", "Simulation time", "Retrievability", "Δ control"),
            _focal_observation_rows(result),
            None,
        ),
        (
            "Post-answer curves",
            ("Scenario", "Event", "At answer", "1 min", "1 hour", "1 day", "7 days", "30 days"),
            _curve_rows(result),
            None,
        ),
        (
            "Card and sibling impact",
            ("Card", "Relationship", *(scenario.name for scenario in result.scenarios)),
            _impact_rows(result),
            "Final current-state observation in each scenario, compared with the control observation at the same ordinal.",
        ),
        (
            "Cohort changes",
            ("Scenario", "Observation", "Cards", "Mean Δ", "Median Δ", "Minimum Δ", "Maximum Δ"),
            _cohort_rows(result),
            None,
        ),
        (
            "Processed-event ledger",
            (
                "Scenario",
                "Event",
                "Card",
                "Rating",
                "Context",
                "Answer time",
                "Elapsed",
                "P before answer",
            ),
            _event_rows(result),
            None,
        ),
    ]
    return render_svelte_document(
        "behavior-lab-results",
        {
            "title": result.experiment_name,
            "model_id": result.model_id,
            "fingerprint": result.checkpoint_fingerprint[:16],
            "focal_card_id": result.focal_card_id,
            "graph_svg": _result_graph(result),
            "sections": [
                {"title": title, "headers": headers, "rows": rows, "intro": intro}
                for title, headers, rows, intro in sections
            ],
            "report_json": json.dumps(
                behavior_lab_result_to_dict(result), indent=2, sort_keys=True
            ),
        },
        title=result.experiment_name,
        title_id="behavior-lab-title",
        generation=generation,
        is_dark=is_dark,
        width="wide",
        root_extra_classes="rwkv-behavior-lab rwkv-behavior-lab--results",
    )


def behavior_lab_result_to_dict(result: BehaviorLabResult) -> dict[str, Any]:
    return asdict(result)


def _result_graph(result: BehaviorLabResult) -> str:
    plot_width = _GRAPH_WIDTH - _GRAPH_LEFT - _GRAPH_RIGHT
    plot_height = _GRAPH_HEIGHT - _GRAPH_TOP - _GRAPH_BOTTOM
    series: list[tuple[str, str, bool, list[tuple[float, float]]]] = []
    maximum_elapsed = 1.0
    for scenario in result.scenarios:
        color = scenario.color or "#1976d2"
        observation_points: list[tuple[float, float]] = []
        for observation in scenario.observations:
            prediction = observation.prediction_for(result.focal_card_id)
            if prediction is None or not math.isfinite(prediction):
                continue
            elapsed = max(0.0, observation.timestamp_seconds - result.baseline_timestamp_seconds)
            maximum_elapsed = max(maximum_elapsed, elapsed)
            observation_points.append((elapsed, prediction))
        series.append((scenario.name, color, False, observation_points))
        for review in scenario.reviews:
            if review.card_id != result.focal_card_id or not review.curve_points:
                continue
            offset = review.timestamp_seconds - result.baseline_timestamp_seconds
            points = [
                (max(0.0, offset + point.elapsed_seconds), point.probability)
                for point in review.curve_points
            ]
            maximum_elapsed = max(maximum_elapsed, *(value[0] for value in points))
            series.append((f"{scenario.name} curve", color, True, points))

    x_max = math.log1p(maximum_elapsed)

    def x_scale(elapsed: float) -> float:
        return _GRAPH_LEFT + plot_width * math.log1p(max(0.0, elapsed)) / x_max

    def y_scale(probability: float) -> float:
        return _GRAPH_TOP + plot_height * (1.0 - min(1.0, max(0.0, probability)))

    lines: list[str] = []
    legend: list[str] = []
    seen_legend: set[tuple[str, bool]] = set()
    for name, color, dashed, points in series:
        if not points:
            continue
        coords = " ".join(f"{x_scale(x):.2f},{y_scale(y):.2f}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        if len(points) == 1:
            x, y = points[0]
            lines.append(
                f'<circle cx="{x_scale(x):.2f}" cy="{y_scale(y):.2f}" r="4" fill="{html.escape(color)}" />'
            )
        else:
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="{html.escape(color)}" '
                f'stroke-width="2.2"{dash} />'
            )
        key = (name, dashed)
        if key not in seen_legend:
            seen_legend.add(key)
            legend.append(
                f'<span><i style="background:{html.escape(color)}"></i>{html.escape(name)}</span>'
            )
    grid: list[str] = []
    for percent in (0, 25, 50, 75, 100):
        y = y_scale(percent / 100)
        grid.append(
            f'<line x1="{_GRAPH_LEFT}" y1="{y:.2f}" x2="{_GRAPH_WIDTH - _GRAPH_RIGHT}" y2="{y:.2f}" class="grid" />'
            f'<text x="{_GRAPH_LEFT - 8}" y="{y + 4:.2f}" text-anchor="end">{percent}%</text>'
        )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        transformed = x_max * fraction
        elapsed = math.expm1(transformed)
        x = _GRAPH_LEFT + plot_width * fraction
        grid.append(
            f'<line x1="{x:.2f}" y1="{_GRAPH_TOP}" x2="{x:.2f}" y2="{_GRAPH_HEIGHT - _GRAPH_BOTTOM}" class="grid" />'
            f'<text x="{x:.2f}" y="{_GRAPH_HEIGHT - 17}" text-anchor="middle">{html.escape(_format_duration(elapsed))}</text>'
        )
    return f"""
<svg class="timeline-chart" viewBox="0 0 {_GRAPH_WIDTH} {_GRAPH_HEIGHT}" role="img"
 aria-label="Retrievability over simulated time"><g class="grid-lines">{"".join(grid)}</g>
 <g class="series">{"".join(lines)}</g></svg><div class="legend">{"".join(legend)}</div>
"""


def _focal_observation_rows(result: BehaviorLabResult) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for scenario in result.scenarios:
        for observation in scenario.observations:
            prediction = observation.prediction_for(result.focal_card_id)
            if prediction is None:
                continue
            deltas = dict(observation_deltas(result, scenario, observation))
            rows.append(
                (
                    scenario.name,
                    observation.label,
                    _format_duration(
                        observation.timestamp_seconds - result.baseline_timestamp_seconds
                    ),
                    _format_probability(prediction),
                    _delta_cell(deltas.get(result.focal_card_id)),
                )
            )
    return rows


def _curve_rows(result: BehaviorLabResult) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    targets = (60.0, 3600.0, 86_400.0, 604_800.0, 2_592_000.0)
    for scenario in result.scenarios:
        for review in scenario.reviews:
            points = {
                round(point.elapsed_seconds): point.probability for point in review.curve_points
            }
            values = [_nearest_curve_probability(points, target) for target in targets]
            rows.append(
                (
                    scenario.name,
                    review.label,
                    _format_probability(review.prediction_before_answer),
                    *(_format_probability(value) if value is not None else "—" for value in values),
                )
            )
    return rows


def _impact_rows(result: BehaviorLabResult) -> list[tuple[Any, ...]]:
    cards = {card.card_id: card for card in result.cards}
    final_observations = [
        scenario.observations[-1] if scenario.observations else None
        for scenario in result.scenarios
    ]
    rows: list[tuple[Any, ...]] = []
    for card_id in sorted(cards)[:200]:
        cells: list[Any] = []
        for scenario, observation in zip(result.scenarios, final_observations, strict=True):
            if observation is None:
                cells.append("—")
                continue
            delta = dict(observation_deltas(result, scenario, observation)).get(card_id)
            cells.append(_delta_cell(delta, heat=True))
        rows.append(
            (
                str(card_id),
                cards[card_id].relation,
                *cells,
            )
        )
    if len(cards) > 200:
        rows.append((f"… {len(cards) - 200:,} more", "", *("" for _ in result.scenarios)))
    return rows


def _cohort_rows(result: BehaviorLabResult) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for scenario in result.scenarios:
        for observation in scenario.observations:
            summary = cohort_delta_summary(result, scenario, observation)
            if summary is None:
                continue
            rows.append(
                (
                    scenario.name,
                    observation.label,
                    f"{int(summary['count']):,}",
                    _format_delta(float(summary["mean"])),
                    _format_delta(float(summary["median"])),
                    _format_delta(float(summary["minimum"])),
                    _format_delta(float(summary["maximum"])),
                )
            )
    return rows


def _event_rows(result: BehaviorLabResult) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for scenario in result.scenarios:
        for review in scenario.reviews:
            rows.append(
                (
                    scenario.name,
                    review.label,
                    str(review.card_id),
                    str(review.rating),
                    review.review_context,
                    f"{review.duration_seconds:g}s",
                    _format_duration(review.elapsed_seconds),
                    _format_probability(review.prediction_before_answer),
                )
            )
    return rows


def _nearest_curve_probability(
    points: Mapping[int, float],
    target: float,
) -> float | None:
    if not points:
        return None
    key = min(points, key=lambda value: abs(value - target))
    return points[key] if abs(key - target) <= max(1.0, target * 0.05) else None


def _format_probability(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{value * 100:.2f}%"


def _format_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _delta_cell(value: float | None, *, heat: bool = False) -> str | dict[str, str]:
    if value is None:
        return "—"
    rendered = _format_delta(value)
    if not heat:
        return rendered
    strength = min(1.0, abs(value) / 0.10)
    if value >= 0:
        color = f"rgba(33, 150, 243, {0.08 + strength * 0.34:.3f})"
    else:
        color = f"rgba(255, 152, 0, {0.08 + strength * 0.34:.3f})"
    return {"text": rendered, "background": color}


def _format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    if value < 86_400:
        return f"{value / 3600:.1f}h"
    if value < 31_536_000:
        return f"{value / 86_400:.1f}d"
    return f"{value / 31_536_000:.1f}y"
