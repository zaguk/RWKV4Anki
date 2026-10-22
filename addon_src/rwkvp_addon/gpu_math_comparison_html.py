from __future__ import annotations

from .gpu_math_comparison import GPUMathComparisonResult
from .model_generation import model_generation_label
from .report_payload import report_document, value_cell


def render_gpu_math_comparison_html(
    result: GPUMathComparisonResult,
    *,
    is_dark: bool = False,
    generation: int = 1,
) -> str:
    count = result.differences.count
    differences = result.differences
    difference_rows = [
        {"cells": [value_cell(label, header=True, css_class=""), value_cell(f"{value:.6f} pp")]}
        for label, value in (
            ("Mean absolute difference", differences.mean_pp),
            ("Median absolute difference (P50)", differences.median_pp),
            ("90th percentile (P90)", differences.p90_pp),
            ("99th percentile (P99)", differences.p99_pp),
            ("Maximum absolute difference", differences.max_pp),
        )
    ]
    return report_document(
        title="FP16 GPU Predictions · Compare Predictions",
        subtitle=(f"{model_generation_label(result.model_generation)} · {result.model_id} · "
                  f"{count:,} paired cards · FP16 state in both modes"),
        sections=[
            {
                "heading": "Absolute prediction differences · On minus Off",
                "wrap": True,
                "table": {
                    "label": "Paired absolute percentage-point differences",
                    # Two columns fit narrow windows using the shared base table;
                    # don't inherit the wide multi-metric comparison minimum.
                    "cssClass": "",
                    "headers": [{"text": "Statistic"}, {"text": "Absolute difference"}],
                    "rows": difference_rows,
                },
            },
        ],
        notes=(
            f"One disposable FP16 card/note state was built from {result.processed_review_count:,} "
            "chronological reviews, then restored unchanged for each math mode. "
            "The same current state-backed cards, query timestamps and inputs were used. "
            "This comparison uses FP16 storage regardless of your storage setting.",
            "Off = GPU (FP32 math); On = GPU (Full FP16). No performance test was run. "
            "Use General > Performance > Prediction Mode > Compare Modes for speed. "
            "No CPU fallback was allowed. GPU kernels: "
            + "; ".join(f"{m.math.upper()}: {m.kernel}" for m in result.measurements) + ".",
            "Differences are |FP16 math − FP32 math| × 100, in percentage points (pp), "
            "not relative percent error. For example, 70.0% vs 70.1% differs by 0.1 pp. "
            "Percentiles use linear interpolation. These describe current forecasts, "
            "not LogLoss, future accuracy or a guarantee for other models/GPUs. "
            "Small changes can affect card ordering near a retention threshold.",
            "Temporary state was discarded. Your settings, saved checkpoint and Anki data "
            "are unchanged.",
        ),
        card_label="FP16 GPU arithmetic comparison results",
        is_dark=is_dark,
        generation=generation,
        root_classes="rwkv-comparison",
    )
