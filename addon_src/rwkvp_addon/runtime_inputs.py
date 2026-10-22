"""Generation-aware timestamp context without importing the native runtime."""

from .model_generation import normalize_model_generation
from .review_rows import ReviewData
from .v5o_review_rows import V5oInputContext

RuntimeInputContext = V5oInputContext


def input_context_for_generation(
    review_data: ReviewData, generation: object
) -> RuntimeInputContext | None:
    selected = normalize_model_generation(generation)
    if selected == "v5o":
        return V5oInputContext.from_review_data(review_data)
    return None
