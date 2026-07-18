from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationDateRange:
    """An inclusive local-calendar range expressed as revlog-id boundaries."""

    start_date: dt.date
    end_date: dt.date
    start_review_id: int
    end_review_id_exclusive: int

    @classmethod
    def from_local_dates(
        cls,
        start_date: dt.date,
        end_date: dt.date,
        *,
        timezone: dt.tzinfo | None = None,
    ) -> EvaluationDateRange:
        if end_date < start_date:
            raise ValueError("The evaluation end date must not be before the start date.")

        next_date = end_date + dt.timedelta(days=1)
        if timezone is None:
            start_datetime = dt.datetime.combine(start_date, dt.time.min).astimezone()
            end_datetime = dt.datetime.combine(next_date, dt.time.min).astimezone()
        else:
            start_datetime = dt.datetime.combine(start_date, dt.time.min, tzinfo=timezone)
            end_datetime = dt.datetime.combine(next_date, dt.time.min, tzinfo=timezone)

        return cls(
            start_date=start_date,
            end_date=end_date,
            start_review_id=int(start_datetime.timestamp() * 1000),
            end_review_id_exclusive=int(end_datetime.timestamp() * 1000),
        )

    @property
    def cache_key(self) -> tuple[str, str]:
        return self.start_date.isoformat(), self.end_date.isoformat()

    def contains_review_id(self, review_id: int) -> bool:
        normalized = int(review_id)
        return self.start_review_id <= normalized < self.end_review_id_exclusive


def review_date_bounds_for_ids(
    first_review_id: int | None,
    last_review_id: int | None,
    *,
    fallback_date: dt.date | None = None,
    timezone: dt.tzinfo | None = None,
) -> tuple[dt.date, dt.date]:
    """Return local dates spanning the collection's first and last revlogs."""

    fallback = fallback_date or dt.date.today()
    if first_review_id is None or last_review_id is None:
        return fallback, fallback

    if timezone is None:
        first = dt.datetime.fromtimestamp(int(first_review_id) / 1000).date()
        last = dt.datetime.fromtimestamp(int(last_review_id) / 1000).date()
    else:
        first = dt.datetime.fromtimestamp(int(first_review_id) / 1000, timezone).date()
        last = dt.datetime.fromtimestamp(int(last_review_id) / 1000, timezone).date()
    if last < first:
        first, last = last, first
    return first, last


def filter_target_review_ids_by_date_range(
    target_review_ids_by_scope: Mapping[str, set[int]],
    date_range: EvaluationDateRange,
) -> dict[str, set[int]]:
    return {
        scope_key: {
            int(review_id)
            for review_id in review_ids
            if date_range.contains_review_id(int(review_id))
        }
        for scope_key, review_ids in target_review_ids_by_scope.items()
    }
