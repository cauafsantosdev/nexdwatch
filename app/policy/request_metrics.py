"""Collects request-local timing and counters without affecting policy output."""

import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

CURRENT_CATEGORY_STAGE: ContextVar[str | None] = ContextVar(
    "current_category_stage", default=None
)


@dataclass(slots=True)
class CategoryRequestProfile:
    """Request-local measurements excluded from recommendation domain output."""

    stage_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    stage_calls: Counter[str] = field(default_factory=Counter)
    counters: dict[str, float | str | None] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one potentially nested stage with a query-purpose context."""
        token = CURRENT_CATEGORY_STAGE.set(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] += time.perf_counter() - started
            self.stage_calls[name] += 1
            CURRENT_CATEGORY_STAGE.reset(token)

    def count(self, name: str, value: float | str | None) -> None:
        """Record a deterministic request-shape diagnostic."""
        self.counters[name] = value

    def milliseconds(self) -> dict[str, float]:
        """Return stable millisecond values for research reports."""
        return {
            name: seconds * 1000 for name, seconds in sorted(self.stage_seconds.items())
        }


def request_stage(
    profile: CategoryRequestProfile | None, name: str
) -> AbstractContextManager[None]:
    """Return a measured stage or a zero-cost-compatible null context."""
    return profile.stage(name) if profile is not None else nullcontext()
