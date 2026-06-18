"""Load and validate CEO / CIS survey items.

Each row of a survey CSV (see data/schema.md) becomes a :class:`SurveyItem`.
The loader enforces that the population distribution matches the option set and
sums to 1, and it carries the ``source_status`` flag through so illustrative
``example`` rows are never silently treated as real measurements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "item_id", "source", "survey_wave", "population", "topic",
    "question_ca", "question_es", "options", "options_es", "pop_dist",
    "dimension", "source_status", "notes",
}

DIST_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SurveyItem:
    item_id: str
    source: str            # CEO | CIS
    survey_wave: str
    population: str         # catalonia | spain
    topic: str
    question: dict          # {"ca": str, "es": str}
    options: dict           # {"ca": [str, ...], "es": [str, ...]}
    pop_dist: np.ndarray    # proportions aligned to options order
    dimension: str
    source_status: str      # verified | example
    notes: str

    @property
    def n_options(self) -> int:
        return len(self.options["ca"])

    @property
    def ordinal_codes(self) -> list:
        """Scale position of each option, or None for non-ordinal (NS/NC) ones.

        Read from the leading integer of the option label, e.g. ``"0 (Cap
        confiança)"`` -> 0, ``"No ho sap"`` -> None. Language-independent (the
        numeric prefix is the same in ca/es).
        """
        out = []
        for o in self.options["ca"]:
            m = re.match(r"\s*(\d+)", o)
            out.append(float(m.group(1)) if m else None)
        return out

    @property
    def is_ordinal(self) -> bool:
        """True when the substantive options form a numeric scale (>=3 coded
        points), e.g. the 0--10 trust and left--right ideology scales. Nominal
        items (independence, identity, monarchy, state model, the verbal economy
        Likert) have no numeric codes and stay JSD-only."""
        return sum(c is not None for c in self.ordinal_codes) >= 3

    def question_for(self, lang: str) -> str:
        return self.question[lang]

    def options_for(self, lang: str) -> list[str]:
        return self.options[lang]


def _split(field: str) -> list[str]:
    return [part.strip() for part in str(field).split("|")]


def load_items(path: str | Path) -> list[SurveyItem]:
    """Load one survey CSV into validated SurveyItem objects."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    items: list[SurveyItem] = []
    for _, r in df.iterrows():
        opts_ca = _split(r["options"])
        opts_es = _split(r["options_es"])
        dist = np.array([float(x) for x in _split(r["pop_dist"])], dtype=np.float64)

        if not (len(opts_ca) == len(opts_es) == len(dist)):
            raise ValueError(
                f"{r['item_id']}: options/options_es/pop_dist length mismatch "
                f"({len(opts_ca)}, {len(opts_es)}, {len(dist)})"
            )
        if abs(dist.sum() - 1.0) > DIST_TOLERANCE:
            raise ValueError(f"{r['item_id']}: pop_dist sums to {dist.sum():.4f}, expected 1.0")

        items.append(SurveyItem(
            item_id=str(r["item_id"]),
            source=str(r["source"]),
            survey_wave=str(r["survey_wave"]),
            population=str(r["population"]),
            topic=str(r["topic"]),
            question={"ca": str(r["question_ca"]), "es": str(r["question_es"])},
            options={"ca": opts_ca, "es": opts_es},
            pop_dist=dist,
            dimension=str(r["dimension"]),
            source_status=str(r["source_status"]),
            notes=str(r["notes"]),
        ))
    return items


def load_datasets(paths: list[str | Path]) -> list[SurveyItem]:
    """Load and concatenate several survey CSVs, checking item_id uniqueness."""
    items: list[SurveyItem] = []
    for p in paths:
        items.extend(load_items(p))
    ids = [it.item_id for it in items]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate item_id across datasets: {sorted(dupes)}")
    return items
