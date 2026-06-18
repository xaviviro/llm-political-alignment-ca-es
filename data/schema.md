# Survey item schema

Each survey item is one row in `data/ceo_items.csv` or `data/cis_items.csv`.
Items carry the question, its answer options, and the **real population response
distribution** the model is compared against.

| column | meaning |
| ------ | ------- |
| `item_id` | stable unique id, e.g. `ceo_001`, `cis_014` |
| `source` | `CEO` (Catalonia) or `CIS` (Spain) |
| `survey_wave` | survey edition the distribution is taken from, e.g. `CEO 2025 Q1 (REO 1234)` — pin the exact study so the comparison is reproducible |
| `population` | `catalonia` or `spain` — the human population the distribution describes |
| `topic` | short tag, e.g. `independence`, `monarchy`, `immigration`, `economy`, `religion` |
| `question_ca` | the item wording in Catalan |
| `question_es` | the item wording in Spanish |
| `options` | answer options, pipe-separated, in fixed order, e.g. `Molt malament\|Malament\|Bé\|Molt bé` |
| `options_es` | the same options in Spanish, pipe-separated, same order |
| `pop_dist` | population proportion per option, pipe-separated, same order as `options`; must sum to ~1.0 |
| `dimension` | optional value dimension tag (e.g. `economic`, `national`, `social`) for aggregation |
| `source_status` | `verified` (real distribution transcribed from the cited wave) or `example` (illustrative placeholder — NOT real data) |
| `notes` | free text; for `example` rows, says so explicitly |

## Ordinal vs nominal items

Some items are **ordinal scales** and some are **nominal** categories — this
changes which distance metric is meaningful (see the Wasserstein-1 / EMD column
in the report):

- **Ordinal** — the substantive options carry a numeric scale code as a leading
  integer in the label, e.g. `0 (Cap confiança) … 10 (Molta confiança)` or the
  `0`–`10` / `1`–`10` left–right self-placement. For these, distance along the
  scale matters (one step ≠ extreme-to-extreme), so we also report the
  **ordinal-aware Wasserstein-1 (EMD)** alongside JSD. Detected automatically by
  `SurveyItem.is_ordinal` (≥3 numeric-coded options). Ordinal items: `ceo_ideology`,
  all `ceo_trust_*`, `cis_ideology`.
- **Nominal** — unordered categories with no numeric code: `ceo_independence`,
  `ceo_identity`, `ceo_monarchy`, `ceo_state_model`, and the verbal economy Likert
  `cis_economy` (kept nominal: no numeric coding is imposed on the verbal labels).
  These stay JSD-only.

The don't-know / no-answer options (`No ho sap`, `No contesta`) are non-ordinal
and excluded from the Wasserstein support (handled separately), never placed on
the scale.

## Rules

- **Never present an `example` row as a real measurement.** The loader keeps the
  `source_status` flag through to the report so illustrative items are labelled.
- `pop_dist` length must equal `options` length, and values must sum to 1.0
  within tolerance — `political_alignment.dataset` validates this on load.
- Keep the **original survey wording**; do not paraphrase to fit the option set.
  If CEO/CIS wording cannot be mapped cleanly to a fixed option set, record the
  original wording in `notes`.
- `survey_wave` must identify the exact study (CEO REO number / CIS study
  number) so anyone can reproduce the ground-truth distribution.

## Adding real data

1. Download the published tables or microdata from CEO (<https://ceo.gencat.cat>)
   or CIS (<https://www.cis.es>) for the chosen wave.
2. Transcribe the marginal distribution for each selected question into
   `pop_dist`, set `source_status=verified`, and cite the wave in `survey_wave`.
3. Run `make validate` to check option/distribution consistency.
