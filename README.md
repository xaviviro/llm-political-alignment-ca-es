# llm-political-alignment-ca-es

> 🌐 **English** · [Català](README.ca.md)

**Measuring the cross-lingual political shift of LLM APIs in Catalan and Spanish,
grounded in real survey data from CEO (Catalonia) and CIS (Spain).**

This repository contains everything needed to reproduce the study: the code, the
survey-derived datasets with full provenance, the analysis, and the paper. It is
designed as a self-contained, reproducible scientific artifact.

---

## 1. The question

We do **not** ask whether a model is "left-wing or right-wing" — that framing is
fragile and the field is moving away from it (see §3). We ask a narrower,
falsifiable question:

> Does a model's **distribution of answers to real survey questions change with
> the language it is prompted in** (Catalan vs Spanish), and how far is it from
> the actual human population?

Two properties make this tractable and rigorous:

- **A real referent.** Every item is compared against a measured population
  distribution from an official survey — **CEO** (Centre d'Estudis d'Opinió,
  Catalonia) and **CIS** (Centro de Investigaciones Sociológicas, Spain) — not
  against an abstract left/right origin.
- **Language as the variable of interest.** The same item, the same population,
  asked in both languages: any movement is the **language-induced shift**.

**Why it matters.** Anyone deploying an LLM for Catalan/Spanish users (public
administration, education, media) inherits a concrete operational risk if the
model's political/value answers depend on the prompt language. And the Catalan
case is essentially **absent** from the multilingual literature on LLMs' political
positioning (Spanish appears only as one language among many) — putting Catalan and Spanish
at the centre, anchored to the populations that speak them, is the contribution.

The full rationale — why this question, why not the Political Compass, what the
method does and does not measure — is in
[`docs/measuring-cross-lingual-shift.md`](docs/measuring-cross-lingual-shift.md)
([català](docs/measuring-cross-lingual-shift.ca.md)). Sources and the prior literature
are in [`docs/references.md`](docs/references.md).

## 2. What we measure

- **Cross-lingual shift (headline)** — the mean Jensen–Shannon distance between a
  model's Catalan and Spanish answer distributions on the *same* item. `0` = the
  model answers identically in both languages; higher = larger language-induced
  shift.
- **Alignment to CEO/CIS** — `1 − JSD(model, population)`, reported as *context*,
  **not** as a moral ranking. Matching a population is not the same as being
  unbiased — under the "observer" framing the model is partly being asked to
  *predict* the population.
- **Refusal rate** — the fraction of samples where the model declined to answer
  (a disclaimer instead of an option), classified as refusal / empty / unparsed.
- **Directional pull** ([`scripts/directional.py`](scripts/directional.py)) — for
  a concept measured on *both* populations, does Catalan pull the model toward the
  Catalan population and Spanish toward the Spanish one? Having two reference
  populations is what distinguishes this from MENAValues, but it is **preliminary
  and not a headline result**: only one concept (left–right ideology) is currently
  measured on both CEO and CIS, so we report this as a future direction, not a
  claim. The headline contribution is the cross-lingual shift itself.

## 3. Method, and why not the Political Compass

The Political Compass is **format-fragile**: Röttger et al. (2024) show that the
response format and small prompt changes flip the result, and the two axes are an
abstract ideological origin with no population referent. We therefore reject it as
the primary instrument and build on the survey-grounded approach of **MENAValues**
(Zahraei & Asgari, 2025), adapted from the MENA region to Catalonia and Spain.

- **Ground truth.** 21 items with the real population marginal distribution: the
  bulk from **CEO** (independence, national identity, ideology, monarchy, the
  preferred relation between Catalonia and Spain, and a broad institutional-trust
  battery — governments, parliaments, parties, courts, police, army, unions, EU,
  UN, …) and two from **CIS** (the assessment of Spain's economy and left–right
  ideology). Each item records its **exact survey wave, source URL and access
  date**; only **aggregated marginals** are distributed, never raw microdata.
- **Conditions.** Every item is probed in **Catalan and Spanish** crossed with
  three **perspective framings** (neutral / personalised / observer), each under
  **several paraphrased prompt templates**. The answer distribution is estimated
  by sampling (closed APIs do not expose logprobs).
- **Models.** Accessed uniformly through [LiteLLM](https://github.com/BerriAI/litellm):
  any hosted API (OpenAI, Anthropic, Google, Groq, …) or a local model via Ollama.
  Large runs use each provider's **Batch API** (≈50% cost) — see
  [`scripts/run_batch.py`](scripts/run_batch.py).

## 4. Rigor

This study is built to be defensible, not just suggestive:

- **Bootstrap 95% confidence intervals** on every reported number; we read CI
  overlap rather than bare rankings.
- **Prompt-template robustness** (the Röttger critique): each framing is probed
  under several paraphrases and the between-template standard deviation is
  reported — a result that depends on phrasing is made visible.
- **Explicit failure handling.** A response where every sample fails to parse is
  marked invalid and **excluded** — never silently treated as a uniform "don't
  know". Each failure is classified (refusal / empty / unparsed) with example text.
- **Translation-equivalence validation**
  ([`scripts/validate_translations.py`](scripts/validate_translations.py)). Because
  half the prompts are translations, a "shift" could be a translation artefact. An
  LLM judge checks that the Catalan and Spanish versions of every item are
  equivalent; the cross-lingual shift ranking is unchanged when the only flagged
  item (a faithful CIS quirk) is excluded — the shift is **not** a translation
  artefact.
- **Reproducible data.** [`scripts/build_dataset.py`](scripts/build_dataset.py)
  rebuilds the datasets directly from the official CEO open microdata and the CIS
  open Excel marginals, selecting per item the latest wave that asked it.
- Pure metrics are **unit-tested** (`make test`).

## 5. Results (summary)

The headline figures are in [`paper/paper.pdf`](paper/paper.pdf). In short:

Across 10 models from five providers (Google, Anthropic, OpenAI, Llama via Groq,
DeepSeek):

- **The two Gemini models shift markedly more** between Catalan and Spanish
  (`0.556` and `0.534`) than every other model (`0.164`–`0.383`) — both in the raw
  shift and after the noise floor (net `0.383` / `0.318`).
- **Subtract the noise floor and the middle reorders.** With ~60 samples over
  11–13 categories the plug-in JSD is biased upward; the *net* shift (raw − floor)
  stratifies the panel: Claude and Llama are moderate (~`0.20`), while **gpt-oss
  and gpt-5.4-mini drop to ~`0.10` — their apparent shift is mostly sampling
  noise**.
- **DeepSeek is the most stable of all**: its raw shift (`0.164`) is essentially
  its noise floor, so the net shift falls to `0.037` ≈ 0 — it answers almost
  identically in both languages.
- **Refusals are low.** The highest is gemini-3.5-flash (`9.4%`), then gpt-oss
  (`5.7`–`6.2%`); most others rarely refuse. Equalising the effective valid sample
  size leaves the ranking unchanged — the Gemini gap is **not** a refusal artefact.
- **The shift concentrates on the national / institutional-trust axis**
  (independence, identity, trust in institutions), while left–right ideology and
  the economy are the **most stable** across languages.
- Alignment to the CEO/CIS populations is moderate across all models and must not
  be read as a moral ranking.

> **Status: preliminary.** A first study on a curated item set; numbers are
> time-stamped to specific survey waves. See the paper's Limitations section.

## 6. Reproduce it

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
make install                 # uv sync --group dev
make test                    # pure metrics + end-to-end on a network-free mock
make dry-run                 # mock run + HTML report, no API keys

# rebuild the datasets from the official CEO/CIS sources
uv run python scripts/build_dataset.py --source all

# evaluate models (set the matching API keys in the environment)
uv run python scripts/model.py --model <id>          # one model, synchronous
uv run python scripts/run_batch.py --model <id> --run # via Batch API (~50% cost)
uv run python scripts/summarize_results.py            # aggregate -> site/index.html

# rigor checks and the paper
uv run python scripts/validate_translations.py        # CA/ES equivalence
uv run python scripts/directional.py                  # directional language-pull
uv run python scripts/paper_figures.py && (cd paper && latexmk -pdf paper.tex)
```

Models are listed in `models.yaml` (LiteLLM ids); evaluation parameters
(languages, framings, sampling, bootstrap) in `config.yaml`. API keys are read
from the environment and never stored in the repo.

## 7. Repository structure

```
political_alignment/        the package
  metrics.py             pure metrics (JSD, alignment, cross-lingual shift, bootstrap)
  dataset.py             load + validate the CEO/CIS survey items
  framings.py            render items into prompts (neutral/personalised/observer + paraphrases)
  providers.py           LiteLLM access + answer-distribution estimation + a mock provider
  survey_alignment.py    the flagship method (shared by the live and batch paths)
  batch.py               Batch-API runner (OpenAI/Groq, Anthropic, Gemini)
scripts/                build_dataset, model, run_batch, summarize_results,
                        validate_translations, directional, paper_figures, ...
data/                   CEO/CIS items (schema in data/schema.md) + translation_check.csv
paper/                  the LaTeX paper, figures and compiled PDF
docs/                   the rationale (measuring-cross-lingual-shift) and references
config.yaml, models.yaml
```

## 8. Data sources and license

- **CEO** — Centre d'Estudis d'Opinió, Generalitat de Catalunya. Open microdata of
  the Baròmetre d'Opinió Política. <https://ceo.gencat.cat> ·
  <https://analisi.transparenciacatalunya.cat>
- **CIS** — Centro de Investigaciones Sociológicas, Gobierno de España. Open Excel
  marginals of the monthly barómetro. <https://www.cis.es>

Code is **MIT**-licensed (see [`LICENSE`](LICENSE)). The survey-derived data is
subject to the terms of CEO and CIS; only aggregated marginals are redistributed,
each with the originating study cited. The companion dataset is published at
<https://huggingface.co/datasets/xaviviro/llm-political-alignment-ca-es>.

## 9. Citation

**DOI: [10.13140/RG.2.2.22319.70561](https://doi.org/10.13140/RG.2.2.22319.70561)**

Vinaixa Roselló, X. (2026). *Measuring the cross-lingual political shift of LLM
APIs in Catalan and Spanish*. Preprint.
<https://doi.org/10.13140/RG.2.2.22319.70561>

See [`CITATION.cff`](CITATION.cff). Methodology after Zahraei & Asgari (2025)
(MENAValues). See [`docs/references.md`](docs/references.md) for the full list.
