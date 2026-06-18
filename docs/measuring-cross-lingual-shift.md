# Measuring political bias across languages — approach and rationale

> 🌐 **English** · [Català](measuring-cross-lingual-shift.ca.md)

This document explains *why* the framework measures political bias the way it
does: what question we are actually asking, which instruments we rejected and
why, which recent literature we build on (including work in languages other than
English), and the honest limits of the test we adopted.

## 1. The question we are asking

We are **not** asking "is model X left-wing or right-wing." That framing is the
one the field is moving away from (§3). We ask a narrower, falsifiable question:

> Does a model's **distribution of answers to real survey questions change with
> the language it is prompted in** (English vs Catalan vs Spanish), and how far
> is that distribution from the **actual human population**?

Two properties make this tractable and rigorous:

- **A real referent.** Each item is compared against a measured population
  distribution from an official survey — CEO (Catalonia) and CIS (Spain) — not
  against an abstract left/right origin.
- **Language as the variable of interest.** The same item, the same population,
  asked in three languages: any movement is the *language-induced shift*, the
  phenomenon we care about.

## 2. Why language, and what other work has found

The finding that **prompt language moves a model's elicited political stance**
is now replicated across several research groups and language families:

- **Bias Beyond Borders: Political Ideology Evaluation and Steering in
  Multilingual LLMs** (2026, arXiv:2601.23001) — English is consistently the
  most libertarian-left; other languages drift toward the centre/right.
- **Multilingual Political Views of Large Language Models: Identification and
  Steering** (2025, arXiv:2507.22623) — language-dependent ideological
  orientation across many languages.
- **Do Political Opinions Transfer Between Western Languages?** (2025,
  arXiv:2508.05553) — English alignment fine-tuning propagates to other
  languages (transfer effects).
- **Framing Political Bias in Multilingual LLMs Across Pakistani Languages**
  (2025, arXiv:2506.00068) — the effect outside Western languages.
- **Analyzing Political Bias in LLMs via Target-Oriented Sentiment
  Classification** (Elbouanani et al., 2025, arXiv:2505.19776) — biases more
  pronounced in English/Spanish/French than in Arabic/Chinese/Russian.
- **Assessing the Political Fairness of Multilingual LLMs: a 21-way
  Multiparallel EuroParl Dataset** (2025, arXiv:2510.20508) — controlled
  multiparallel comparison across European languages.

Against this backdrop, the recurring English-centric findings (ChatGPT and peers
trending left-libertarian: Hartmann et al. 2023; Rozado 2024, *PLOS ONE*;
Motoki, Pinho Neto & Rodrigues 2024, *Public Choice*) are a **single-language
slice** of a multilingual phenomenon.

**Catalan is essentially absent from this literature; Spanish appears only as
one of many languages.** A study that puts Catalan and Spanish at the centre,
anchored to the populations that actually speak them (CEO, CIS), is the gap we
target.

## 3. Why we discarded the Political Compass

The Political Compass Test (and the familiar two-axis quadrant) is the default
people reach for. We rejected it as our **primary instrument**:

- **Röttger et al. (2024), *Political Compass or Spinning Arrow? Towards More
  Meaningful Evaluations for Values and Opinions in LLMs*, ACL 2024** is the
  decisive critique: the result is **format-fragile** — forcing a multiple
  choice vs allowing open generation flips the outcome, and small prompt changes
  move the needle. The "compass position" is often an artifact of one wording.
- **Feng et al. (2023), ACL** showed the axes can be probed, but the axes
  themselves are an **abstract ideological origin** with no population referent:
  "distance from the compass centre" is not distance from any real human group.
- **Construct problem.** A single forced choice per proposition, scored against
  proprietary weights, conflates many things into two numbers and hides the
  language effect we want to isolate.

We therefore do not use the compass as a measurement. At most it survives as a
*familiar visualization*, never as the ground truth.


## 4. What we adopted: MENAValues, adapted to CEO + CIS

We build on **Zahraei & Asgari (2025), *I Am Aligned, But With Whom? MENA Values
Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs*,
arXiv:2510.13154**, and adapt it from the MENA region to Catalonia and Spain.

### 4.1 The method we inherit

- Compare a model's **answer distribution** to a **real population
  distribution** from authoritative surveys.
- Cross **perspective framings** (neutral / personalised / third-person
  observer) with **language modes**.
- Characterise failure modes: **logit leakage** (the text refuses while the
  internal probability mass is biased), **linguistic determinism / cross-lingual
  value shift** (answers collapse by prompt language), and **reasoning-induced
  degradation**.

### 4.2 Our adaptation

- **Ground truth = CEO (Catalonia) + CIS (Spain)** real marginals, not MENA
  surveys. Items: independence, national identity, ideology, monarchy, and
  state-model from CEO; the assessment of Spain's economy and left–right
  ideology from CIS — each with the exact survey wave, source URL and access
  date, distributing **aggregated marginals only**.
- **Languages = English / Catalan / Spanish.**
- Metrics implemented as `alignment = 1 − JSD(model, population)`, **cross-lingual
  consistency**, and **between-template robustness** (see §5.4).

### 4.3 What this test says — and what it does not

**It says:** how closely a model's answer distribution matches a *specified human
population*, and whether that match **changes with the prompt language**. This is
empirical and grounded, not an abstract axis.

**It does not say:**

- It is **not a moral ranking**. "Closer to the CEO/CIS population" is not
  "better" or "less biased" in any normative sense.
- **Matching the population ≠ being unbiased.** Under the *observer* and
  *personalised* framings the model is partly being asked to **predict** the
  population — so a high alignment there is closer to a forecasting score than to
  a statement of the model's own values. (In our own runs the observer framing
  aligns best, exactly as this caveat predicts.)
- It measures **representativeness and language-stability**, not correctness.

### 4.4 Its rigor — an honest verdict

We read the full MENAValues paper. Our assessment:

**Strong empirical engineering:** 95% bootstrap confidence intervals throughout
(B=1,000); a large, real ground truth (864 items from World Values Survey Wave 7
and the Arab Opinion Index 2022, with post-stratification weights); human-
validated translations; transparent, formula-level metric definitions (NVAS,
CLCS, a 75%-of-max-logprob threshold for logit leakage, KL divergence).

**Weaker where it matters for inference:**

- **No significance testing or multiple-comparison control** across the large
  models × countries × framings surface — directional ↑↓ marks risk over-reading
  noise (the CIs mitigate this only partly).
- **No prompt-template robustness** — it crosses framings but does **not** vary
  the wording *within* a framing, so the very Röttger fragility it positions
  against is not controlled.
- **Construct validity is unexamined** — the persona/observer framing makes high
  alignment partly tautological, and the limitations section does not engage with
  this.
- It is an **unreviewed preprint**.

### 4.5 Where our adaptation is stronger, and where it shares the gaps

- **Stronger:** we add **prompt-template robustness** (N paraphrases per framing,
  reporting the between-template SD) — closing the Röttger gap MENAValues leaves
  open; we make **parse failures explicit** (a response where every sample fails
  is marked invalid and excluded, never silently treated as a uniform "don't
  know"); and we attach **full provenance** (survey wave + source + access date)
  to every item, with bootstrap CIs on every reported number.
- **Shared gaps:** a small item set is preliminary; the construct-validity caveat
  (§5.3) applies to us too; and the logit-leakage analysis is only possible on
  providers that expose token log-probabilities (OpenAI, Gemini, local vLLM/
  llama.cpp), not on the closed chat APIs we sample from.

## 5. What "rigor" means for us in practice

- Report **representativeness, not rankings**; show CI overlap rather than bare
  orderings.
- Report the **between-template SD** so a phrasing-sensitive result is visible.
- Keep **NS/NC** (don't-know/no-answer) categories faithful to the survey and be
  explicit about how they are handled.
- Treat 5–7 items as **preliminary**; numbers are time-stamped to a survey wave.
- Never present an illustrative placeholder as a real measurement; the loader
  carries the `source_status` flag through to the report.

## References

See [`references.md`](references.md) for the primary methodological sources
(MENAValues, Röttger, Feng) and the survey data providers.
