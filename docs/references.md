# References

The framework's methodology is grounded in the recent literature that moves the
evaluation of LLMs' elicited political positions away from the static Political
Compass toward **survey-grounded cultural alignment**.

> The rationale — why we ask the question this way, why we discarded the
> Political Compass, and what MENAValues does and does not measure — is written
> up in [`measuring-cross-lingual-shift.md`](measuring-cross-lingual-shift.md)
> ([català](measuring-cross-lingual-shift.ca.md)).

## Primary methodological basis

- **Zahraei, P. S., & Asgari, E. (2025).** *I Am Aligned, But With Whom? MENA
  Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in
  LLMs.* arXiv:2510.13154. <https://arxiv.org/abs/2510.13154>
  Measures how closely a model's answer distribution matches a real human
  population distribution from authoritative surveys, crossing perspective
  framings (neutral / personalised / third-person observer) with language modes.
  Names three failure modes this framework operationalises: **logit leakage**
  (text refuses while internal probability mass is biased), **linguistic
  determinism** (representations collapse populations by language family), and
  **reasoning-induced degradation**. Dataset/code:
  <https://github.com/llm-lab-org/MENA-Values-Benchmark-Evaluating-Cultural-Alignment-and-Multilingual-Bias-in-Large-Language-Models>

## Supporting literature

- **Röttger, P., et al. (2024).** *Political Compass or Spinning Arrow? Towards
  More Meaningful Evaluations for Values and Opinions in LLMs.* ACL 2024.
  The robustness critique that motivates moving past the static compass:
  response format and small prompt changes flip results.
- **Feng, S., et al. (2023).** *From Pretraining Data to Language Models to
  Downstream Tasks: Tracking the Trails of Political Biases.* ACL 2023.
  Probability/masked-token probing of political stance.
### Multilingual political bias (work in / across languages)

- *Bias Beyond Borders: Political Ideology Evaluation and Steering in
  Multilingual LLMs* (2026). arXiv:2601.23001.
- *Multilingual Political Views of Large Language Models: Identification and
  Steering* (2025). arXiv:2507.22623.
- *Do Political Opinions Transfer Between Western Languages? An Analysis of
  Unaligned and Aligned Multilingual LLMs* (2025). arXiv:2508.05553.
- *Framing Political Bias in Multilingual LLMs Across Pakistani Languages*
  (2025). arXiv:2506.00068.
- Elbouanani et al. *Analyzing Political Bias in LLMs via Target-Oriented
  Sentiment Classification* (2025). arXiv:2505.19776.
- *Assessing the Political Fairness of Multilingual LLMs: A Case Study based on a
  21-way Multiparallel EuroParl Dataset* (2025). arXiv:2510.20508.

Recurring finding: the prompt language shifts the elicited stance; English is
consistently the most libertarian-left. Catalan is essentially absent from this
literature; Spanish appears only as one language among many.

### English-centric findings (a single-language slice)

- Hartmann, J., et al. (2023). *The political ideology of conversational AI.*
- Rozado, D. (2024). *The political preferences of LLMs.* PLOS ONE.
- Motoki, F., Pinho Neto, V., & Rodrigues, V. (2024). *More human than human:
  measuring ChatGPT political bias.* Public Choice.

## Survey data sources

- **CEO** — Centre d'Estudis d'Opinió, Generalitat de Catalunya. Baròmetre
  d'Opinió Política. <https://ceo.gencat.cat>
- **CIS** — Centro de Investigaciones Sociológicas, Gobierno de España.
  Barómetros and political-attitude studies. <https://www.cis.es>
