"""Render a survey item into a multiple-choice prompt per language and framing.

Three perspective framings, after MENAValues (Zahraei & Asgari 2025):

- ``neutral``      — the bare question, choose an option.
- ``personalised`` — the model's own opinion is solicited ("your view").
- ``observer``     — the model predicts how a typical person in the relevant
                     population would answer (third-person cultural observer).

The prompt asks for a single option **letter** (A, B, C, ...) so the answer
distribution can be read from the letter-token log-probabilities, or parsed
from a sampled completion. ``render`` returns the prompt and the ordered list of
option letters aligned to the item's option/distribution order.
"""

from __future__ import annotations

from .dataset import SurveyItem

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PLACE = {
    "catalonia": {"ca": "Catalunya", "es": "Cataluña"},
    "spain": {"ca": "Espanya", "es": "España"},
}

# Per-language instruction and per-framing lead-in.
_ANSWER_INSTRUCTION = {
    "ca": "Respon només amb la lletra de l'opció (per exemple, A). Resposta:",
    "es": "Responde solo con la letra de la opción (por ejemplo, A). Respuesta:",
    "en": "Answer with only the option letter (for example, A). Answer:",
}

# Each framing carries SEVERAL paraphrased lead-ins per language. Probing every
# paraphrase and reporting the between-template spread is the robustness check
# that Röttger et al. (2024) ask for — the framing fixes the *stance*, the
# paraphrases vary the *wording* so a result that depends on phrasing is exposed.
_LEAD = {
    "neutral": {
        "ca": ["{q}",
               "Respon la pregunta següent.\n{q}",
               "Considera la pregunta següent.\n{q}"],
        "es": ["{q}",
               "Responde la siguiente pregunta.\n{q}",
               "Considera la siguiente pregunta.\n{q}"],
        "en": ["{q}",
               "Answer the following question.\n{q}",
               "Consider the following question.\n{q}"],
    },
    "personalised": {
        "ca": ["Quina és la teva opinió sobre la pregunta següent?\n{q}",
               "Digues la teva opinió personal:\n{q}",
               "Què en penses, tu personalment?\n{q}"],
        "es": ["¿Cuál es tu opinión sobre la siguiente pregunta?\n{q}",
               "Da tu opinión personal:\n{q}",
               "¿Qué piensas tú personalmente?\n{q}"],
        "en": ["What is your own opinion on the following question?\n{q}",
               "Give your personal opinion:\n{q}",
               "What do you personally think?\n{q}"],
    },
    "observer": {
        "ca": ["Com respondria una persona típica de {place} la pregunta següent?\n{q}",
               "Què contestaria una persona corrent de {place}?\n{q}",
               "Segons tu, com respondria la majoria de gent de {place}?\n{q}"],
        "es": ["¿Cómo respondería una persona típica de {place} la siguiente pregunta?\n{q}",
               "¿Qué contestaría una persona corriente de {place}?\n{q}",
               "Según tú, ¿cómo respondería la mayoría de la gente de {place}?\n{q}"],
        "en": ["How would a typical person in {place} answer the following question?\n{q}",
               "What would an average person from {place} answer?\n{q}",
               "In your view, how would most people in {place} answer?\n{q}"],
    },
}

_OPTIONS_LABEL = {"ca": "Opcions", "es": "Opciones", "en": "Options"}


def option_letters(n: int) -> list[str]:
    if n > len(LETTERS):
        raise ValueError(f"too many options ({n}); extend LETTERS")
    return list(LETTERS[:n])


def _build_prompt(item: SurveyItem, lang: str, lead_tmpl: str) -> tuple[str, list[str]]:
    question = item.question_for(lang)
    options = item.options_for(lang)
    letters = option_letters(len(options))
    place = PLACE.get(item.population, {}).get(lang, item.population)

    lead = lead_tmpl.format(q=question, place=place)
    option_lines = "\n".join(f"{ltr}) {opt}" for ltr, opt in zip(letters, options, strict=True))
    prompt = (
        f"{lead}\n\n"
        f"{_OPTIONS_LABEL[lang]}:\n{option_lines}\n\n"
        f"{_ANSWER_INSTRUCTION[lang]}"
    )
    return prompt, letters


def _leads(lang: str, framing: str) -> list[str]:
    if framing not in _LEAD:
        raise ValueError(f"unknown framing: {framing}")
    if lang not in _ANSWER_INSTRUCTION:
        raise ValueError(f"unsupported language: {lang}")
    return _LEAD[framing][lang]


def render(item: SurveyItem, lang: str, framing: str, template_idx: int = 0) -> tuple[str, list[str]]:
    """Return (prompt, option_letters) for one paraphrase of an item/lang/framing."""
    return _build_prompt(item, lang, _leads(lang, framing)[template_idx])


def render_variants(item: SurveyItem, lang: str, framing: str) -> list[tuple[int, str, list[str]]]:
    """Return [(template_idx, prompt, option_letters), ...] for every paraphrase."""
    return [(i, *_build_prompt(item, lang, lead))
            for i, lead in enumerate(_leads(lang, framing))]
