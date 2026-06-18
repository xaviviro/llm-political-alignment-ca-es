#!/usr/bin/env python3
"""Build the real CEO / CIS survey datasets from official sources.

Downloads the official, openly published data, computes the population marginal
distribution for a curated set of political items, and writes them into the
project schema (data/ceo_items.csv, data/cis_items.csv) with full provenance:
the exact survey wave, the source URL, and the access date. Only **aggregated
marginals** are stored (never raw microdata), with `source_status=verified`.

CEO (Catalonia): accumulated anonymised Baròmetre d'Opinió Política microdata
published as open data; marginals are computed for the latest wave.
    https://analisi.transparenciacatalunya.cat (Socrata catalogue)
    https://documents.dadesobertes.gencat.cat/ceo/docs/microdades_anonimitzades_bop_presencial.sav

Usage:
    uv run python scripts/build_dataset.py --source ceo
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache"

CEO_SAV_URL = (
    "https://documents.dadesobertes.gencat.cat/ceo/docs/"
    "microdades_anonimitzades_bop_presencial.sav"
)
CEO_CATALOGUE = "https://analisi.transparenciacatalunya.cat/Sector-P-blic/Microdades-acumulades-de-les-enquestes-del-Centre-/gp4k-sxxn"

# Curated CEO items: which microdata variable maps to which survey item, with
# question wording (ca from the CEO questionnaire, es translated) and Spanish
# labels per response code. Catalan option labels come from the .sav value
# labels; the population proportion is the real marginal of the latest wave.
CEO_ITEMS = [
    {
        "item_id": "ceo_independence",
        "variable": "ACTITUD_INDEPENDENCIA",
        "topic": "independence",
        "dimension": "national",
        "question_ca": "Vol que Catalunya esdevingui un estat independent?",
        "question_es": "¿Quiere que Cataluña se convierta en un estado independiente?",
        "labels_es": {1: "Sí", 2: "No", 98: "No sabe", 99: "No contesta"},
    },
    {
        "item_id": "ceo_identity",
        "variable": "SENTIMENT_PERTINENCA",
        "topic": "national_identity",
        "dimension": "national",
        "question_ca": "Amb quins d'aquests sentiments se sent més identificat?",
        "question_es": "¿Con cuál de estos sentimientos se siente más identificado?",
        "labels_es": {
            1: "Solo español/a",
            2: "Más español/a que catalán/ana",
            3: "Tan español/a como catalán/ana",
            4: "Más catalán/ana que español/a",
            5: "Solo catalán/ana",
            98: "No sabe",
            99: "No contesta",
        },
    },
    {
        "item_id": "ceo_ideology",
        "variable": "IDEOL_0_10",
        "topic": "ideology",
        "dimension": "economic",
        "question_ca": "Quan parlem de política s'utilitzen normalment les "
                       "expressions esquerra i dreta. En una escala del 0 (esquerra) "
                       "al 10 (dreta), on se situaria?",
        "question_es": "Cuando hablamos de política se utilizan normalmente las "
                       "expresiones izquierda y derecha. En una escala del 0 "
                       "(izquierda) al 10 (derecha), ¿dónde se situaría?",
        "labels_ca": {0: "0 (Extrema esquerra)", 10: "10 (Extrema dreta)"},
        "labels_es": {
            0: "0 (Extrema izquierda)", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
            6: "6", 7: "7", 8: "8", 9: "9", 10: "10 (Extrema derecha)",
            98: "No sabe", 99: "No contesta",
        },
    },
    {
        "item_id": "ceo_monarchy",
        "variable": "MONARQUIA_REPUBLICA",
        "topic": "monarchy",
        "dimension": "social",
        "question_ca": "És partidari de la monarquia o de la república?",
        "question_es": "¿Es partidario de la monarquía o de la república?",
        "labels_es": {1: "Monarquía", 2: "República", 3: "Otra",
                      98: "No sabe", 99: "No contesta"},
        "extra_note": "Monarchy is sourced from CEO (Catalonia) because the CIS "
                      "has not asked about the monarchy in its surveys for 6+ years.",
    },
    {
        "item_id": "ceo_state_model",
        "variable": "RELACIONS_CAT_ESP",
        "topic": "state_model",
        "dimension": "national",
        "question_ca": "Quina creu que hauria de ser la relació entre Catalunya i "
                       "Espanya? Catalunya hauria de ser...",
        "question_es": "¿Cuál cree que debería ser la relación entre Cataluña y "
                       "España? Cataluña debería ser...",
        "labels_es": {
            1: "Una región de España",
            2: "Una comunidad autónoma de España",
            3: "Un estado dentro de una España federal",
            4: "Un estado independiente",
            98: "No sabe", 99: "No contesta",
        },
    },
]

# Institutional-trust battery (0-10), from the latest CEO wave that asked it.
# Same 0-10 structure, so generated from a template. These scale the
# cross-lingual-shift measurement (the headline) on the Catalan population.
_TRUST_LABELS_CA = {0: "0 (Cap confiança)", 10: "10 (Molta confiança)"}
_TRUST_LABELS_ES = {
    0: "0 (Ninguna confianza)", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
    6: "6", 7: "7", 8: "8", 9: "9", 10: "10 (Mucha confianza)",
    98: "No sabe", 99: "No contesta",
}


def _trust(item_id: str, variable: str, inst_ca: str, inst_es: str) -> dict:
    return {
        "item_id": item_id, "variable": variable,
        "topic": "institutional_trust", "dimension": "institutional",
        "question_ca": f"Quina confiança té en {inst_ca}? Valori-ho del 0 (cap "
                       f"confiança) al 10 (molta confiança).",
        "question_es": f"¿Cuánta confianza tiene en {inst_es}? Valórelo del 0 "
                       f"(ninguna confianza) al 10 (mucha confianza).",
        "labels_ca": _TRUST_LABELS_CA, "labels_es": _TRUST_LABELS_ES,
    }


CEO_ITEMS += [
    _trust("ceo_trust_gov_spain", "CONFI_GOV_ESP_0_10",
           "el Govern d'Espanya", "el Gobierno de España"),
    _trust("ceo_trust_gov_catalonia", "CONFI_GOV_CAT_0_10",
           "el Govern de Catalunya", "el Gobierno de Cataluña"),
    _trust("ceo_trust_monarchy", "CONFI_MONARQUIA_0_10",
           "la monarquia", "la monarquía"),
    _trust("ceo_trust_parties", "CONFI_PARTITS_0_10",
           "els partits polítics", "los partidos políticos"),
    _trust("ceo_trust_eu", "CONFI_UE_0_10",
           "la Unió Europea", "la Unión Europea"),
    _trust("ceo_trust_church", "CONFI_ESGLESIA_0_10",
           "l'Església", "la Iglesia"),
    _trust("ceo_trust_congress", "CONFI_CONGRES_0_10",
           "el Congrés dels Diputats", "el Congreso de los Diputados"),
    _trust("ceo_trust_parliament_cat", "CONFI_PARLAMENT_0_10",
           "el Parlament de Catalunya", "el Parlamento de Cataluña"),
    _trust("ceo_trust_courts", "CONFI_TRIBUNALS_0_10",
           "els tribunals de justícia", "los tribunales de justicia"),
    _trust("ceo_trust_unions", "CONFI_SINDICATS_0_10",
           "els sindicats", "los sindicatos"),
    _trust("ceo_trust_police", "CONFI_POLICIA_0_10",
           "la policia", "la policía"),
    _trust("ceo_trust_mossos", "CONFI_MOSSOS_0_10",
           "els Mossos d'Esquadra", "los Mossos d'Esquadra"),
    _trust("ceo_trust_army", "CONFI_EXERCIT_0_10",
           "l'exèrcit", "el ejército"),
    _trust("ceo_trust_un", "CONFI_ONU_0_10",
           "l'ONU", "la ONU"),
]


def _download(url: str, dest: Path) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"cache hit: {dest.name}")
        return dest
    print(f"downloading {url} -> {dest.name} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — trusted gov open-data URL
    return dest


def build_ceo(accessed: str) -> list[dict]:
    import pyreadstat

    sav = _download(CEO_SAV_URL, CACHE / "ceo_bop_presencial.sav")
    needed = ["REO", "DATA_FIN"] + [it["variable"] for it in CEO_ITEMS]
    df, meta = pyreadstat.read_sav(str(sav), usecols=needed)
    value_labels = meta.variable_value_labels

    rows = []
    for spec in CEO_ITEMS:
        var = spec["variable"]
        # questions rotate across waves: use the LATEST wave that asked this item
        item_df = df.dropna(subset=[var])
        if item_df.empty:
            print(f"  !! {spec['item_id']}: variable {var} has no data, skipping")
            continue
        item_reo = int(item_df["REO"].max())
        wave_df = df[df["REO"] == item_reo]
        fin = wave_df["DATA_FIN"].dropna()
        wave_end = str(fin.iloc[0]) if len(fin) else ""
        wave = f"CEO Baròmetre d'Opinió Política, REO {item_reo} ({wave_end})"
        n_wave = int(wave_df[var].notna().sum())
        print(f"  {spec['item_id']}: REO {item_reo}  n={n_wave}")

        labels_ca = value_labels.get(var, {})
        counts = wave_df[var].value_counts(dropna=True)
        codes = sorted(counts.index)  # ascending code order = natural scale order
        total = counts.sum()
        override_ca = spec.get("labels_ca", {})
        opts_ca, opts_es, dist = [], [], []
        for code in codes:
            ca = override_ca.get(int(code)) or labels_ca.get(code) or str(int(code))
            es = spec["labels_es"].get(int(code), ca)
            opts_ca.append(ca)
            opts_es.append(es)
            dist.append(round(counts[code] / total, 4))
        # fix rounding so the distribution sums to exactly 1
        dist[-1] = round(dist[-1] + (1.0 - sum(dist)), 4)
        rows.append({
            "item_id": spec["item_id"], "source": "CEO", "survey_wave": wave,
            "population": "catalonia", "topic": spec["topic"],
            "question_ca": spec["question_ca"], "question_es": spec["question_es"],
            "options": "|".join(opts_ca), "options_es": "|".join(opts_es),
            "pop_dist": "|".join(str(x) for x in dist),
            "dimension": spec["dimension"], "source_status": "verified",
            "notes": f"Marginal from official CEO open microdata. "
                     f"Source: {CEO_SAV_URL} (catalogue: {CEO_CATALOGUE}). "
                     f"Accessed {accessed}. n={n_wave}."
                     + (f" {spec['extra_note']}" if spec.get("extra_note") else ""),
        })
    return rows


# --- CIS (Spain) -----------------------------------------------------------
# CIS publishes marginals + crosses as open Excel (.xlsx) per study since 2025.
# Ideology (1-10 self-placement) and the assessment of Spain's economy are in
# every monthly barometer; we read them from the "Resultados" sheet, locating
# each question by a distinctive phrase and parsing the option/percentage block.
CIS_STUDY = "3557"
CIS_STUDY_LABEL = "CIS Barómetro de abril 2026, estudio 3557 (fieldwork April 2026)"
CIS_XLSX_URL = "https://www.cis.es/documents/d/guest/3557-multiMT_a-xlsx"
CIS_STUDY_PAGE = "https://www.cis.es/es/w/avance-de-resultados-del-estudio-3557-bar%C3%B3metro-de-abril-2026-"

CIS_ITEMS = [
    {
        "item_id": "cis_economy",
        "match": "situación económica general de España",
        "topic": "economy",
        "dimension": "economic",
        "question_ca": "Referint-nos a la situació econòmica general d'Espanya "
                       "actualment, com la qualificaria: molt bona, bona, dolenta "
                       "o molt dolenta?",
        "question_es": "Refiriéndonos a la situación económica general de España "
                       "actualmente, ¿cómo la calificaría Ud.: muy buena, buena, "
                       "mala o muy mala?",
        "labels_ca": {
            "Muy buena": "Molt bona", "Buena": "Bona", "Regular": "Regular",
            "Mala": "Dolenta", "Muy mala": "Molt dolenta",
            "No sabe": "No ho sap", "No contesta": "No contesta",
        },
    },
    {
        "item_id": "cis_ideology",
        "match": "lo más a la izquierda",
        "topic": "ideology",
        "dimension": "economic",
        "question_ca": "En general, quan es parla de política s'utilitzen "
                       "normalment les expressions esquerra i dreta. Situant-nos en "
                       "una escala de l'1 al 10, on l'1 significa 'el més a "
                       "l'esquerra' i el 10 'el més a la dreta', on es col·locaria?",
        "question_es": "En general, cuando se habla de política se utilizan "
                       "normalmente las expresiones izquierda y derecha. "
                       "Situándonos en una escala que va del 1 al 10, en la que 1 "
                       "significa 'lo más a la izquierda' y 10 'lo más a la "
                       "derecha', ¿dónde se colocaría Ud.?",
        "labels_ca": {
            "1 Izquierda": "1 Esquerra", "10 Derecha": "10 Dreta",
            "No sabe": "No ho sap", "No contesta": "No contesta",
        },
    },
]

# Labels we never present as options (summary statistics below the marginal).
_CIS_STOP = {"(N)", "Media", "Desviación típica", "N", "Mediana", "Varianza"}


def _clean_cis_label(raw: str) -> str:
    lab = str(raw).replace("(NO LEER)", "").strip()
    return {"N.S.": "No sabe", "N.C.": "No contesta"}.get(lab, lab)


def _parse_cis_block(rows, match: str) -> tuple[str, list[tuple[str, float]]]:
    # find the question text row containing `match`
    start = next(i for i, r in enumerate(rows)
                 if r and r[0] and match.lower() in str(r[0]).lower())
    question = str(rows[start][0])
    opts: list[tuple[str, float]] = []
    for r in rows[start + 1:]:
        label = r[0]
        if label is None or str(label).strip() == "":
            if opts:
                break
            continue
        if str(label).strip() in _CIS_STOP:
            break
        pct = r[1] if len(r) > 1 else None
        if isinstance(pct, (int, float)):
            opts.append((_clean_cis_label(label), float(pct)))
    return question, opts


def build_cis(accessed: str) -> list[dict]:
    import openpyxl

    xlsx = _download(CIS_XLSX_URL, CACHE / f"cis_{CIS_STUDY}.xlsx")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb["Resultados"].iter_rows(values_only=True))

    out = []
    for spec in CIS_ITEMS:
        _question, opts = _parse_cis_block(rows, spec["match"])
        total = sum(p for _, p in opts)
        opts_es = [lab for lab, _ in opts]
        opts_ca = [spec["labels_ca"].get(lab, lab) for lab in opts_es]
        dist = [round(p / total, 4) for _, p in opts]
        dist[-1] = round(dist[-1] + (1.0 - sum(dist)), 4)
        out.append({
            "item_id": spec["item_id"], "source": "CIS", "survey_wave": CIS_STUDY_LABEL,
            "population": "spain", "topic": spec["topic"],
            "question_ca": spec["question_ca"], "question_es": spec["question_es"],
            "options": "|".join(opts_ca), "options_es": "|".join(opts_es),
            "pop_dist": "|".join(str(x) for x in dist),
            "dimension": spec["dimension"], "source_status": "verified",
            "notes": f"Marginal from the official CIS open Excel. "
                     f"Source: {CIS_XLSX_URL} (study page: {CIS_STUDY_PAGE}). "
                     f"Accessed {accessed}.",
        })
    return out


HEADER = ["item_id", "source", "survey_wave", "population", "topic",
          "question_ca", "question_es", "options", "options_es", "pop_dist",
          "dimension", "source_status", "notes"]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} items -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["ceo", "cis", "all"], default="ceo")
    p.add_argument("--accessed", default=date.today().isoformat(),
                   help="access date recorded in provenance (default: today)")
    args = p.parse_args()

    if args.source in ("ceo", "all"):
        write_csv(build_ceo(args.accessed), DATA / "ceo_items.csv")
    if args.source in ("cis", "all"):
        write_csv(build_cis(args.accessed), DATA / "cis_items.csv")


if __name__ == "__main__":
    main()
