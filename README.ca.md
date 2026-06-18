# llm-political-alignment-ca-es

> 🌐 [English](README.md) · **Català**

**Mesura del desplaçament polític per llengua de les API de models de llenguatge
(LLM) en català i castellà, ancorat en dades reals d'enquesta del CEO (Catalunya)
i el CIS (Espanya).**

Aquest repositori conté tot el necessari per reproduir l'estudi: el codi, els
conjunts de dades derivats d'enquestes amb procedència completa, l'anàlisi i l'article
científic. És un artefacte científic autocontingut i reproduïble.

---

## 1. La pregunta

**No** preguntem si un model és "d'esquerres o de dretes" —és un enquadrament
fràgil que el camp està abandonant (vegeu §3). Fem una pregunta més estreta i
falsable:

> Canvia la **distribució de respostes d'un model a preguntes d'enquesta reals
> segons la llengua amb què se'l consulta** (català vs castellà), i com de lluny
> està de la població humana real?

Dues propietats ho fan rigorós:

- **Un referent real.** Cada ítem es compara amb una distribució poblacional
  mesurada d'una enquesta oficial —**CEO** (Catalunya) i **CIS** (Espanya)—, no
  amb un origen abstracte esquerra/dreta.
- **La llengua com a variable d'interès.** El mateix ítem, la mateixa població,
  preguntat en totes dues llengües: qualsevol moviment és el **desplaçament
  induït per la llengua**.

**Per què importa.** Qui desplegui un LLM per a usuaris catalans/castellans
(administració, educació, mitjans) hereta un risc operatiu concret si les
respostes polítiques del model depenen de la llengua de la indicació. I el cas català
és pràcticament **absent** de la literatura multilingüe sobre el posicionament
polític dels LLM (el castellà hi surt com una llengua més) —posar el català i el castellà al centre,
ancorats a les poblacions que els parlen, és la contribució.

La justificació completa —per què aquesta pregunta, per què no el Political
Compass, què mesura i què no— és a
[`docs/measuring-cross-lingual-shift.ca.md`](docs/measuring-cross-lingual-shift.ca.md).
Les fonts i la literatura, a [`docs/references.md`](docs/references.md).

## 2. Què mesurem

- **Desplaçament translingüe (titular)** — la divergència de Jensen–Shannon
  mitjana entre les distribucions de resposta en català i en castellà al *mateix*
  ítem. `0` = respon igual; més alt = més desplaçament per llengua.
- **Alineació amb CEO/CIS** — `1 − JSD(model, població)`, reportada com a
  *context*, **no** com a rànquing moral.
- **Taxa de rebuig** — fracció de mostres on el model es nega a respondre
  (descàrrec en lloc d'opció), classificada com a rebuig / buit / no analitzable.
- **Atracció direccional** ([`scripts/directional.py`](scripts/directional.py)) —
  per a un concepte mesurat en *totes dues* poblacions, el català atrau el model
  cap a la població catalana i el castellà cap a l'espanyola? Tenir dues poblacions
  de referència és el que ho distingeix de MENAValues, però és **preliminar i no
  un resultat de capçalera**: de moment només un concepte (la ideologia
  esquerra–dreta) es mesura a la vegada al CEO i al CIS, així que ho reportem com a
  via de futur, no com una afirmació. La contribució principal és el desplaçament
  translingüe.

## 3. Mètode, i per què no el Political Compass

El Political Compass és **fràgil al format** (Röttger et al., 2024) i els seus
eixos són un origen ideològic abstracte sense referent poblacional. El rebutgem
com a instrument principal i ens basem en l'enfocament ancorat en enquestes de
**MENAValues** (Zahraei & Asgari, 2025), adaptat de la regió MENA a Catalunya i
Espanya.

- **Veritat de referència.** 21 ítems amb la distribució marginal real de població:
  la majoria del **CEO** (independència, identitat nacional, ideologia, monarquia,
  model d'estat i una àmplia bateria de confiança institucional) i dos del **CIS**
  (situació econòmica d'Espanya i ideologia esquerra–dreta). Cada ítem grava la
  seva **onada exacta, la URL de la font i la data d'accés**; només es distribueixen
  **marginals agregats**, mai microdades.
- **Condicions.** Cada ítem es prova en **català i castellà** × tres
  **enquadraments** (neutral / personalitzat / observador), cadascun amb diverses
  **paràfrasis** de la indicació. La distribució s'estima per mostreig.
- **Models.** Accés uniforme via [LiteLLM](https://github.com/BerriAI/litellm):
  qualsevol API (OpenAI, Anthropic, Google, Groq…) o model local via Ollama. Les
  tirades grans usen la **Batch API** de cada proveïdor (≈50% de cost).

## 4. Rigor

- **Intervals de confiança de remostreig (bootstrap) al 95%** a cada número.
- **Robustesa de plantilles** (crítica de Röttger): cada enquadrament es prova amb
  diverses paràfrasis i es reporta la desviació entre plantilles.
- **Fallades explícites.** Una resposta on cap mostra s'analitza es marca invàlida
  i **s'exclou** —mai s'emmascara com a uniforme—; cada fallada es classifica
  (rebuig / buit / no analitzable) amb text d'exemple.
- **Validació d'equivalència de traducció**
  ([`scripts/validate_translations.py`](scripts/validate_translations.py)): un
  jutge LLM comprova que les versions CA i ES de cada ítem són equivalents; el
  rànquing de desplaçament no canvia excloent l'únic ítem marcat → el desplaçament
  **no és un artefacte de traducció**.
- **Dades reproduïbles.** [`scripts/build_dataset.py`](scripts/build_dataset.py)
  reconstrueix els conjunts de dades des de les microdades obertes del CEO i l'Excel obert
  del CIS, triant per ítem l'última onada que el va preguntar.
- Mètriques pures **amb tests** (`make test`).

## 5. Resultats (resum)

Les figures completes són a l'article científic ([`paper/paper.pdf`](paper/paper.pdf)). En resum:

Sobre 10 models de cinc proveïdors (Google, Anthropic, OpenAI, Llama via Groq i
DeepSeek):

- **Els dos models Gemini es desplacen molt més** entre català i castellà
  (`0.556` i `0.534`) que la resta (`0.164`–`0.383`), tant en cru com en net
  (`0.383` / `0.318`).
- **En restar el soroll de fons, el bloc del mig es reordena.** Amb ~60 mostres
  sobre ítems d'11–13 categories la JSD té biaix positiu; el desplaçament *net*
  (cru − soroll) estratifica el panell: Claude i Llama queden moderats (~`0.20`),
  mentre que **gpt-oss i gpt-5.4-mini cauen a ~`0.10` — el seu desplaçament és en
  bona part soroll de mostreig**.
- **DeepSeek és el més estable de tots**: el seu desplaçament cru (`0.164`) és
  pràcticament el seu soroll de fons, així que el net cau a `0.037` ≈ 0 — respon
  gairebé igual en totes dues llengües.
- **Els rebuigs són baixos.** El màxim és gemini-3.5-flash (`9.4%`), després
  gpt-oss (`5.7`–`6.2%`); la resta rebutja poc. En igualar la n vàlida efectiva el
  rànquing es manté: la distància dels Gemini **no** és un artefacte del rebuig.
- **El desplaçament es concentra en l'eix nacional / de confiança institucional**
  (independència, identitat, confiança en institucions); la ideologia esquerra–dreta
  i l'economia són els **més estables**.
- L'alineació amb les poblacions és moderada i **no** s'ha de llegir com un
  rànquing moral.

> **Estat: preliminar.** Primer estudi sobre un conjunt curat d'ítems; els números
> tenen data d'onades concretes. Vegeu les Limitacions de l'article científic.

## 6. Com reproduir-ho

Requereix [`uv`](https://docs.astral.sh/uv/).

```bash
make install
make test
make dry-run                 # simulació amb model fictici, sense claus

uv run python scripts/build_dataset.py --source all
uv run python scripts/run_batch.py --model <id> --run   # via Batch API
uv run python scripts/summarize_results.py
uv run python scripts/validate_translations.py
uv run python scripts/directional.py
uv run python scripts/paper_figures.py && (cd paper && latexmk -pdf paper.tex)
```

Models a `models.yaml`; paràmetres a `config.yaml`. Les claus d'API es llegeixen
de l'entorn i mai no es desen al repositori.

## 7. Fonts de dades i llicència

- **CEO** — Centre d'Estudis d'Opinió, Generalitat de Catalunya. <https://ceo.gencat.cat>
- **CIS** — Centro de Investigaciones Sociológicas, Govern d'Espanya. <https://www.cis.es>

Codi sota **MIT** ([`LICENSE`](LICENSE)). Les dades d'enquesta estan subjectes als
termes del CEO i del CIS; només es redistribueixen marginals agregats amb l'estudi
d'origen citat. Dataset:
<https://huggingface.co/datasets/xaviviro/llm-political-alignment-ca-es>.

## 8. Citació

**DOI: [10.13140/RG.2.2.22319.70561](https://doi.org/10.13140/RG.2.2.22319.70561)**

Vinaixa Roselló, X. (2026). *Desplaçament polític per llengua en models de
llenguatge*. Preprint. <https://doi.org/10.13140/RG.2.2.22319.70561>

Vegeu [`CITATION.cff`](CITATION.cff). Metodologia segons Zahraei & Asgari (2025)
(MENAValues). Vegeu [`docs/references.md`](docs/references.md) per a la llista completa.
