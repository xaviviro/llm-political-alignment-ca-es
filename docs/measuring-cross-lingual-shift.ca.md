# Mesurar el biaix polític entre llengües — enfocament i justificació

> 🌐 [English](measuring-cross-lingual-shift.md) · **Català**

Aquest document explica *per què* el marc mesura el biaix polític com ho fa:
quina pregunta ens fem realment, quins instruments hem rebutjat i per què, en
quina literatura recent ens basem (inclosa la feta en llengües altres que
l'anglès) i els límits honestos del test que hem adoptat.

## 1. La pregunta que ens fem

**No** preguntem "el model X és d'esquerres o de dretes". Aquest enquadrament és
justament el que el camp està abandonant (§3). Fem una pregunta més estreta i
falsable:

> Canvia la **distribució de respostes d'un model a preguntes d'enquesta reals
> en funció de la llengua amb què se'l consulta** (anglès vs català vs castellà),
> i com de lluny està aquesta distribució de la **població humana real**?

Dues propietats ho fan tractable i rigorós:

- **Un referent real.** Cada ítem es compara amb una distribució poblacional
  mesurada d'una enquesta oficial — CEO (Catalunya) i CIS (Espanya) — no amb un
  origen abstracte esquerra/dreta.
- **La llengua com a variable d'interès.** El mateix ítem, la mateixa població,
  preguntat en tres llengües: qualsevol moviment és el *desplaçament induït per
  la llengua*, el fenomen que ens importa.

## 2. Per què la llengua, i què ha trobat la resta de feina

La troballa que **la llengua de la indicació mou la postura política elicitada d'un
model** ja s'ha replicat en diversos grups de recerca i famílies lingüístiques:

- **Bias Beyond Borders: Political Ideology Evaluation and Steering in
  Multilingual LLMs** (2026, arXiv:2601.23001) — l'anglès és sistemàticament el
  més libertari d'esquerres; altres llengües viren cap al centre/dreta.
- **Multilingual Political Views of Large Language Models: Identification and
  Steering** (2025, arXiv:2507.22623) — orientació ideològica dependent de la
  llengua en moltes llengües.
- **Do Political Opinions Transfer Between Western Languages?** (2025,
  arXiv:2508.05553) — l'alineament fet en anglès es propaga a altres llengües
  (efectes de transferència).
- **Framing Political Bias in Multilingual LLMs Across Pakistani Languages**
  (2025, arXiv:2506.00068) — l'efecte fora de les llengües occidentals.
- **Analyzing Political Bias in LLMs via Target-Oriented Sentiment
  Classification** (Elbouanani et al., 2025, arXiv:2505.19776) — biaixos més
  pronunciats en anglès/castellà/francès que en àrab/xinès/rus.
- **Assessing the Political Fairness of Multilingual LLMs: un conjunt de dades
  multiparal·lel d'EuroParl de 21 vies** (2025, arXiv:2510.20508) — comparació
  controlada entre llengües europees.

En aquest context, les troballes recurrents centrades en l'anglès (ChatGPT i
companyia tirant cap a libertari d'esquerres: Hartmann et al. 2023; Rozado 2024,
*PLOS ONE*; Motoki, Pinho Neto i Rodrigues 2024, *Public Choice*) són **una
llesca d'una sola llengua** d'un fenomen multilingüe.

**El català és pràcticament absent d'aquesta literatura; el castellà hi apareix
només com una llengua més.** Un estudi que posi el català i el castellà al
centre, ancorat a les poblacions que els parlen de debò (CEO, CIS), és el buit
que perseguim.

## 3. Per què vam descartar el Political Compass

El Political Compass Test (i el conegut quadrant de dos eixos) és el recurs per
defecte. El vam rebutjar com a **instrument principal**:

- **Röttger et al. (2024), *Political Compass or Spinning Arrow? Towards More
  Meaningful Evaluations for Values and Opinions in LLMs*, ACL 2024** és la
  crítica decisiva: el resultat és **fràgil al format** — forçar una opció
  múltiple vs permetre generació oberta capgira el resultat, i petits canvis de
  la indicació mouen l'agulla. La "posició al compass" sovint és un artefacte d'una
  redacció.
- **Feng et al. (2023), ACL** van mostrar que els eixos es poden sondejar, però
  els eixos mateixos són un **origen ideològic abstracte** sense referent
  poblacional: "distància al centre del compass" no és distància a cap grup humà
  real.
- **Problema de constructe.** Una sola elecció forçada per proposició, puntuada
  amb pesos propietaris, barreja moltes coses en dos números i amaga l'efecte de
  la llengua que volem aïllar.

Per tant no fem servir el compass com a mesura. Com a molt sobreviu com a
*visualització familiar*, mai com a veritat de referència.


## 4. El que hem adoptat: MENAValues, adaptat al CEO + CIS

Ens basem en **Zahraei & Asgari (2025), *I Am Aligned, But With Whom? MENA Values
Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs*,
arXiv:2510.13154**, i l'adaptem de la regió MENA a Catalunya i Espanya.

### 4.1 El mètode que heretem

- Comparar la **distribució de respostes** d'un model amb una **distribució
  poblacional real** d'enquestes de referència.
- Creuar **enquadraments de perspectiva** (neutral / personalitzat / observador
  en tercera persona) amb **modes de llengua**.
- Caracteritzar modes de fallada: **fuita de logits** (el text rebutja mentre la
  massa de probabilitat interna és esbiaixada), **determinisme lingüístic /
  desplaçament de valors entre llengües** (les respostes col·lapsen segons la
  llengua de la indicació) i **degradació induïda pel raonament**.

### 4.2 La nostra adaptació

- **Veritat de referència = CEO (Catalunya) + CIS (Espanya)**, marginals reals,
  no enquestes de MENA. Ítems: independència, identitat nacional, ideologia,
  monarquia i model d'estat del CEO; la valoració de l'economia d'Espanya i la
  ideologia esquerra–dreta del CIS — cadascun amb l'onada exacta, la URL de la
  font i la data d'accés, distribuint **només marginals agregats**.
- **Llengües = anglès / català / castellà.**
- Mètriques: `alineació = 1 − JSD(model, població)`, **consistència translingüe**
  i **robustesa entre plantilles** (vegeu §5.4).

### 4.3 Què diu aquest test — i què no

**Diu:** com de a prop està la distribució de respostes d'un model d'una
*població humana concreta*, i si aquesta proximitat **canvia amb la llengua de la
indicació**. És empíric i ancorat, no un eix abstracte.

**No diu:**

- **No és un rànquing moral.** "Més a prop de la població del CEO/CIS" no vol dir
  "millor" ni "menys esbiaixat" en cap sentit normatiu.
- **Coincidir amb la població ≠ no tenir biaix.** Sota els enquadraments
  *observador* i *personalitzat* se li demana en part que **prediga** la població
  — així que una alineació alta allà s'assembla més a una puntuació de predicció
  que a una afirmació dels valors propis del model. (A les nostres pròpies
  execucions l'enquadrament observador alinea més, just com prediu aquest advertiment.)
- Mesura **representativitat i estabilitat lingüística**, no correcció.

### 4.4 El seu rigor — un veredicte honest

Hem llegit l'article sencer de MENAValues. La nostra valoració:

**Enginyeria empírica sòlida:** intervals de confiança de remostreig (bootstrap) del 95% a tot
arreu (B=1.000); una veritat de referència gran i real (864 ítems del World Values
Survey Wave 7 i l'Arab Opinion Index 2022, amb pesos de post-estratificació);
traduccions validades per humans; definicions de mètrica transparents i a nivell
de fórmula (NVAS, CLCS, un llindar del 75% del màxim log-prob per a la fuita de
logits, divergència KL).

**Més fluix on importa per a la inferència:**

- **Cap test de significació ni control de comparacions múltiples** sobre la gran
  superfície models × països × enquadraments — les marques direccionals ↑↓
  arrisquen sobre-llegir soroll (els IC ho mitiguen només en part).
- **Cap robustesa de plantilles** — creua enquadraments però **no** varia la
  redacció *dins* d'un enquadrament, així que la mateixa fragilitat de Röttger
  contra la qual es posiciona no està controlada.
- **La validesa de constructe no s'examina** — l'enquadrament persona/observador
  fa l'alineació alta en part tautològica, i la secció de limitacions no ho
  aborda.
- És una **prepublicació (preprint) sense revisar**.

### 4.5 On la nostra adaptació és més forta, i on comparteix els forats

- **Més forta:** afegim **robustesa de plantilles** (N paràfrasis per
  enquadrament, reportant la SD entre plantilles) — tancant el forat de Röttger
  que MENAValues deixa obert; fem **explícites les fallades d'anàlisi (parse-fails)** (una resposta on
  totes les mostres fallen es marca invàlida i s'exclou, mai tractada en silenci
  com un uniforme "no ho sé"); i adjuntem **procedència completa** (onada + font
  + data d'accés) a cada ítem, amb IC de remostreig (bootstrap) a cada número reportat.
- **Forats compartits:** un conjunt petit d'ítems és preliminar; l'advertiment de
  validesa de constructe (§5.3) també ens aplica; i l'anàlisi de fuita de logits
  només és possible en proveïdors que exposen log-probabilitats de tokens
  (OpenAI, Gemini, vLLM/llama.cpp locals), no en les APIs de xat tancades de què
  mostregem.

## 5. Què vol dir "rigor" per a nosaltres a la pràctica

- Reportar **representativitat, no rànquings**; mostrar el solapament d'IC en
  lloc d'ordenacions pelades.
- Reportar la **SD entre plantilles** perquè un resultat sensible a la redacció
  sigui visible.
- Mantenir les categories **No ho sap / No contesta** fidels a l'enquesta i ser
  explícits sobre com es gestionen.
- Tractar 5–7 ítems com a **preliminars**; els números tenen data d'una onada
  d'enquesta.
- No presentar mai un marcador il·lustratiu com una mesura real; el carregador
  propaga l'indicador `source_status` fins a l'informe.

## Referències

Vegeu [`references.md`](references.md) per a les fonts metodològiques principals
(MENAValues, Röttger, Feng) i els proveïdors de dades d'enquesta.
