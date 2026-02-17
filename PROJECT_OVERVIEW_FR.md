# Impact - Explication Projet (Vision, Methode Agentique, Technique)

## Objectif (point de vue produit)
L'objectif est de transformer un article PDF (souvent peu structure) en une sortie exploitable pour la decision:

- des **effets standardises fiables** (d / g / SMD),
- traces a leur **evidence source** (quote, page, type de source),
- plus une lecture **qualite interne + credibilite externe + score global**.

L'idee n'est pas seulement d'extraire, mais de **trier, normaliser, consolider et fiabiliser** les resultats pour eviter le bruit (doublons, formulations incoherentes, faux positifs statistiques) afin d'obtenir un score de fiabilité.

---

## Worflow complet en blocs

| Bloc | Role principal | Entrees | Sorties | Points techniques |
|---|---|---|---|---|
| Bloc 1 Ingestion | Structurer PDF en evidences exploitables | PDF | `00_metadata.json`, `01_text_index.json`, `02_tables.json` | Chunking avec overlap, section guess, rendu image des pages table |
| Bloc 2 Block Router | Ne garder que les blocs utiles | Text passages | `03_block_flags.json` | Regex + LLM optionnel, fallback heuristique |
| Bloc 3 Effects Engine | Produire effets d/g/SMD fiables | Blocs effets + tables | `04_effects.json` | Multi-source extraction + agents 2/3/4 |
| Bloc 4 Qualite interne rapide | Score methodologique rapide | Blocs qualite | `05_quality_quick.json` | Grille 5 criteres |
| Bloc 5 Credibilite externe | Contextualiser article (venue/citations) | Metadata | `06_external_credibility.json` | OpenAlex + scoring |
| Bloc 6 Score final | Synthese fiabilite | Effets + qualite + credibilite | `07_summary_score.json`, `12_reliability.json` | Ponderation explicite |

---
## Explications détaillées blocs par blocs

### Bloc 1 - Ingestion (niveau algorithmique)

Objectif technique: passer d'un PDF brut a un graphe d'evidences adressees (chunk + page + section + id).

#### 1) Initialisation document
- Input: chemin PDF.
- Actions:
  - verifier existence et extension,
  - calculer `pdf_hash`,
  - definir `paper_id` (manuel ou derive du nom + hash),
  - creer dossier de sortie `outputs/<paper_id>`.

#### 2) Extraction texte page par page
- On construit `pages_text` (liste ordonnee des pages).
- Strategie:
  - parseur A,
  - fallback parseur B si echec.
- Invariant: `pages_text[i]` correspond a la page `i+1`.

#### 3) Detection TOC (sommaire) pour aider le sectioning
- Scan des 12 premieres pages max.
- Si une page contient des marqueurs TOC, on extrait des lignes type:
  - `Titre .... 12`
- Ces titres deviennent un dictionnaire d'ancrage pour reconnaitre des headings plus tard.

#### 4) Segmentation page -> blocs -> chunks
- Pour chaque page:
  - split en lignes,
  - detection headings avec regles:
    - longueur plausible,
    - pas `Figure/Table/Copyright`,
    - pas "ligne numerique pure",
    - match aliases section (abstract/results/methods...),
    - match patterns numerotes,
    - ou ratio MAJ eleve (titre probable),
    - ou similarite forte avec TOC.
  - creation des blocs entre headings.
- Chaque bloc est ensuite chunke:
  - `safe_overlap = clamp(chunk_overlap, 0, chunk_chars-1)`,
  - `step = chunk_chars - safe_overlap`,
  - on evite de couper au milieu d'un mot (recul de la borne de fin).
- Chaque chunk recoit:
  - `evidence_id = sha1(paper_id|page|section|block_idx|chunk_idx|text[:240])`.

#### 5) Extraction tables structurees
- Pipeline:
  1. tentative extraction tables structurees,
  2. fallback extracteur secondaire si vide/echec.
- Une table est consideree "structuree" si:
  - >= 2 lignes,
  - au moins une ligne avec >= 3 cellules non vides,
  - au moins 2 lignes avec >= 2 cellules non vides.

#### 6) Detection des pages tableau et voie image
- On detecte les pages candidates via regex `Table <num/roman>`.
- `pages_for_images = marker_pages - pages_deja_structurees`.
- Pour ces pages:
  - rendu PNG haute lisibilite,
  - ajout/mise a jour d'objets table avec `status=image_only` et `image_path`.
- But: ne pas perdre les tableaux non parseables en structure.

#### 7) Metadata
- Base heuristique locale (titre, auteurs, annee, DOI).
- Option enrichissement GROBID.
- Option refinement LLM metadata (premieres pages uniquement).

Pseudo-code simplifie:

```text
pdf -> pages_text
pages_text -> section blocks -> chunks -> evidence_ids
pdf -> structured_tables
pages_text -> marker_pages
marker_pages - structured_pages -> table_page_png
metadata = heuristics (+ optional enrichments)
write 00/01/02
```

### Bloc 2 - Classification (Block Router) deterministic

Objectif technique: filtrer les evidences pour ne garder que les passages pertinents.

#### 1) Scoring heuristique par chunk
Pour chaque chunk:
- `reference_like` si:
  - section `references`, ou
  - patterns bibliographiques, ou
  - beaucoup de `et al.` + annees.
- `results_candidate` si presence de:
  - markers table/figure,
  - stats tokens (p, t, F, chi2, CI...),
  - ou pattern effet.

#### 2) Construction flags
- Flags booleens:
  - `contains_results`
  - `contains_effect_size`
  - `contains_methods`
  - `contains_population`
  - `contains_bias_info`
- Confidence heuristique:
  - base 0.45,
  - +0.2 si effet,
  - +0.1 si resultats,
  - +0.1 si methods,
  - cap a 0.95.

#### 3) LLM optionnel (sur sous-ensemble)
- On envoie uniquement les `results_candidate`.
- Retour JSON booleen.
- Fusion finale LLM + heuristiques (OR + garde-fous references).

#### 4) Sorties de routage
- `relevant_effect_blocks`: priorite aux blocs avec effet; fallback blocs resultats.
- `relevant_quality_blocks`: methods ou bias.

### Bloc 3 - Effects Engine deterministic (avec options LLM)

Objectif technique: produire des effets propres, dedupliques et exploitables.

#### Agent 1 - Extractor

1) Deterministe texte:
- detecte patterns `d/g/SMD`,
- parse valeur + CI,
- infere predictor/groupe/timepoint/contexte,
- classe scope (`study_effect`, `literature_cited`, `model_stat`).

2) Deterministe table structuree:
- detecte colonnes effet via headers autorises (`d`, `cohen d`, `hedges g`, `SMD`),
- rejette headers bannis (`B`, `SE`, `SD`, `r`, ...),
- extrait valeurs par ligne.

3) Regles fortes:
- si `abs(value) > 1.5` => `model_stat` par defaut,
- contexte "model parameters" => `model_stat`.

4) LLM optionnels:
- texte: snippets des blocs effet, JSON strict,
- vision: images `image_only` encodees base64, JSON strict.

#### Agent 2 - Normalizer

Deterministic local:
- normalise `effect_type`,
- ordonne CI (`ci_low <= ci_high`),
- nettoie quote (taille/format),
- harmonise `Group/Domain/Predictor/Timepoint`,
- aligne `comparison` si manquant.

LLM-first optionnel:
- batch indexe des effets deja extraits,
- autorise seulement la normalisation de labels,
- interdit de changer `value/effect_type/scope`.

#### Agent 3 - Consolidator

1) Clustering:
- tri initial par qualite de source/quote,
- insertion dans un cluster existant si compatibilite:
  - meme scope,
  - meme type effet,
  - `|delta value| <= 0.01`,
  - predictor compatible,
  - groupe/timepoint/domain compatibles.

2) Merge cluster:
- valeur representative = mediane/mediane moyenne,
- groupe/predictor/domain/timepoint les plus informatifs,
- merge evidence_ids + notes + refs,
- marque `consolidated_cluster=n` si fusion.

3) Priorite source:
- `table` > `text deterministic` > `text LLM` > `table_image_vision`.

#### Agent 4 - Validator

Regles:
- drop si valeur absente,
- reclasser `study_effect` en `model_stat` si `|value| > 1.5`,
- drop `study_effect` si `group` ou `predictor` critiques en `unknown`,
- drop si quote incomplete (heuristiques de qualite de citation),
- combler `comparison` avec `group` si necessaire.

Resultat:
- liste finale triee par `abs(value)` decroissante.

### Bloc 4 - Qualite interne rapide deterministic

Objectif: score methodologique rapide, interpretable, peu couteux.

#### 1) Signaux detectes
- randomization
- control_group
- sample_size_reported
- attrition_reported
- blinding_reported

#### 2) Regles locales
- regex sur les `relevant_quality_blocks`.
- score:
  - `yes = +0.2`
  - `unclear = +0.1`
  - `no = +0.0`
- score borne dans `[0,1]`.

#### 3) Option LLM
- meme schema, aggregation multi-batch,
- fallback heuristique si LLM indisponible.

### Bloc 5 - Credibilite externe deterministic

Objectif: evaluer la credibilite de l'article via signaux bibliometriques.

#### 1) Resolution OpenAlex
- essai DOI direct,
- sinon recherche titre + auteur + filtre annee.

#### 2) Selection resultat
- overlap lexical titre,
- bonus annee,
- seuil minimum de fiabilite.

#### 3) Score deterministic
- +0.4 si match titre/oeuvre,
- +0.1 / +0.2 / +0.3 / +0.4 selon palier citations,
- +0.2 si venue connue,
- clamp `[0,1]`.

#### 4) Option LLM
- ajuste/valide le score sur metadata resolue.

### Bloc 6 - Score final deterministic

Objectif: produire une fiabilite par effet puis globale.

#### 1) Score par effet
- `calc_score` (qualite intrinsique du coefficient),
- `internal_quality_score` (bloc 4),
- `external_score` (bloc 5).

Formule:

```text
reliability_total = 0.5 * calc_score
                  + 0.3 * internal_quality_score
                  + 0.2 * external_score
```

#### 2) Verdict par effet
- High / Moderate / Low / Not usable selon seuils.

#### 3) Score global
- moyenne des effets utilisables,
- conclusion finale (`utilisable`, `utilisable avec prudence`, `non utilisable`).

## 3quinquies. Deep dive technique (meme niveau de detail pour blocs 2 a 6)

### Bloc 2 - Classification: logique exacte de decision

#### Input reel
Chaque entree est un chunk de `01_text_index.json` avec:
- `evidence_id`
- `page`
- `section_guess`
- `text`

#### Pipeline deterministic
1. Normalisation du texte (espaces, casse, ponctuation OCR).
2. Calcul de `reference_like`:
   - `section_guess == references`, ou
   - regex bibliographie, ou
   - pattern "beaucoup de et al. + annees".
3. Calcul de `results_candidate`:
   - regex table/figure, ou
   - regex stats (p/t/F/chi2/CI/SE/BIC/log-likelihood), ou
   - regex effet (d/g/SMD).
4. Construction des 5 flags booleens.
5. Calcul d'une confidence heuristique:
   - base 0.45,
   - +0.2 si effet,
   - +0.1 si resultats,
   - +0.1 si methods,
   - borne max 0.95.
6. Si `reference_like`, on force `contains_results=False`.

#### Branche LLM optionnelle
1. On envoie uniquement les chunks `results_candidate`.
2. Le LLM renvoie les 5 booleens + confidence.
3. Merge final avec les regles:
   - OR logique sur les booleens,
   - garde-fou deterministic (references restent neutralisees).

#### Sorties de routage
- `relevant_effect_blocks = contains_effect_size`
- fallback si vide: `contains_results` (limite `[:80]`)
- `relevant_quality_blocks = contains_methods or contains_bias_info` (limite `[:120]`)

Pseudo-code:

```text
for chunk in text_index:
  ref = is_reference_like(chunk)
  cand = has_results_pattern(chunk) and not ref
  flags = heuristic_flags(chunk, cand, ref)
  if use_llm and cand:
    flags = merge(flags, llm_flags(chunk))
build relevant_effect_blocks / relevant_quality_blocks
```

### Bloc 3 - Effects Engine: logique exacte de bout en bout

#### Etape A - Agent 1 Extractor (multi-sources)

Source 1: texte deterministe
- regex `effect_type` autorise: `d`, `g`, `SMD`.
- parse de la valeur numerique.
- parse CI si presente.
- infere:
  - predictor: contexte autour du pattern effet,
  - group: termes population/comparaison,
  - timepoint: baseline/pre/post/follow-up,
  - scope: `study_effect` vs `literature_cited` vs `model_stat`.

Source 2: table structuree deterministic
- detection des colonnes effet dans les 3 premieres lignes:
  - headers autorises: `d`, `cohen d`, `hedges g`, `SMD`.
  - headers bannis: `B`, `SE`, `SD`, `r`, etc.
- extraction ligne par ligne.
- scope `model_stat` si contexte modele ou `abs(value) > 1.5`.

Source 3: LLM texte (optionnel)
- input: snippets JSON indexes sur `relevant_effect_blocks`.
- sortie attendue: liste stricte d'effets.
- filtre post-LLM:
  - `effect_type` valide,
  - valeur numerique valide,
  - quote coherent avec le snippet.

Source 4: LLM vision table (optionnel)
- input: images PNG de tables `image_only` encodees base64.
- sortie attendue: effets avec header/value/group/predictor/domain.
- filtre post-LLM:
  - header compatible avec `d/g/SMD`,
  - valeur numerique,
  - nettoyage labels/scope.

#### Etape B - Agent 2 Normalizer

Deterministic local:
- harmonisation `effect_type`, CI, quote, `Group/Domain/Predictor/Timepoint`.
- correction `comparison` si manquant.
- normalisation des tokens `unknown`.

LLM-first optionnel:
- input: batch d'effets deja extraits (pas de recalc des valeurs).
- contrainte: LLM ne peut normaliser que les labels.
- reject implicite des updates incoherentes.

#### Etape C - Agent 3 Consolidator

1. Tri initial par qualite (source + quote).
2. Clustering incremental:
   - meme `effect_scope`,
   - meme `effect_type`,
   - `|delta value| <= 0.01`,
   - predictor compatibles (cle simplifiee),
   - groupes/timing/domain compatibles.
3. Merge de cluster:
   - valeur representative = mediane (ou moyenne des 2 centraux),
   - selection du meilleur label de groupe/predictor/domain/timepoint,
   - fusion `evidence_ids`, `notes`, `table_row_refs`,
   - note `consolidated_cluster=n`.

Priorite source deterministic:
- `table` (0)
- `text deterministic` (1)
- `text LLM` (2)
- `table_image_vision` (3)

#### Etape D - Agent 4 Validator

Regles de rejet/reclassement:
- drop si `value` absente,
- reclasser `study_effect -> model_stat` si `abs(value) > 1.5`,
- drop `study_effect` si `group` ou `predictor` critiques sont `unknown`,
- drop si quote incomplete (heuristique fin de phrase, longueur minimale).

Sortie:
- liste finale triee par magnitude `abs(value)` decroissante.

### Bloc 4 - Qualite interne rapide: detail deterministic

#### Input reel
- subset `relevant_quality_blocks`.

#### Regles de signal
- `randomization` via regex randomisation.
- `control_group` via regex comparator/control.
- `sample_size_reported` via pattern `n = ...`.
- `attrition_reported` via attrition/dropout/lost to follow-up.
- `blinding_reported` via blind/blinding.

#### Scoring deterministic
- chaque axe:
  - yes = 0.2
  - unclear = 0.1
  - no = 0.0
- somme des 5 axes, clamp `[0,1]`.

#### Si LLM actif
- LLM score par batch + evidence ids.
- aggregation flags:
  - si au moins un `yes` => `yes`,
  - sinon si au moins un `no` => `no`,
  - sinon `unclear`.
- fallback deterministic si batches LLM invalides.

### Bloc 5 - Credibilite externe: detail deterministic

#### Input reel
- metadata article: titre, DOI, auteurs, annee.

#### Resolution OpenAlex
1. Tentative DOI direct.
2. Sinon recherche texte (titre + 1er auteur + annee).
3. Scoring local des candidats par overlap lexical titre + bonus annee.
4. Seuil de fiabilite minimal pour accepter un match.

#### Features sorties
- `title_match_found`
- `venue`
- `publisher`
- `citation_count`
- `authors_found`

#### Score deterministic
- +0.4 si match trouve,
- citations:
  - >=200: +0.4
  - >=50: +0.3
  - >=10: +0.2
  - >=1: +0.1
- +0.2 si `venue` non inconnue.
- clamp `[0,1]`.

#### Mapping niveau
- `High` >= 0.75
- `Moderate` >= 0.45
- `Low` > 0
- `Unknown` sinon

### Bloc 6 - Score final: detail deterministic

#### Score par effet (`calc_score`)
- base 0.5 si valeur presente,
- +0.25 si `effect_type in {d,g,SMD}`,
- +0.15 si CI presente,
- +0.1 si `stat_consistency == pass`,
- -0.25 si `calc_confidence == not_derivable`,
- clamp `[0,1]`.

#### Fiabilite totale par effet

```text
total = 0.5 * calc_score
      + 0.3 * internal_quality_score
      + 0.2 * external_score
```

#### Verdict par effet
- `High` si total >= 0.75
- `Moderate` si total >= 0.55
- `Low` si total >= 0.4
- `Not usable` sinon

#### Agregation globale
- moyenne des `reliability_score_total` des effets != `Not usable`.
- conclusion:
  - `utilisable` si global >= 0.65
  - `utilisable avec prudence` si global >= 0.45
  - `non utilisable` sinon

---

## 4. Technologies utilisees

### Stack
- Python
- Streamlit (UI)
- Pydantic v2 (schemas stricts)
- requests (API HTTP)
- python-dotenv
- sqlite3 (index local)

### Parsing PDF
- PyMuPDF (`fitz`) pour texte + rendu image
- pdfplumber pour tables
- camelot (fallback tables, optionnel selon env)

### APIs externes
- OpenAI Chat Completions (JSON strict, texte + vision)
- OpenAlex (credibilite externe)
- GROBID (optionnel) pour enrichissement metadata/structure

---

## 5. Algorithmes et logique de traitement

### Extraction / Classification
- Regex de detection effets/statistiques/methodes/population/biais.
- Filtre references/bibliographie pour limiter faux positifs.
- Prompting JSON strict pour controler les sorties LLM.

### Normalisation / Mapping
- Normalisation de texte (NFKC, nettoyage espaces, decimal comma).
- Mapping `Group / Domain / Predictor` via heuristiques + LLM.
- Domain contraint a:
  - `Intra-personal`
  - `Extra-personal`
  - `Material environment and education`
  - `Socio-political environment`
  - `Work and Activities`
  - `Health`
  - `unknown`

### Consolidation
- Clustering par compatibilite semantique + tolerance numerique.
- Fusion evidence_ids / notes / table_row_refs.
- Selection du meilleur representant par priorite source + qualite de quote.

### Scoring final
- Score par effet:
  - `0.5 * calc_score + 0.3 * internal_quality + 0.2 * external_score`
- Conclusion globale:
  - `utilisable`, `utilisable avec prudence`, `non utilisable`.

---

## 6. Ce qui est deja implemente vs ce qui manque

### Deja implemente
- Pipeline complet 6 blocs.
- Agentic effects engine (extractor/normalizer/consolidator/validator).
- Traceabilite evidence (quote/page/source/evidence_ids).
- Score qualite interne rapide.
- Credibilite externe OpenAlex.
- Checks de coherence de sortie (`quality_checks.py`).

### Pas encore implemente (ou partiel)
- Evaluation **Risk of Bias** complete type ROB2/ROBINS-I domaine par domaine.
- Verification statistique approfondie (`stat_consistency` est majoritairement `unknown`).
- Resolution explicite des contradictions inter-resultats (au-dela du dedup).
- Conversions methodologiques avancees (t/F/r/B -> d) avec preuve suffisante.
- Suite de tests automatises significative (dossier `tests/` quasi vide).
- Calibration quantitatives sur benchmark annote (precision/recall/F1 formelles).

---

## 8. Proposition de roadmap technique (court terme)

1. Ajouter un module `Bias Agent` explicite (domaines + judgement + evidence).
2. Ajouter un module `Contradiction Agent` (meme predictor/groupe/timing, signe oppose).
3. Instrumenter des metrics offline (precision dedup, precision extraction, faux positifs model_stat).
4. Ajouter des tests unitaires/integration sur cas PDF representatifs.
5. Exposer des toggles UI fins (ex: activer/desactiver table vision, agent2 LLM, seuils dedup).




## 4. Agents de l'engine effets (Bloc 3)

### Agent 1 - Extractor
- Extrait large depuis:
  - texte deterministe (regex d/g/SMD),
  - tables structurees (headers d/g/SMD),
  - LLM texte (optionnel),
  - LLM vision sur tables image_only (optionnel).
- Objectif: **rappel** (ne pas rater un effet candidat).

### Agent 2 - Normalizer (LLM-first)
- Standardise:
  - `group` (ex: `mothers` -> `Mothers`),
  - `predictor` (ex: `observed negative communication` -> `Negative Communication`),
  - `timepoint` (ex: `post-birth vs pre-birth` -> `transition effect`),
  - `domain` dans une taxonomie contrainte.
- Si LLM indisponible: fallback deterministe.

### Agent 3 - Consolidator (dedup)
- Regroupe les candidats proches:
  - meme scope/type,
  - meme variable/predictor proche,
  - groupe/timing/domain compatibles,
  - valeur compatible avec tolerance `+/- 0.01`.
- Conserve une entree unique, avec priorite source:
  - `table` > `text deterministe` > `text LLM` > `table_image_vision`.
- Fusionne evidences, notes et meilleure quote.

### Agent 4 - Validator
- Retire les entrees faibles/ambigues:
  - champs critiques `unknown`,
  - citation insuffisante/tronquee.
- Reclasse `study_effect` en `model_stat` si `|value| > 1.5`.

---

## 9. Slide prete - Effect Engine

### Contenu slide (version compacte)

Titre:
- `Effects Engine: de l'extraction brute a un signal fiable`

Message cle:
- `On maximise le rappel au debut, puis on impose de la coherence et de l'unicite avec 4 agents.`

Schema:

```text
Input (effect_blocks + tables)
    |
    v
Agent 1 Extractor (large recall)
  - texte regex d/g/SMD
  - tables structurees
  - LLM texte (optionnel)
  - LLM vision table_image_vision (optionnel)
    |
    v
Agent 2 Normalizer (uniformisation)
  - labels Group/Domain/Predictor/Timepoint
  - format numerique et CI
    |
    v
Agent 3 Consolidator (dedup)
  - cluster par scope/type/valeur(+/-0.01)/contexte
  - priorite source: table > text_det > text_llm > vision
    |
    v
Agent 4 Validator (qualite finale)
  - suppression inconnus/incomplets
  - reclassement model_stat si |value| > 1.5
    |
    v
Output: effets uniques, traces (quote/page/source), exploitables pour scoring
```

Regles critiques a afficher:
- `Tol. dedup: |delta value| <= 0.01`
- `Hard guardrail: |effect| > 1.5 => model_stat`
- `Sortie finale: 1 ligne = 1 effet consolide + evidences fusionnees`

KPIs utiles a mettre en bas de slide:
- `raw_candidates` (avant normalisation)
- `after_dedup` (apres agent 3)
- `final_validated` (apres agent 4)
- `dedup_gain = 1 - after_dedup/raw_candidates`

Limites actuelles (honnette et technique):
- Normalisation semantique encore sensible aux formulations tres ambigues.
- Conflits scientifiques reels (effets opposes) pas encore traites par un agent dedie.
- `stat_consistency` partiellement renseigne (`unknown` frequent).

### Notes speaker (60-90 sec)
- `Notre moteur d'effets est volontairement en 2 phases: rappel large puis reduction forte du bruit.`
- `Agent 1 capte tout ce qui ressemble a un effet, depuis texte, tables parsees et tables en image.`
- `Agent 2 harmonise les labels pour eviter que "mothers" et "Mothers" deviennent 2 variables differentes.`
- `Agent 3 est le coeur de la robustesse: il fusionne les doublons avec une tolerance numerique et une logique de compatibilite contextuelle.`
- `Agent 4 applique des garde-fous methodologiques pour ne garder que des effets exploitables dans le scoring final.`
