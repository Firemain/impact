## Objectif

Construire un outil qui, à partir d’un PDF d’article scientifique, produit automatiquement :

1. Les actions évaluées (interventions, comparateurs, population, contexte)
2. Les outcomes mesurés (mesures, instruments, timepoints, groupes)
3. Les tailles d’effet disponibles ou calculables (Cohen’s d, Hedges g, SMD), avec traçabilité des sources
4. Un score de fiabilité réutilisable qui combine :

* fiabilité du calcul de la taille d’effet
* fiabilité de l’étude, via une grille de risque de biais adaptée au design
* fiabilité du papier comme source, via des signaux de reporting et cohérence

5. Un stockage structuré pour réutiliser les résultats plus tard, par outcome et timepoint (pas juste par article)

## Contraintes clés

* Tout doit être traçable. Chaque conclusion doit pointer vers des preuves (snippets texte, lignes de tableaux, page).
* Le système ne doit pas “halluciner”. Sans preuve suffisante, il doit renvoyer “inconnu” et expliquer ce qui manque.
* Le design de l’étude conditionne le reste. Si le design est mal détecté, tout le scoring devient faux.
* Les LLM doivent être utilisés pour extraction et pré-remplissage. Les décisions finales (scoring, validations de cohérence) doivent être déterministes autant que possible.

## Périmètre MVP

MVP orienté “une exécution sur un PDF” avec export JSON et mini rapport, plus indexation locale.

* Support initial : RCT parallel-group uniquement pour RoB2 V1.
* Détection de design incluse dès le départ. Si non RCT, on ne fait pas RoB2.
* Calcul d’effet : priorité aux cas calculables avec mean, SD, n.
* On prépare l’extension vers ROBINS-I, AMSTAR2, ITS, mais sans les implémenter au début.

---

# Vue d’ensemble du flow multi-agent

Le flow que tu proposes est bon. On le formalise en 8 modules principaux, chacun produisant des artefacts persistés.

1. Ingestion et parsing
2. Détecteur de design
3. Extracteur actions
4. Extracteur outcomes
5. Détecteur d’effets
6. Scoring effet
7. Scoring article
8. Scoring biais
9. Score global

Chaque étape doit produire :

* un JSON normalisé
* un ensemble de preuves (snippets, tables) liées par ID
* des logs de décision

---

# Structure d’artefacts et stockage

## Arborescence de sortie

Pour chaque PDF analysé, créer un dossier :
`outputs/{paper_id}/`

Contenu minimal :

* `00_metadata.json`
* `01_text_index.json`
* `02_tables.json`
* `03_design.json`
* `04_actions.json`
* `05_outcomes.json`
* `06_effect_candidates.json`
* `07_effects.json`
* `08_bias_evidence.json`
* `09_bias_answers.json`
* `10_bias_judgement.json`
* `11_article_signals.json`
* `12_reliability.json`
* `report.md`

## Indexation pour réutilisation

Une base locale (SQLite) avec tables :

* `papers(paper_id, title, year, doi, design, pdf_hash, created_at)`
* `results(result_id, paper_id, outcome, timepoint, comparison, effect_type, effect_value, ci_low, ci_high, calc_confidence, bias_overall, reliability_score, created_at)`
* `evidence(evidence_id, paper_id, type, page, section, text, table_id, row_ref)`

`result_id` doit être stable. Exemple : hash de `{paper_id, outcome, timepoint, comparison, analysis_set}`.

---

# Étapes détaillées, objectifs et sorties

## Étape 1. Ingestion et parsing

### Objectif

Transformer le PDF en données exploitables : texte par page, sections approximatives, tableaux si possible, plus un index.

### Entrées

* PDF (bytes ou path)

### Sorties

* `01_text_index.json` : liste de passages `{evidence_id, page, section_guess, text}`
* `02_tables.json` : tables normalisées `{table_id, page, title_guess, rows}`
* `00_metadata.json` : titre, auteurs, année si détectables, hash du PDF

### Notes d’implémentation

* Extraction texte : PyMuPDF ou pdfplumber.
* Découpage sections : heuristiques sur headings, “Methods”, “Results”, etc.
* Tables : camelot (stream) si possible, sinon vide avec statut.

---

## Étape 2. Détecteur de design

### Objectif

Classifier l’étude, car tout dépend de ça.

### Classes MVP

* `rct_parallel`
* `non_rct`
* `systematic_review_meta`
* `its_longitudinal`
* `unknown`

### Sorties

* `03_design.json` :

  * `design_label`
  * `confidence`
  * `supporting_evidence_ids`
  * `hard_blocks` (ex. “RoB2_not_applicable”)

### Heuristiques recommandées

* Chercher des marqueurs en Methods, pas dans References.
* RCT : “randomly assigned”, “allocation concealment”, “trial arms”, flow CONSORT.
* Meta-analysis : “systematic review”, “meta-analysis”, “included studies”, “RevMan”.
* ITS : “interrupted time series”, “slope change”, “level change”, “time series”.

Règle dure :

* Si `design_label != rct_parallel`, on n’exécute pas RoB2. On sort un biais “not_applicable” et on le log.

---

## Étape 3. Extracteur actions

### Objectif

Lister ce qui est évalué. Intervention(s), comparateur(s), contexte, durée, population.

### Entrées

* `01_text_index.json`
* `02_tables.json`
* `03_design.json`

### Sorties

* `04_actions.json` :

  * `population`
  * `interventions[]` avec composantes, dose/durée
  * `comparators[]`
  * `setting`
  * `inclusion_exclusion` si trouvé
  * `evidence_ids` associés

### Méthode

* Heuristiques pour isoler les sections Intervention, Participants, Procedure.
* LLM optionnel pour structurer en JSON strict, basé uniquement sur snippets.

---

## Étape 4. Extracteur outcomes

### Objectif

Identifier toutes les mesures, instruments, timepoints, et éventuellement outcomes primaires/secondaires.

### Sorties

* `05_outcomes.json` :

  * `outcomes[]` :

    * `label`
    * `instrument`
    * `timepoints[]`
    * `grouping` (groupes / bras)
    * `primary_secondary` si trouvé
    * `evidence_ids`

---

## Étape 5. Détecteur d’effets

### Objectif

Trouver où sont les tailles d’effet reportées, ou les chiffres permettant de les calculer.

### Sorties

* `06_effect_candidates.json` :

  * candidats de type :

    * `reported_effect` (d, g, SMD, ES)
    * `descriptives` (mean, SD, n)
    * `test_stats` (t, F, p, CI)
  * chaque candidat : localisation, evidence_ids, table refs

### Règles

* Priorité à Results + Tables outcomes.
* Éviter citations et discussion pour le calcul.

---

## Étape 6. Scoring effet

### Objectif

Produire les tailles d’effet finales, avec méthode, incertitude si possible, et niveau de confiance.

### Sorties

* `07_effects.json` :

  * `effects[]` :

    * `result_spec` (outcome, timepoint, comparison, groups)
    * `effect_type` (d, g, SMD)
    * `value`, `se`, `ci_low`, `ci_high` si calculable
    * `derivation_method` (reported, computed_from_means, converted_from_t, assumption_based)
    * `assumptions[]` (ex. corr pre-post)
    * `calc_confidence` (exact, assumption_based, not_derivable)
    * `evidence_ids`, `table_row_refs`

### Méthode de calcul V1

* Between-groups continuous outcome post-test :

  * calcul d ou g depuis mean, SD, n
* Si d reporté :

  * le prendre, puis tenter un recalcul si les descriptives existent, comparer l’écart

---

## Étape 7. Scoring article

### Objectif

Évaluer la qualité du papier comme source. Ce n’est pas un biais causal, c’est un score “source-quality” utile pour la réutilisation.

### Sorties

* `11_article_signals.json` :

  * signaux de reporting :

    * présence mean/SD/n
    * présence CI
    * transparence sur exclusions
    * présence CONSORT
    * présence protocole/registry id
    * clarté outcome/timepoint
  * `reporting_score` (0 à 1 ou 0 à 100)
  * preuves

Note. Ne pas faire de “prestige score” basé sur citations. C’est instable et pas nécessaire pour MVP.

---

## Étape 8. Scoring biais

### Objectif

Appliquer l’outil de risque de biais adapté au design. Dans MVP, seulement RoB2 pour RCT.

### Entrées

* `03_design.json` doit être `rct_parallel` sinon not_applicable.

### Sorties

* `08_bias_evidence.json` : bundles par domaine
* `09_bias_answers.json` : réponses aux signalling questions avec preuves
* `10_bias_judgement.json` : jugements domaines + overall

### Politique LLM

* Le LLM pré-remplit uniquement.
* Si absence de preuves, réponse = NI.
* Le jugement de domaine et overall est calculé par règles déterministes conservatrices.

---

## Étape 9. Score global

### Objectif

Donner un score final réutilisable par résultat.

### Sorties

* `12_reliability.json` :

  * par effet (result_id) :

    * `calc_score`
    * `bias_score`
    * `reporting_score`
    * `consistency_score`
    * `reliability_score_total`
    * `verdict` (High, Moderate, Low, Not usable)
    * justification courte
    * liens evidence ids

### Pondération suggérée MVP

* calc_score 30%
* bias_score 50%
* reporting_score 20%
* consistency_score inclus dans calc_score ou bonus

---

# Instructions à donner à Codex pour développer étape par étape

## Principes d’implémentation

* Chaque étape est une fonction pure autant que possible. Entrées JSON, sortie JSON.
* Chaque sortie est persistée sur disque et enregistrée en SQLite.
* Chaque champ doit avoir un type strict via Pydantic.
* Chaque extrait de preuve doit être référencé par `evidence_id`.
* Aucune étape ne doit dépendre d’une information non présente dans ses entrées. Si manquante, marquer “unknown”.

## Organisation du code recommandée

* `src/orchestrator.py` : enchaîne les étapes, gère outputs/{paper_id}/
* `src/models.py` : Pydantic models pour tous les JSON
* `src/steps/` :

  * `ingest.py`
  * `design.py`
  * `actions.py`
  * `outcomes.py`
  * `effects_detect.py`
  * `effects_compute.py`
  * `article_signals.py`
  * `rob2_evidence.py`
  * `rob2_fill.py`
  * `rob2_score.py`
  * `reliability.py`
* `src/storage/sqlite.py` : persistance

## Séquencement de dev

1. Ingestion + export text_index et tables
2. Design classifier avec preuves
3. Extracteur actions (heuristique simple)
4. Extracteur outcomes
5. Détecteur d’effets candidats
6. Calcul d’effet V1 from means
7. Article signals V1
8. RoB2 evidence + fill + score V1
9. Reliability score + report + sqlite

## Interface

MVP peut être notebook + CLI. UI Streamlit ensuite.

---

# Définition du “Done” pour chaque étape

* Étape 1 done : on obtient texte structuré et tables sur 3 PDFs différents, même si tables vides.
* Étape 2 done : le design est correct sur au moins un RCT, une meta-analysis, un ITS.
* Étapes 3 à 5 done : le système liste interventions, outcomes, et trouve des candidats d’effet.
* Étape 6 done : au moins un d/g/SMD est calculé correctement quand mean/SD/n existent.
* Étape 8 done : RoB2 renvoie un jugement conservateur, traçable, uniquement si RCT détecté.
* Étape 9 done : un score final est produit par résultat et stocké.

