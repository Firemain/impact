# Rapport Pipeline

- paper_id: `1771268049-nihms-100109-1-8037d1af`
- generated_at_utc: `2026-02-16T18:54:19.896989`

## Etapes
- `01` Ingestion et parsing: **completed** (6.775s) - Ingestion terminee: 34 pages, 109 passages, 1 tableaux.
- `02` Detection du design: **completed** (1.406s) - Design detecte: unknown (confiance 0.00)
- `03` Extraction des actions: **completed** (1.168s) - Actions extraites: 0 interventions, 2 comparateurs, population=parents
- `04` Extraction des outcomes: **completed** (1.005s) - Outcomes extraits: 5, confiance=0.80
- `05` Detection des effets: **completed** (0.092s) - Candidats d effet detectes: 30
- `05b` Typage des effets: **completed** (0.028s) - Candidats types: 30
- `06` Extraction effets normalises: **completed** (0.004s) - Effets bruts produits: 7 (derivables=7)
- `06b` Dedoublonnage canonique: **completed** (0.003s) - Effets canoniques: 7
- `06c` Check statistique: **completed** (0.012s) - Check statistique applique: failed=0/7
- `07` Signaux article: **completed** (0.035s) - Signaux article calcules: score=0.65
- `08` Score de biais: **completed** (0.008s) - RoB2: status=not_applicable, overall=not_applicable
- `09` Fiabilite globale: **completed** (0.004s) - Fiabilite calculee pour 7 resultats (score moyen=0.00)

## Synthese
- design: `unknown` (confidence=0.00)
- actions: interventions=0, comparators=2, population=parents
- outcomes: count=5, confidence=0.80
- effects: total=7, derivable=7
- article_reporting_score: 0.65
- rob2: status=not_applicable, overall=not_applicable
- reliability: items=7, mean_score=0.00
- sqlite_index: `C:\Users\87fug\Documents\centrale\impact\outputs\impact_index.sqlite`

## Choix Methodologiques
- Objectif: maximiser la tracabilite et limiter les hallucinations.
- Metadata: LLM active (premieres pages uniquement) pour corriger titre/auteurs quand l en-tete PDF est bruite.
- Extraction structuree: LLM activee sur snippets cibles uniquement, jamais sur le PDF complet.
- Mode extraction effets: strict (max_snippets=40, snippet_chars=1200).
- Design router: rules-first, fallback LLM si confiance < 0.75.
- Post-traitement effets: typage (role/level), dedoublonnage canonique, puis check statistique (CI/value).
- Effets: 7 effets produits
- Effets: extraction_mode=strict
- Effets: reported_effect regex entries: 0
- Effets: LLM skipped: no reported_effect snippets available.
- Effets: computed_from_means entries: 7
- Effets: canonical_effects=7
- Effets: raw_effects=7
- Effets: stat_check_failed=0
- Effets: stat_check_total=7
- Effets: primary_effects=0
- Effets: secondary_effects=7

## Actions Detaillees
- Population: parents
- Setting: community
- Inclusion/Exclusion: Based on this information, participants were coded as having lived together premaritally or not.
- Interventions:
  - unknown
- Comparateurs:
  - control group | dose/duree=unknown | preuves=3
  - usual care | dose/duree=unknown | preuves=1

## Effets Extraits (d / g / SMD)
- Action de reference: intervention vs control group
| result_id | primaire | role | grouping | outcome_norm | timepoint_norm | effect_type | value | CI95 | stat_check | dedup | origine/methode | fiabilite |
|---|---|---|---|---|---|---:|---:|---|---|---:|---|---|
| `res_4ee1a410df789624` | False | unclear | unknown | relationship_satisfaction | baseline | g | -136.102 | [-140.308; -131.895] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_a70abd79d9ef7787` | False | unclear | unknown | relationship_satisfaction | baseline | g | 0.000 | [-1.600; 1.600] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_79bd33f4a2e04623` | False | unclear | unknown | relationship_satisfaction | baseline | g | 1.159 | [0.123; 2.195] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_1c1dc34c44e5642d` | False | unclear | unknown | relationship_satisfaction | baseline | g | 0.037 | [-0.944; 1.018] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_ce53ea07429885c8` | False | unclear | unknown | relationship_satisfaction | baseline | g | -773.036 | [-917.501; -628.571] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_923c032695a6fb10` | False | unclear | unknown | relationship_satisfaction | baseline | g | 0.416 | [-0.169; 1.001] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |
| `res_9d6ed9ccb3a032b6` | False | unclear | unknown | relationship_satisfaction | baseline | g | -0.001 | [-0.062; 0.061] | pass | 1 | computed/computed_from_means/exact | 0.00 (Not usable) |

## Fichiers JSON
- 00_ingest_logs.json
- 00_metadata.json
- 01_text_index.json
- 02_tables.json
- 03_design.json
- 04_actions.json
- 05_outcomes.json
- 06_effect_candidates.json
- 06b_effect_typing.json
- 07_effects.json
- 07_effects_canonical.json
- 07_effects_raw.json
- 07_effects_secondary.json
- 07_effects_stat_check.json
- 08_bias_evidence.json
- 09_bias_answers.json
- 10_bias_judgement.json
- 11_article_signals.json
- 12_reliability.json
