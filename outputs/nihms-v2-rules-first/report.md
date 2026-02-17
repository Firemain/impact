# Rapport Pipeline

- paper_id: `nihms-v2-rules-first`
- generated_at_utc: `2026-02-16T19:08:29.085227`

## Etapes
- `01` Ingestion et parsing: **completed** (2.843s) - Ingestion terminee: 34 pages, 109 passages, 1 tableaux.
- `02` Detection du design: **completed** (0.058s) - Design detecte: unknown (confiance 0.00)
- `03` Extraction des actions: **completed** (0.084s) - Actions extraites: 0 interventions, 2 comparateurs, population=parents
- `04` Extraction des outcomes: **completed** (0.039s) - Outcomes extraits: 5, confiance=0.80
- `05` Detection des effets: **completed** (0.085s) - Candidats d effet detectes: 35
- `05b` Typage des effets: **completed** (0.021s) - Candidats types: 35
- `06` Extraction effets normalises: **completed** (0.005s) - Effets bruts produits: 14 (derivables=14)
- `06b` Dedoublonnage canonique: **completed** (0.001s) - Effets canoniques: 11
- `06c` Check statistique: **completed** (0.003s) - Check statistique applique: failed=0/11
- `07` Signaux article: **completed** (0.018s) - Signaux article calcules: score=0.65
- `08` Score de biais: **completed** (0.003s) - RoB2: status=not_applicable, overall=not_applicable
- `09` Fiabilite globale: **completed** (0.001s) - Fiabilite calculee pour 11 resultats (score moyen=0.00)

## Synthese
- design: `unknown` (confidence=0.00)
- actions: interventions=0, comparators=2, population=parents
- outcomes: count=5, confidence=0.80
- effects: total=11, derivable=11
- article_reporting_score: 0.65
- rob2: status=not_applicable, overall=not_applicable
- reliability: items=11, mean_score=0.00
- sqlite_index: `C:\Users\87fug\Documents\centrale\impact\outputs\impact_index.sqlite`

## Choix Methodologiques
- Objectif: maximiser la tracabilite et limiter les hallucinations.
- Metadata: heuristiques deterministes uniquement (LLM desactive).
- Extraction structuree: heuristiques + regex deterministes (LLM desactive).
- Design router: rules-first, fallback LLM si confiance < 0.75.
- Post-traitement effets: typage (role/level), dedoublonnage canonique, puis check statistique (CI/value).
- Effets: 14 effets produits
- Effets: extraction_mode=strict
- Effets: reported_effect regex entries: 14
- Effets: LLM extraction disabled.
- Effets: canonical_effects=11
- Effets: raw_effects=14
- Effets: stat_check_failed=0
- Effets: stat_check_total=11
- Effets: primary_effects=0
- Effets: secondary_effects=11

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
| `res_4722b059fc419c76` | False | unclear | mothers | unknown | baseline | d | -0.470 | n/a | unknown | 2 | reported/reported/exact | 0.00 (Not usable) |
| `res_902016b7bbc17c20` | False | unclear | mothers | unknown | baseline | d | -0.660 | n/a | unknown | 2 | reported/reported/exact | 0.00 (Not usable) |
| `res_f5d90f293d75a436` | False | unclear | mothers | unknown | baseline | d | -0.560 | n/a | unknown | 2 | reported/reported/exact | 0.00 (Not usable) |
| `res_b5995b688b9a605d` | False | unclear | mothers | unknown | baseline | d | -0.710 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_00163b2678e1cde9` | False | unclear | mothers | unknown | baseline | d | -0.450 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_a0e26c26c5dfdd59` | False | unclear | mothers | unknown | baseline | d | 0.610 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_63167fb5b0679a01` | False | unclear | mothers | unknown | baseline | d | 0.570 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_ca40bae6164981c3` | False | unclear | mothers | reported_outcome | baseline | d | 0.540 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_df06ee444c472b29` | False | unclear | mothers | reported_outcome | baseline | d | 0.770 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_4106a7fa3ad91b88` | False | unclear | mothers | unknown | baseline | d | -0.610 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |
| `res_9486b4a9f3ea4d64` | False | pooled_overall | overall | unknown | baseline | d | -0.190 | n/a | unknown | 1 | reported/reported/exact | 0.00 (Not usable) |

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
