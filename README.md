# Rank'impact

Outil d'analyse automatique d'articles scientifiques.  
A partir d'un PDF, le pipeline extrait des tailles d'effet (d / g / SMD), evalue la qualite methodologique, mesure la credibilite externe (citations, revue, auteurs) et produit un score global de fiabilite.

## Pipeline

```
PDF → Ingestion → Routage blocs → Extraction effets → Qualite interne → Credibilite externe → Evaluation article → Score global
       PyMuPDF     Heuristic+LLM   4 modes (regex,      5 criteres        OpenAlex              OpenAlex+SCImago     Ponderation
       pdfplumber                    LLM texte/vision)                     Crossref/DOI           LLM                 explicite
```

Les 7 etapes produisent chacune un JSON structure dans `outputs/<paper_id>/`.

## Prerequis

- Python 3.10+
- Cle OpenAI (optionnel — sans cle, le pipeline fonctionne en mode deterministe regex/regles)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

```powershell
Copy-Item .env.example .env
```

Edite `.env` avec tes cles :

| Variable | Description | Requis |
|---|---|---|
| `OPENAI_API_KEY` | Cle API OpenAI (GPT-4.1-mini) | Recommande |
| `OPENAI_API_BASE` | Base URL OpenAI (defaut : `https://api.openai.com/v1`) | Non |
| `OPENALEX_EMAIL` | Email pour l'API OpenAlex (meilleur rate limit) | Recommande |
| `GROBID_URL` | URL du serveur GROBID | Non |

## Utilisation

### Interface Streamlit

```powershell
streamlit run app.py
```

### CLI

```powershell
# Pipeline complet
python -m src.cli pipeline "chemin\vers\article.pdf"

# Ingestion seule
python -m src.cli ingest "chemin\vers\article.pdf"

# Controle qualite d'un run existant
python -m src.cli check "outputs\<paper_id>"
```

## Sorties

Chaque analyse ecrit dans `outputs/<paper_id>/` :

| Fichier | Contenu |
|---|---|
| `00_metadata.json` | Metadonnees du PDF (titre, auteurs, DOI) |
| `01_text_index.json` | Passages texte indexes par section/page |
| `02_tables.json` | Tableaux extraits |
| `03_block_flags.json` | Classification des blocs (results/effects/methods…) |
| `04_effects.json` | Tailles d'effet extraites (d/g/SMD) |
| `05_quality_quick.json` | Score de qualite methodologique |
| `06_external_credibility.json` | Credibilite externe (citations, revue, auteurs) |
| `07_summary_score.json` | Score global de fiabilite |
| `08_article_evaluation.json` | Evaluation detaillee article/journal/auteur |
| `12_reliability.json` | Score de fiabilite par effet |
| `report.md` | Rapport synthetique lisible |

## Structure du projet

```
app.py                  # Interface Streamlit
src/
  cli.py                # Interface ligne de commande
  orchestrator.py       # Enchainement des 7 etapes du pipeline
  models.py             # Modeles Pydantic (schemas JSON)
  effect_labels.py      # Taxonomie des effets (domaines, predicteurs)
  quality_checks.py     # Controles qualite post-run
  text_normalize.py     # Normalisation texte
  steps/                # Implementation de chaque etape
    ingest.py           # Etape 1 : parsing PDF
    block_router.py     # Etape 2 : classification des blocs
    effects_extract.py  # Etape 3 : extraction d/g/SMD
    quality_quick.py    # Etape 4 : grille qualite interne
    external_credibility.py  # Etape 5 : OpenAlex/Crossref
    article_evaluation.py    # Etape 6 : evaluation article
    scoring.py          # Etape 7 : score final
    scimago_client.py   # Lookup SCImago (SJR, quartile)
    openalex_client.py  # Client API OpenAlex
    doi_resolution.py   # Resolution DOI via Crossref
    reliability.py      # Calcul fiabilite par effet
  storage/
    sqlite.py           # Persistance SQLite locale
data/                   # Donnees SCImago (1999-2024, 26 fichiers CSV)
scripts/
  merge_scimago.py      # Utilitaire : fusion des CSV SCImago
outputs/                # Resultats des analyses
documents/              # PDFs d'articles a analyser
```
