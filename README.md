# Impact Pipeline

Application pour analyser un article en pdf pour obtenir :
- des effets (`d/g/SMD`) tracables (quote/page/source),
- une qualite interne rapide,
- une credibilite externe,
- un score global de fiabilite.

## Prerequis
- Python 3.10+
- (Optionnel) cle OpenAI pour les etapes LLM

## Installation
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration
Copie le fichier d'exemple puis renseigne les variables utiles:
```powershell
Copy-Item .env.example .env
```

Variables principales:
- `OPENAI_API_KEY` (optionnel mais recommande)
- `OPENAI_API_BASE` (par defaut OpenAI)
- `OPENALEX_EMAIL` (optionnel, recommande pour OpenAlex)
- `GROBID_URL` (optionnel)

## Lancer l'app (UI)
```powershell
streamlit run app.py
```

## Lancer en CLI
Pipeline complet:
```powershell
python -m src.cli pipeline "C:\chemin\vers\article.pdf"
```

Ingestion seule:
```powershell
python -m src.cli ingest "C:\chemin\vers\article.pdf"
```

Controles qualite d'un run:
```powershell
python -m src.cli check "outputs\<paper_id>"
```

## Sorties
Chaque run ecrit dans `outputs/<paper_id>/`:
- `00_metadata.json`
- `01_text_index.json`
- `02_tables.json`
- `03_block_flags.json`
- `04_effects.json`
- `05_quality_quick.json`
- `06_external_credibility.json`
- `07_summary_score.json`
- `12_reliability.json`
- `report.md`

## Notes
- Sans cle OpenAI, le pipeline fonctionne en mode deterministe (regex/regles).
- Les resultats les plus utiles pour l'analyse sont `04_effects.json` et `07_summary_score.json`.
