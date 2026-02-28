from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.effect_labels import derive_group_domain_predictor
from src.models import IngestConfig, PipelineRunSummary
from src.orchestrator import PIPELINE_STEPS, run_full_pipeline

PALETTE = {
    "impact_green": "#4CAF50",
    "compass_blue": "#2E7DBD",
    "accent_yellow": "#F2C94C",
    "bg_main": "#F3F1E8",
    "bg_soft": "#FAFAF7",
    "text_black": "#111111",
    "text_gray": "#5C5C5C",
}

STATE_ICONS = {
    "pending": "[ ]",
    "running": "__SPINNER__",
    "completed": "[OK]",
    "skipped": "[>>]",
    "failed": "[X]",
}

STATE_LABELS = {
    "pending": "En attente",
    "running": "En cours",
    "completed": "Termine",
    "skipped": "Ignore",
    "failed": "Echec",
}

STEP_TOOLTIPS = {
    "01": (
        "Bloc 1 - Ingestion\n"
        "Extraction du texte et des tableaux du PDF.\n"
        "Sorties: metadata + passages + tables."
    ),
    "02": (
        "Bloc 2 - Classification des blocs\n"
        "Pour chaque bloc, flags yes/no: results, effect_size, methods, population, bias_info.\n"
        "Sortie: 03_block_flags.json."
    ),
    "03": (
        "Bloc 3 - Extraction des effets\n"
        "Extraction ciblee d/g/SMD uniquement sur les blocs effect_size pertinents.\n"
        "Sortie: 04_effects.json."
    ),
    "04": (
        "Bloc 4 - Qualite interne rapide\n"
        "Grille simple (randomisation, controle, n, attrition, blinding) + score.\n"
        "Sortie: 05_quality_quick.json."
    ),
    "05": (
        "Bloc 5 - Credibilite externe\n"
        "OpenAlex (titre, venue, citations, auteurs) + score de renommee.\n"
        "Sortie: 06_external_credibility.json."
    ),
    "06": (
        "Bloc 6 - Score final\n"
        "Combine effets + qualite interne + credibilite externe.\n"
        "Sorties: 07_summary_score.json + report.md."
    ),
}


def main() -> None:
    st.set_page_config(page_title="Impact - Suivi pipeline", layout="wide")
    _inject_css()

    st.title("Impact - Suivi du pipeline")
    st.caption("Charge un PDF puis suis chaque etape en temps reel.")

    with st.sidebar:
        st.subheader("Parametres d execution")
        output_root = st.text_input(
            "Dossier de sortie",
            value="outputs",
            help="Dossier racine ou le pipeline ecrit les resultats.",
        )
        paper_id = st.text_input(
            "Paper ID (optionnel)",
            value="",
            help="Identifiant manuel du papier. Si vide, il est genere automatiquement.",
        )
        chunk_chars = st.number_input(
            "Taille max d un chunk (chars)",
            min_value=200,
            max_value=5000,
            value=1800,
            step=100,
            help="Longueur cible d un passage texte dans 01_text_index.json.",
        )
        chunk_overlap = st.number_input(
            "Overlap entre chunks (chars)",
            min_value=0,
            max_value=1000,
            value=220,
            step=20,
            help="Nombre de caracteres partages entre deux chunks consecutifs.",
        )
        min_chunk_chars = st.number_input(
            "Taille min d un chunk (chars)",
            min_value=40,
            max_value=1000,
            value=120,
            step=20,
            help="Evite de produire des chunks trop courts et peu informatifs.",
        )
        metadata_pages_scan = st.number_input(
            "Nb pages pour metadata",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="Nombre de premieres pages analysees pour extraire titre/auteurs/annee.",
        )
        use_grobid = st.checkbox(
            "Activer GROBID (optionnel)",
            value=False,
            help="Utilise GROBID en complement pour enrichir la structure et la metadata.",
        )
        grobid_url = st.text_input(
            "URL GROBID",
            value="http://localhost:8070",
            disabled=not use_grobid,
            help="URL du service GROBID (ex: http://localhost:8070).",
        )
        use_openai_metadata = st.checkbox(
            "Affiner metadata avec OpenAI",
            value=True,
            help="Utilise OPENAI_API_KEY dans .env. N envoie que les premieres pages.",
        )
        use_openai_extraction = st.checkbox(
            "Affiner extraction avec OpenAI",
            value=True,
            help=(
                "Utilise le LLM sur des blocs cibles (pas le PDF complet): "
                "classification yes/no, extraction des effets, quick quality."
            ),
        )
        openai_extraction_max_snippets = st.number_input(
            "Taille lot snippets LLM",
            min_value=6,
            max_value=80,
            value=40,
            step=2,
            disabled=not use_openai_extraction,
            help="Nombre de blocs envoyes par appel LLM (le pipeline peut faire plusieurs appels).",
        )
        openai_effect_snippet_chars = st.number_input(
            "Taille snippet effets (chars)",
            min_value=300,
            max_value=3200,
            value=1200,
            step=100,
            disabled=not use_openai_extraction,
            help="Nombre de caracteres envoyes par snippet au module LLM d extraction des effets.",
        )
        openai_model = st.text_input(
            "Modele OpenAI",
            value="gpt-4.1-mini",
            disabled=not (use_openai_metadata or use_openai_extraction),
            help="Modele utilise pour metadata et/ou extraction structuree selon options activees.",
        )
        ui_delay = st.slider(
            "Delai visuel entre etapes (s)",
            min_value=0.0,
            max_value=1.5,
            value=0.35,
            step=0.05,
            help="Ralentit legerement l affichage pour suivre le flow pas a pas.",
        )

    uploaded_pdf = st.file_uploader(
        "Fichier PDF",
        type=["pdf"],
        help="Article PDF a analyser.",
    )
    run_clicked = st.button(
        "Lancer le pipeline",
        type="primary",
        disabled=uploaded_pdf is None,
        help="Demarre l ingestion puis le flow complet des etapes.",
    )

    st.subheader("Suivi des etapes")
    progress_bar = st.progress(0.0, text="En attente de lancement")
    step_placeholders, step_states = _init_step_placeholders()
    step_lookup = {step.step_id: step for step in PIPELINE_STEPS}
    total_steps = len(PIPELINE_STEPS)

    if not run_clicked:
        st.caption("Les etapes du pipeline sont affichees ci-dessous avant execution.")
        return

    if uploaded_pdf is None:
        st.warning("Charge un PDF avant de lancer.")
        return

    pdf_path = _persist_uploaded_pdf(uploaded_pdf.name, uploaded_pdf.getvalue())
    st.info(f"Fichier source enregistre dans `{pdf_path}`")

    ingest_config_kwargs = {
        "chunk_chars": int(chunk_chars),
        "chunk_overlap": int(chunk_overlap),
        "min_chunk_chars": int(min_chunk_chars),
        "metadata_pages_scan": int(metadata_pages_scan),
        "use_grobid": use_grobid,
        "use_openai_metadata": use_openai_metadata,
        "use_openai_extraction": use_openai_extraction,
        "openai_extraction_max_snippets": int(openai_extraction_max_snippets),
        "openai_effect_snippet_chars": int(openai_effect_snippet_chars),
        "openai_model": openai_model,
    }
    if use_grobid:
        ingest_config_kwargs["grobid_url"] = grobid_url
    ingest_config = IngestConfig(**ingest_config_kwargs)
    progress_bar.progress(0.0, text="Initialisation...")

    def on_progress(step_id: str, state: str, message: str, payload: Dict[str, object]) -> None:
        step = step_lookup[step_id]
        step_states[step_id] = state
        _render_step(
            placeholder=step_placeholders[step_id],
            step_id=step.step_id,
            step_name=step.name,
            state=state,
            detail=message,
        )
        completed_steps = _count_terminal_steps(step_states)
        ratio = completed_steps / total_steps if total_steps else 0
        progress_bar.progress(
            ratio,
            text=f"{completed_steps}/{total_steps} etapes traitees",
        )

    try:
        summary = run_full_pipeline(
            pdf_path=pdf_path,
            output_root=output_root,
            paper_id=paper_id or None,
            ingest_config=ingest_config,
            progress_callback=on_progress,
            visualization_delay_seconds=ui_delay,
        )
    except Exception as exc:
        progress_bar.progress(
            _count_terminal_steps(step_states) / total_steps,
            text="Pipeline en echec",
        )
        st.error(f"Echec du pipeline: {exc}")
        return

    progress_bar.progress(1.0, text="Pipeline termine")
    st.success(f"Pipeline termine pour paper_id `{summary.paper_id}`")
    _render_summary(summary)


def _persist_uploaded_pdf(filename: str, content: bytes) -> Path:
    upload_dir = Path("outputs") / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_name = _sanitize_filename(filename)
    output_path = upload_dir / f"{timestamp}_{safe_name}"
    output_path.write_bytes(content)
    return output_path.resolve()


def _sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename).strip("_")
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned or "uploaded.pdf"


def _init_step_placeholders() -> Tuple[Dict[str, DeltaGenerator], Dict[str, str]]:
    placeholders: Dict[str, DeltaGenerator] = {}
    states: Dict[str, str] = {}

    for step in PIPELINE_STEPS:
        placeholders[step.step_id] = st.empty()
        states[step.step_id] = "pending"
        _render_step(
            placeholder=placeholders[step.step_id],
            step_id=step.step_id,
            step_name=step.name,
            state="pending",
            detail=step.description,
        )
    return placeholders, states


def _render_step(
    placeholder: DeltaGenerator,
    step_id: str,
    step_name: str,
    state: str,
    detail: str,
) -> None:
    icon = STATE_ICONS.get(state, "[ ]")
    if icon == "__SPINNER__":
        icon_html = '<span class="loader" aria-label="running"></span>'
    else:
        icon_html = f"<span>{icon}</span>"

    state_styles = {
        "pending": {
            "border": "#5C5C5C",
            "background": "#F0EEE4",
            "status": "#5C5C5C",
        },
        "running": {
            "border": PALETTE["accent_yellow"],
            "background": "#FFF7D7",
            "status": "#8A6A00",
        },
        "completed": {
            "border": PALETTE["impact_green"],
            "background": "#E7F6E8",
            "status": PALETTE["impact_green"],
        },
        "skipped": {
            "border": PALETTE["compass_blue"],
            "background": "#E4EFF8",
            "status": PALETTE["compass_blue"],
        },
        "failed": {
            "border": PALETTE["text_black"],
            "background": "#EDE8DF",
            "status": PALETTE["text_black"],
        },
    }
    style = state_styles.get(state, state_styles["pending"])
    card_style = f"border-left-color:{style['border']};background:{style['background']};"
    status_style = f"color:{style['status']};"
    tooltip_raw = STEP_TOOLTIPS.get(step_id, "")
    if tooltip_raw:
        tooltip_body = html.escape(tooltip_raw).replace("\n", "<br>")
        info_html = (
            '<span class="step-info-wrap">'
            '<span class="step-info">i</span>'
            f'<span class="step-tooltip">{tooltip_body}</span>'
            "</span>"
        )
    else:
        info_html = ""

    label = STATE_LABELS.get(state, state)
    card_html = f"""
    <div class="step-card" style="{card_style}">
      <div class="step-title">{icon_html}<span><strong>{step_id} - {step_name}</strong></span>{info_html}</div>
      <div class="step-status" style="{status_style}">{label}</div>
      <div class="step-detail">{detail}</div>
    </div>
    """
    placeholder.markdown(card_html, unsafe_allow_html=True)


def _count_terminal_steps(states: Dict[str, str]) -> int:
    return sum(1 for value in states.values() if value in {"completed", "skipped", "failed"})


def _render_summary(summary: PipelineRunSummary) -> None:
    st.subheader("Resume d execution")
    st.write(f"Dossier de sortie: `{summary.output_dir}`")

    rows = []
    for step in summary.steps:
        rows.append(
            {
                "step_id": step.step_id,
                "name": step.name,
                "status": step.status,
                "duration_s": round(step.duration_seconds, 3),
                "message": step.message,
            }
        )

    st.dataframe(rows, use_container_width=True)

    output_dir = Path(summary.output_dir)
    if output_dir.exists():
        _render_article_overview(output_dir)
        _render_effects_overview(output_dir)
        _render_report_markdown(output_dir)
        st.write("Fichiers generes:")
        st.code("\n".join(path.name for path in sorted(output_dir.glob("*"))))
        _render_output_json_expanders(output_dir)


def _render_article_overview(output_dir: Path) -> None:
    metadata_payload = _load_json_file(output_dir / "00_metadata.json")
    quality_payload = _load_json_file(output_dir / "05_quality_quick.json")
    external_payload = _load_json_file(output_dir / "06_external_credibility.json")
    reliability_payload = _load_json_file(output_dir / "07_summary_score.json")
    effects_payload = _load_json_file(output_dir / "04_effects.json")

    if not isinstance(metadata_payload, dict):
        return

    title = str(metadata_payload.get("title", "unknown"))
    authors_raw = metadata_payload.get("authors", [])
    authors = ", ".join(authors_raw) if isinstance(authors_raw, list) and authors_raw else "unknown"
    year = metadata_payload.get("year", "unknown")
    doi = str(metadata_payload.get("doi", "unknown"))
    pages = metadata_payload.get("num_pages", "unknown")
    paper_id = metadata_payload.get("paper_id", "unknown")

    global_score = _as_float(reliability_payload, "global_score")
    conclusion = (
        str(reliability_payload.get("conclusion", "unknown"))
        if isinstance(reliability_payload, dict)
        else "unknown"
    )
    quality_score = _as_float(quality_payload, "internal_quality_score")
    external_score = _as_float(external_payload, "external_score")
    quality_note = (
        str(quality_payload.get("justification", ""))
        if isinstance(quality_payload, dict)
        else ""
    )
    venue = (
        str(external_payload.get("venue", "unknown"))
        if isinstance(external_payload, dict)
        else "unknown"
    )
    citations = (
        str(external_payload.get("citation_count", "unknown"))
        if isinstance(external_payload, dict)
        else "unknown"
    )

    effects_count = 0
    study_effects_count = 0
    literature_effects_count = 0
    model_stats_count = 0
    if isinstance(effects_payload, dict):
        values = effects_payload.get("effects", [])
        if isinstance(values, list):
            effects_count = len(values)
            for effect in values:
                if not isinstance(effect, dict):
                    continue
                scope = str(effect.get("effect_scope", "study_effect"))
                if scope == "literature_cited":
                    literature_effects_count += 1
                elif scope == "model_stat":
                    model_stats_count += 1
                else:
                    study_effects_count += 1

    st.subheader("Fiche article")
    st.markdown(f"**Titre**: {title}")
    st.markdown(f"**Auteurs**: {authors}")
    st.caption(f"paper_id={paper_id} | annee={year} | doi={doi} | pages={pages}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Note globale", _fmt_score(global_score))
    col2.metric("Qualite interne", _fmt_score(quality_score))
    col3.metric("Credibilite externe", _fmt_score(external_score))
    col4.metric("Effets etude", str(study_effects_count))
    st.markdown(f"**Conclusion**: {conclusion}")
    st.caption(
        f"Effets total={effects_count} | cites={literature_effects_count} | stats_modele={model_stats_count}"
    )
    st.caption(f"Venue: {venue} | Citations: {citations}")
    if quality_note:
        st.caption(f"Note methode (raw): {quality_note}")


def _render_report_markdown(output_dir: Path) -> None:
    report_path = output_dir / "report.md"
    if not report_path.exists():
        return
    st.subheader("Rapport")
    with st.expander("report.md", expanded=True):
        st.markdown(report_path.read_text(encoding="utf-8", errors="replace"))


def _render_effects_overview(output_dir: Path) -> None:
    effects_path = output_dir / "04_effects.json"
    if not effects_path.exists():
        return

    effects_payload = _load_json_file(effects_path)
    if not isinstance(effects_payload, dict):
        return
    effects = effects_payload.get("effects", [])
    if not isinstance(effects, list) or not effects:
        return

    study_rows: list[dict[str, Any]] = []
    cited_rows: list[dict[str, Any]] = []
    model_stat_rows: list[dict[str, Any]] = []
    for effect in effects:
        if not isinstance(effect, dict):
            continue
        spec = effect.get("result_spec", {}) if isinstance(effect.get("result_spec"), dict) else {}
        value = effect.get("value")
        effect_type = str(effect.get("effect_type", "unknown"))
        ci_low = effect.get("ci_low")
        ci_high = effect.get("ci_high")

        group, domain, predictor = derive_group_domain_predictor(
            group_raw=str(effect.get("grouping_label") or spec.get("groups") or ""),
            predictor_raw=str(spec.get("outcome") or ""),
            domain_raw=str(effect.get("outcome_label_normalized") or ""),
            context=str(effect.get("quote", "")),
        )
        duration = str(effect.get("timepoint_label_normalized") or spec.get("timepoint") or "unknown")

        effect_size = _fmt_effect_size(effect_type, value, ci_low, ci_high)

        row = {
            "Groups": group,
            "Domain": domain,
            "Predictor": predictor,
            "Effect size": effect_size,
            "Effect duration": duration,
            "Quote": str(effect.get("quote", ""))[:120],
            "Page": effect.get("source_page", "unknown"),
            "Source": effect.get("source_kind", "unknown"),
        }
        effect_scope = str(effect.get("effect_scope", "study_effect"))
        if effect_scope == "literature_cited":
            cited_rows.append(row)
        elif effect_scope == "model_stat":
            model_stat_rows.append(row)
        else:
            study_rows.append(row)

    study_rows.sort(key=lambda item: _extract_abs_effect_value(item.get("Effect size", "")), reverse=True)
    cited_rows.sort(key=lambda item: _extract_abs_effect_value(item.get("Effect size", "")), reverse=True)
    model_stat_rows.sort(key=lambda item: _extract_abs_effect_value(item.get("Effect size", "")), reverse=True)

    st.subheader("Effets de l'etude")
    if study_rows:
        st.dataframe(study_rows, use_container_width=True)
    else:
        st.caption("Aucun effet de l'etude extrait.")

    st.subheader("Effets cites dans la litterature")
    if cited_rows:
        st.dataframe(cited_rows, use_container_width=True)
    else:
        st.caption("Aucun effet cite extrait.")

    st.subheader("Stats de modele (exclues des effect sizes)")
    if model_stat_rows:
        st.dataframe(model_stat_rows, use_container_width=True)
    else:
        st.caption("Aucune stat de modele detectee.")


def _fmt_effect_size(effect_type: str, value: Any, ci_low: Any, ci_high: Any) -> str:
    value_num = _as_float_value(value)
    if value_num is None:
        return f"{effect_type}=n/a"
    ci_low_num = _as_float_value(ci_low)
    ci_high_num = _as_float_value(ci_high)
    if ci_low_num is not None and ci_high_num is not None:
        return f"{effect_type}={value_num:.3f} (CI [{ci_low_num:.3f}; {ci_high_num:.3f}])"
    return f"{effect_type}={value_num:.3f}"


def _extract_abs_effect_value(effect_size: Any) -> float:
    text = str(effect_size)
    match = re.search(r"=\s*([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    try:
        return abs(float(match.group(1)))
    except Exception:
        return 0.0


def _as_float(payload: Any, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    return _as_float_value(payload.get(key))


def _as_float_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _fmt_score(value: Any) -> str:
    score = _as_float_value(value)
    if score is None:
        return "n/a"
    return f"{score:.2f}"


def _render_output_json_expanders(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        return

    st.write("Contenu JSON (menus deployants):")
    for json_path in json_files:
        with st.expander(json_path.name, expanded=False):
            file_size_kb = json_path.stat().st_size / 1024
            st.caption(f"Taille: {file_size_kb:.1f} KB")
            if file_size_kb > 1800:
                st.warning(
                    "Fichier volumineux: affichage d un apercu pour garder l interface fluide."
                )
                preview = _read_text_preview(json_path, max_chars=12000)
                st.code(preview, language="json")
                continue

            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                st.json(payload)
            except Exception as exc:
                st.error(f"Impossible de parser le JSON: {exc}")
                st.code(_read_text_preview(json_path, max_chars=12000), language="json")


def _read_text_preview(path: Path, max_chars: int = 12000) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(max_chars + 1)
    if len(content) > max_chars:
        return content[:max_chars] + "\n... (apercu tronque)"
    return content


def _load_json_file(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --impact-green: #4CAF50;
            --compass-blue: #2E7DBD;
            --accent-yellow: #F2C94C;
            --bg-main: #F3F1E8;
            --bg-soft: #FAFAF7;
            --text-black: #111111;
            --text-gray: #5C5C5C;
          }
          .stApp,
          .stAppViewContainer,
          [data-testid="stAppViewContainer"] {
            background: var(--bg-main) !important;
            color: var(--text-black) !important;
            color-scheme: light !important;
          }
          [data-testid="stSidebar"],
          [data-testid="stSidebar"] > div:first-child {
            background: var(--bg-soft) !important;
            border-right: 2px solid rgba(46, 125, 189, 0.35) !important;
          }
          h1, h2, h3, h4 {
            color: var(--compass-blue) !important;
          }
          .stCaption, p {
            color: var(--text-gray) !important;
          }
          strong, b {
            color: var(--text-black) !important;
          }
          .stTextInput label,
          .stNumberInput label,
          .stSlider label,
          .stCheckbox label,
          .stFileUploader label {
            color: var(--compass-blue) !important;
            font-weight: 600 !important;
          }
          [data-baseweb="input"] > div,
          [data-baseweb="select"] > div,
          [data-baseweb="textarea"] > div {
            background: var(--bg-soft) !important;
            border: 1.6px solid var(--compass-blue) !important;
            border-radius: 10px !important;
            min-height: 42px !important;
          }
          [data-baseweb="input"] > div:focus-within,
          [data-baseweb="select"] > div:focus-within,
          [data-baseweb="textarea"] > div:focus-within {
            border-color: var(--impact-green) !important;
            box-shadow: 0 0 0 1px var(--impact-green) !important;
          }
          [data-baseweb="input"] input,
          [data-baseweb="select"] input,
          [data-baseweb="textarea"] textarea {
            color: var(--text-black) !important;
            caret-color: var(--compass-blue) !important;
          }
          [data-testid="stNumberInput"] button {
            color: var(--compass-blue) !important;
            border-radius: 8px !important;
          }
          [data-testid="stNumberInput"] button:hover {
            color: var(--impact-green) !important;
            background: #E9F6EA !important;
          }
          [data-testid="stCheckbox"] div[role="checkbox"] {
            background: var(--bg-soft) !important;
            border: 2px solid var(--compass-blue) !important;
          }
          [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
            background: var(--impact-green) !important;
            border-color: var(--impact-green) !important;
          }
          [data-testid="stSlider"] [role="slider"] {
            background: var(--compass-blue) !important;
            border: 2px solid var(--compass-blue) !important;
          }
          [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-soft) !important;
            border: 2px dashed var(--compass-blue) !important;
            border-radius: 12px !important;
          }
          [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--impact-green) !important;
            background: #EEF7EE !important;
          }
          .stButton > button {
            background: var(--compass-blue) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--compass-blue) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
          }
          .stButton > button:hover {
            background: #256CA4 !important;
            border-color: var(--impact-green) !important;
            color: #FFFFFF !important;
          }
          [data-testid="stProgressBar"] > div > div > div > div {
            background: linear-gradient(90deg, var(--compass-blue), var(--impact-green)) !important;
          }
          .step-card {
            border: 1px solid rgba(46, 125, 189, 0.35) !important;
            border-left: 8px solid var(--compass-blue) !important;
            border-radius: 12px !important;
            padding: 0.85rem 1rem !important;
            margin-bottom: 0.55rem !important;
            background: var(--bg-soft) !important;
            box-shadow: 0 3px 10px rgba(17, 17, 17, 0.06) !important;
          }
          .step-title {
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
            font-size: 1rem !important;
            color: var(--text-black) !important;
          }
          .step-info {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 700;
            cursor: help;
            background: var(--accent-yellow);
            color: var(--text-black) !important;
            border: 1px solid var(--compass-blue);
          }
          .step-info-wrap {
            margin-left: auto;
            position: relative;
            display: inline-flex;
            align-items: center;
          }
          .step-tooltip {
            position: absolute;
            right: 0;
            top: calc(100% + 8px);
            min-width: 340px;
            max-width: 520px;
            background: #FAFAF7;
            color: #111111 !important;
            border: 1px solid #2E7DBD;
            border-left: 5px solid #F2C94C;
            border-radius: 10px;
            padding: 0.65rem 0.75rem;
            font-size: 0.82rem;
            line-height: 1.35;
            box-shadow: 0 8px 18px rgba(17, 17, 17, 0.16);
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transform: translateY(4px);
            transition: opacity 0.16s ease, transform 0.16s ease, visibility 0.16s ease;
            pointer-events: none;
            white-space: normal;
          }
          .step-info-wrap:hover .step-tooltip,
          .step-info-wrap:focus-within .step-tooltip {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
          }
          .step-status {
            color: var(--compass-blue) !important;
            font-size: 0.84rem !important;
            margin-top: 0.12rem !important;
            font-weight: 700 !important;
          }
          .step-detail {
            color: var(--text-gray) !important;
            font-size: 0.9rem !important;
            margin-top: 0.2rem !important;
          }
          .loader {
            width: 12px;
            height: 12px;
            border: 2px solid rgba(46, 125, 189, 0.2);
            border-top-color: var(--compass-blue);
            border-radius: 50%;
            display: inline-block;
            animation: spin 0.9s linear infinite;
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
