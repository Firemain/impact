"""Impact – Streamlit UI.

Professional interface with three main views:
1. Pipeline execution & monitoring
2. Article evaluation dashboard (scores)
3. Effects & study analysis
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.effect_labels import derive_group_domain_predictor
from src.models import IngestConfig, PipelineRunSummary
from src.orchestrator import PIPELINE_STEPS, run_full_pipeline
from src.steps.scoring import ScoringThresholds

# ──────────────────────── Design tokens ──────────────────────────

PALETTE = {
    "primary": "#1B2A4A",
    "primary_light": "#2D4A7A",
    "accent": "#3B82F6",
    "accent_light": "#60A5FA",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "bg": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_sidebar": "#F1F5F9",
    "text": "#0F172A",
    "text_secondary": "#64748B",
    "border": "#E2E8F0",
}


# ──────────────────────── Entrypoint ──────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Rank'impact – Analyseur d'articles",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    # ── Header ──
    st.markdown(
        '<div class="app-header">'
        '<div class="app-logo"><span>Rank\'impact</span></div>'
        '<div class="app-subtitle">Analyseur d\'articles</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        _render_sidebar()

    # ── Main content via tabs ──
    tab_pipeline, tab_evaluation, tab_effects, tab_study, tab_report = st.tabs([
        "⚙️  Pipeline", "📈  Article", "🔬  Effets", "🧪  Étude", "📋  Rapport"
    ])

    with tab_pipeline:
        _render_pipeline_tab()

    with tab_evaluation:
        _render_evaluation_tab()

    with tab_effects:
        _render_effects_tab()

    with tab_study:
        _render_study_tab()

    with tab_report:
        _render_report_tab()


# ──────────────────────── Sidebar ─────────────────────────────────

def _render_sidebar() -> None:
    st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)

    with st.expander("📁 Sortie", expanded=True):
        st.session_state.setdefault("output_root", "outputs")
        st.session_state["output_root"] = st.text_input(
            "Dossier de sortie", value=st.session_state["output_root"],
            help="Dossier racine pour les résultats du pipeline.",
            key="cfg_output_root",
        )
        st.session_state.setdefault("paper_id", "")
        st.session_state["paper_id"] = st.text_input(
            "ID de l'article (optionnel)", value=st.session_state["paper_id"],
            help="Identifiant manuel. Généré automatiquement si vide.",
            key="cfg_paper_id",
        )

    with st.expander("🔧 Ingestion", expanded=False):
        st.session_state["chunk_chars"] = st.number_input(
            "Taille des blocs (caractères)", 200, 5000, 1800, 100,
        )
        st.session_state["chunk_overlap"] = st.number_input(
            "Chevauchement (caractères)", 0, 1000, 220, 20,
        )
        st.session_state["min_chunk_chars"] = st.number_input(
            "Bloc minimum (caractères)", 40, 1000, 120, 20,
        )
        st.session_state["metadata_pages_scan"] = st.number_input(
            "Pages de métadonnées", 1, 8, 4, 1,
        )

    with st.expander("🤖 Paramètres LLM", expanded=False):
        st.session_state["use_grobid"] = st.checkbox("Activer GROBID", value=False)
        st.session_state["grobid_url"] = st.text_input(
            "URL GROBID", "http://localhost:8070",
            disabled=not st.session_state.get("use_grobid", False),
            key="cfg_grobid_url",
        )
        st.session_state["use_openai_metadata"] = st.checkbox(
            "Affiner les métadonnées avec OpenAI", value=True,
        )
        st.session_state["use_openai_extraction"] = st.checkbox(
            "Utiliser OpenAI pour l'extraction", value=True,
        )
        st.session_state["openai_extraction_max_snippets"] = st.number_input(
            "Taille du lot LLM", 6, 80, 40, 2,
            disabled=not st.session_state.get("use_openai_extraction", True),
        )
        st.session_state["openai_effect_snippet_chars"] = st.number_input(
            "Extrait d'effet (caractères)", 300, 3200, 1200, 100,
            disabled=not st.session_state.get("use_openai_extraction", True),
        )
        st.session_state["openai_model"] = st.text_input(
            "Modèle", "gpt-4.1-mini",
            disabled=not (st.session_state.get("use_openai_metadata", True) or st.session_state.get("use_openai_extraction", True)),
            key="cfg_openai_model",
        )

    with st.expander("📊 Seuils de notation", expanded=False):
        st.caption("Ajustez les seuils qui déterminent le score maximum (= 1.0) pour chaque dimension.")
        st.session_state["threshold_citations_max"] = st.number_input(
            "📄 Citations pour score max", 50, 5000, 500, 50,
            help="Nombre de citations au-delà duquel le score Article atteint 1.0.",
            key="cfg_thresh_citations",
        )
        st.session_state["threshold_h_index_max"] = st.number_input(
            "👤 Indice h pour score max", 10, 200, 40, 5,
            help="Indice h au-delà duquel le score Auteur atteint 1.0.",
            key="cfg_thresh_hindex",
        )
        st.session_state["threshold_institutions_max"] = st.number_input(
            "🌐 Institutions pour score max", 2, 30, 8, 1,
            help="Nombre d'institutions distinctes pour atteindre le score Réseau max.",
            key="cfg_thresh_institutions",
        )
        st.session_state["threshold_author_works_max"] = st.number_input(
            "🌐 Publications moy. auteurs pour score max", 20, 1000, 200, 20,
            help="Moyenne de publications par auteur pour atteindre le score Réseau max.",
            key="cfg_thresh_author_works",
        )

    st.markdown(
        '<div style="text-align:center;margin-top:2rem;opacity:0.4;font-size:0.75rem;">'
        'Impact v2.0</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────── Pipeline Tab ────────────────────────────

STATE_LABELS = {
    "pending": ("⏳", "En attente"),
    "running": ("⏳", "En cours"),
    "completed": ("✅", "Terminé"),
    "skipped": ("⏭️", "Ignoré"),
    "failed": ("❌", "Échoué"),
}


def _render_pipeline_tab() -> None:
    # ── New analysis ──
    st.markdown("#### 📤 Nouvelle analyse")
    col_upload, col_action = st.columns([3, 1])
    with col_upload:
        uploaded_pdf = st.file_uploader(
            "Déposer un article au format PDF",
            type=["pdf"],
            help="Glissez votre PDF ici pour démarrer l'analyse.",
        )
    with col_action:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_clicked = st.button(
            "🚀 Démarrer l'analyse",
            type="primary",
            disabled=uploaded_pdf is None,
            use_container_width=True,
        )

    # Step progress
    step_container = st.container()
    with step_container:
        progress_bar = st.progress(0.0, text="En attente du PDF…")
        step_cols = st.columns(len(PIPELINE_STEPS))
        step_placeholders: Dict[str, DeltaGenerator] = {}
        step_states: Dict[str, str] = {}
        for i, step in enumerate(PIPELINE_STEPS):
            with step_cols[i]:
                step_placeholders[step.step_id] = st.empty()
                step_states[step.step_id] = "pending"
                _render_step_badge(
                    step_placeholders[step.step_id],
                    step.step_id, step.name, "pending", step.description,
                )

    # ── Previous analyses ──
    st.divider()
    _render_previous_analyses()

    if not run_clicked or uploaded_pdf is None:
        return

    pdf_path = _persist_uploaded_pdf(uploaded_pdf.name, uploaded_pdf.getvalue())

    config_kwargs: Dict[str, Any] = {
        "chunk_chars": int(st.session_state.get("chunk_chars", 1800)),
        "chunk_overlap": int(st.session_state.get("chunk_overlap", 220)),
        "min_chunk_chars": int(st.session_state.get("min_chunk_chars", 120)),
        "metadata_pages_scan": int(st.session_state.get("metadata_pages_scan", 4)),
        "use_grobid": st.session_state.get("use_grobid", False),
        "use_openai_metadata": st.session_state.get("use_openai_metadata", True),
        "use_openai_extraction": st.session_state.get("use_openai_extraction", True),
        "openai_extraction_max_snippets": int(st.session_state.get("openai_extraction_max_snippets", 40)),
        "openai_effect_snippet_chars": int(st.session_state.get("openai_effect_snippet_chars", 1200)),
        "openai_model": st.session_state.get("openai_model", "gpt-4.1-mini"),
    }
    if st.session_state.get("use_grobid"):
        config_kwargs["grobid_url"] = st.session_state.get("grobid_url", "http://localhost:8070")
    ingest_config = IngestConfig(**config_kwargs)

    step_lookup = {s.step_id: s for s in PIPELINE_STEPS}
    total_steps = len(PIPELINE_STEPS)

    def on_progress(step_id: str, state: str, message: str, payload: Dict[str, object]) -> None:
        step_states[step_id] = state
        step = step_lookup[step_id]
        _render_step_badge(step_placeholders[step_id], step.step_id, step.name, state, message)
        done = sum(1 for v in step_states.values() if v in {"completed", "skipped", "failed"})
        progress_bar.progress(done / total_steps, text=f"{done}/{total_steps} étapes terminées")

    try:
        # Build scoring thresholds from sidebar settings
        scoring_thresholds = ScoringThresholds(
            citations_max=int(st.session_state.get("threshold_citations_max", 500)),
            h_index_max=int(st.session_state.get("threshold_h_index_max", 40)),
            institutions_max=int(st.session_state.get("threshold_institutions_max", 8)),
            author_works_max=int(st.session_state.get("threshold_author_works_max", 200)),
        )
        summary = run_full_pipeline(
            pdf_path=pdf_path,
            output_root=st.session_state.get("output_root", "outputs"),
            paper_id=st.session_state.get("paper_id") or None,
            ingest_config=ingest_config,
            progress_callback=on_progress,
            scoring_thresholds=scoring_thresholds,
        )
    except Exception as exc:
        progress_bar.progress(
            sum(1 for v in step_states.values() if v in {"completed", "skipped", "failed"}) / total_steps,
            text="Le pipeline a échoué",
        )
        st.error(f"Erreur du pipeline : {exc}")
        return

    progress_bar.progress(1.0, text="✅ Analyse terminée")
    st.session_state["last_output_dir"] = summary.output_dir
    st.session_state["last_paper_id"] = summary.paper_id

    st.success(f"Analyse terminée — identifiant : **{summary.paper_id}**")

    # Quick summary table
    with st.expander("Résumé de l'exécution", expanded=False):
        rows = []
        for step in summary.steps:
            rows.append({
                "Étape": f"{step.step_id} — {step.name}",
                "Statut": step.status,
                "Durée": f"{step.duration_seconds:.1f}s",
                "Détails": step.message,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Show output files
    output_dir = Path(summary.output_dir)
    if output_dir.exists():
        with st.expander("Fichiers de sortie", expanded=False):
            _render_output_json_expanders(output_dir)


def _render_previous_analyses() -> None:
    """Show a list of previously analyzed documents from the outputs folder."""
    output_root = Path(st.session_state.get("output_root", "outputs"))
    if not output_root.exists():
        return

    # Scan for folders with metadata
    analyses: list[dict] = []
    for folder in sorted(output_root.iterdir(), reverse=True):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        meta_path = folder / "00_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            title = meta.get("title", "Sans titre")
            authors = meta.get("authors", [])
            year = meta.get("year", "—")
            doi = meta.get("doi", "")
            paper_id = meta.get("paper_id", folder.name)

            # Article score (from article evaluation)
            article_score = None
            keywords: list[str] = []
            eval_path = folder / "08_article_evaluation.json"
            if eval_path.exists():
                try:
                    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                    keywords = eval_data.get("keywords", []) or []
                    scores = eval_data.get("scores", {})
                    if isinstance(scores, dict):
                        g = scores.get("global", {})
                        if isinstance(g, dict) and g.get("value") is not None:
                            article_score = float(g["value"])
                except Exception:
                    pass

            # Study score (from reliability / summary)
            study_score = None
            summary_path = folder / "07_summary_score.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    gs = summary.get("global_score")
                    if gs is not None:
                        study_score = float(gs)
                except Exception:
                    pass

            # Effects count
            effects_count = 0
            effects_path = folder / "04_effects.json"
            if effects_path.exists():
                try:
                    eff_data = json.loads(effects_path.read_text(encoding="utf-8"))
                    effects_list = eff_data.get("effects", [])
                    if isinstance(effects_list, list):
                        effects_count = len([e for e in effects_list if isinstance(e, dict) and e.get("effect_scope") == "study_effect"])
                except Exception:
                    pass

            analyses.append({
                "folder": folder,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "paper_id": paper_id,
                "article_score": article_score,
                "study_score": study_score,
                "effects_count": effects_count,
                "keywords": keywords,
            })
        except Exception:
            continue

    if not analyses:
        return

    st.markdown("#### 📂 Analyses précédentes")

    for i, analysis in enumerate(analyses[:20]):
        authors_str = ", ".join(str(a) for a in analysis["authors"][:3])
        if len(analysis["authors"]) > 3:
            authors_str += " et al."

        kw_list = analysis["keywords"][:4]
        kw_str = " · ".join(kw_list) if kw_list else ""

        # Title + authors line
        col_title, col_art, col_study, col_eff, col_btn = st.columns([5, 1, 1, 0.8, 0.8])
        with col_title:
            st.markdown(
                f"**{analysis['title'][:80]}**  \n"
                f"<span style='font-size:0.78rem;color:#64748B;'>"
                f"{authors_str} · {analysis['year']}"
                f"{'  · ' + str(analysis['doi']) if analysis['doi'] else ''}"
                + (f"<br>🏷️ {kw_str}" if kw_str else "")
                + f"</span>",
                unsafe_allow_html=True,
            )
        with col_art:
            val = analysis["article_score"]
            st.metric("Article", f"{val:.2f}" if val is not None else "—")
        with col_study:
            val = analysis["study_score"]
            st.metric("Étude", f"{val:.2f}" if val is not None else "—")
        with col_eff:
            n = analysis["effects_count"]
            st.metric("Effets", str(n) if n > 0 else "—")
        with col_btn:
            st.markdown("<div style='padding-top:0.6rem'></div>", unsafe_allow_html=True)
            if st.button("📂 Ouvrir", key=f"open_analysis_{i}", help="Ouvrir cette analyse"):
                st.session_state["last_output_dir"] = str(analysis["folder"])
                st.rerun()


def _render_step_badge(
    placeholder: DeltaGenerator,
    step_id: str,
    name: str,
    state: str,
    detail: str,
) -> None:
    icon, label = STATE_LABELS.get(state, ("⏳", state))
    state_colors = {
        "pending":   "gray",
        "running":   "blue",
        "completed": "green",
        "skipped":   "orange",
        "failed":    "red",
    }
    color = state_colors.get(state, "gray")
    with placeholder.container():
        st.caption(f"**{step_id}** — {name}")
        st.markdown(f":{color}[{icon} {label}]")
        if detail:
            st.caption(detail[:80])


# ──────────────────────── Evaluation Tab ──────────────────────────

def _render_evaluation_tab() -> None:
    output_dir = _get_last_output_dir()
    if not output_dir:
        st.info("Lancez le pipeline d'abord, ou sélectionnez une analyse précédente.")
        _render_folder_picker(key_suffix="eval_top")
        return

    eval_data = _load_json(output_dir / "08_article_evaluation.json")
    metadata = _load_json(output_dir / "00_metadata.json")

    if not eval_data:
        st.warning("Données d'évaluation introuvables. Vérifiez que le pipeline s'est terminé correctement.")
        _render_folder_picker(key_suffix="eval_warn")
        return

    # ── Article header card ──
    title = eval_data.get("title") or (metadata.get("title") if metadata else "Unknown")
    year = eval_data.get("year") or (metadata.get("year") if metadata else "—")
    doi = eval_data.get("doi") or "—"
    authors_extracted = eval_data.get("authors_extracted", [])
    if authors_extracted:
        author_names = [a.get("full_name", "") if isinstance(a, dict) else str(a) for a in authors_extracted[:6]]
        authors_str = ", ".join(n for n in author_names if n)
        if len(authors_extracted) > 6:
            authors_str += f" (+{len(authors_extracted) - 6} autres)"
    elif metadata and isinstance(metadata.get("authors"), list):
        authors_str = ", ".join(str(a) for a in metadata["authors"][:6])
    else:
        authors_str = "—"

    journal_info = eval_data.get("journal_extracted", {})
    journal_name = journal_info.get("name") if isinstance(journal_info, dict) else "—"
    keywords = eval_data.get("keywords", [])
    document_type = eval_data.get("document_type") or (metadata.get("document_type") if metadata else None) or "unknown"
    organization = eval_data.get("organization") or (metadata.get("organization") if metadata else None)

    doc_type_labels = {
        "journal_article": "Article de journal",
        "report": "Rapport",
        "working_paper": "Working Paper",
        "thesis": "Thèse",
        "book_chapter": "Chapitre de livre",
        "preprint": "Preprint",
        "unknown": "—",
    }
    doc_type_display = doc_type_labels.get(document_type, document_type)

    # Build meta line
    meta_parts = [f'<span>📅 {year}</span>']
    if document_type not in ("unknown", "journal_article"):
        meta_parts.append(f'<span>📋 {html.escape(doc_type_display)}</span>')
    if organization:
        meta_parts.append(f'<span>🏢 {html.escape(organization)}</span>')
    if journal_name and journal_name != "—" and journal_name.lower() not in ("null", "none"):
        meta_parts.append(f'<span>📰 {html.escape(str(journal_name))}</span>')
    elif document_type in ("unknown", "journal_article"):
        meta_parts.append(f'<span>📰 —</span>')
    meta_parts.append(f'<span>🔗 {html.escape(str(doi))}</span>')

    st.markdown(
        f'<div class="article-card">'
        f'<h2 class="article-title">{html.escape(str(title))}</h2>'
        f'<p class="article-authors">{html.escape(authors_str)}</p>'
        f'<div class="article-meta">'
        + " ".join(meta_parts)
        + f'</div>'
        + (f'<div class="article-keywords">' + " ".join(f'<span class="keyword-tag">{html.escape(k)}</span>' for k in keywords[:8]) + '</div>' if keywords else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Scores ──
    scores = eval_data.get("scores", {})
    if not scores:
        st.warning("Aucune donnée de score disponible.")
        return

    global_score = scores.get("global", {})
    global_val = global_score.get("value", 0) if isinstance(global_score, dict) else 0

    # Global score prominent display
    _render_score_gauge(global_val, "Score global de l'article")

    # Sub-scores in columns
    st.markdown(
        "### Détail des scores"
        " <span title=\"Score = 35% Article + 20% Revue + 15% Auteur + 20% Champ + 10% Réseau — "
        "voir la section Méthodologie ci-dessous pour le détail des justifications.\""
        " style=\"cursor:help;font-size:0.9rem;\">ℹ️</span>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    sub_keys = [
        ("article", "📄 Article", "Impact des citations"),
        ("journal", "📰 Revue", "Prestige de la revue"),
        ("author", "👤 Auteur", "Notoriété des auteurs"),
        ("field_norm", "🏷️ Norme de champ", "Impact ajusté au domaine"),
        ("network", "🌐 Réseau", "Étendue des collaborations"),
    ]
    for col, (key, label, desc) in zip(cols, sub_keys):
        with col:
            sub = scores.get(key, {})
            val = sub.get("score", 0) if isinstance(sub, dict) else 0
            _render_sub_score_card(label, val, desc)

    # Detailed breakdown
    st.markdown("### Composants détaillés")
    detail_tabs = st.tabs(["📄 Article", "📰 Revue", "👤 Auteur", "🏷️ Champ", "🌐 Réseau"])

    with detail_tabs[0]:
        art = scores.get("article", {})
        if isinstance(art, dict):
            c1, c2, c3 = st.columns(3)
            c1.metric("Citations brutes", art.get("raw_citations", "—"))
            c2.metric("Citations (log)", f"{art.get('log_citations', 0):.2f}")
            c3.metric("Percentile", f"{art.get('percentile_citations', 0):.1%}")

    with detail_tabs[1]:
        jour = scores.get("journal", {})
        if isinstance(jour, dict):
            c1, c2, c3, c4 = st.columns(4)
            citedness = jour.get("openalex_mean_citedness")
            c1.metric("Citabilité (2 ans)", f"{citedness:.2f}" if citedness else "—")
            c2.metric("Quartile SCImago", jour.get("scimago_quartile") or "—")
            sjr = jour.get("scimago_sjr")
            c3.metric("SJR", f"{sjr:.3f}" if sjr else "—")
            c4.metric("Score", f"{jour.get('score', 0):.2f}")

    with detail_tabs[2]:
        auth = scores.get("author", {})
        if isinstance(auth, dict):
            author_list = auth.get("authors", [])
            if author_list:
                for a_info in author_list:
                    if isinstance(a_info, dict):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**{a_info.get('display_name', '—')}**")
                        c2.metric("Indice h", a_info.get("h_index", "—"))
                        c3.metric("Citations", f"{a_info.get('cited_by_count', 0):,}")
            else:
                st.caption("Aucune donnée sur les auteurs.")
            st.metric("Indice h agrégé", auth.get("aggregated_h_index", "—"))

    with detail_tabs[3]:
        field = scores.get("field_norm", {})
        if isinstance(field, dict):
            c1, c2, c3 = st.columns(3)
            c1.metric("Domaine", field.get("field_concept_name", "—"))
            pct = field.get("field_percentile")
            c2.metric("Percentile", f"{pct:.1%}" if pct is not None else "—")
            zscore = field.get("field_zscore")
            c3.metric("Score Z", f"{zscore:.2f}" if zscore is not None else "—")

    with detail_tabs[4]:
        net = scores.get("network", {})
        if isinstance(net, dict):
            c1, c2 = st.columns(2)
            c1.metric("Travaux moy. par auteur", f"{net.get('avg_author_works_count', 0):.0f}")
            c2.metric("Institutions", net.get("institutions_count", 0))
            insts = net.get("institutions", [])
            if insts:
                st.caption("Institutions : " + ", ".join(str(i) for i in insts[:10]))

    # Weights
    with st.expander("ℹ️ Méthodologie & pondération des scores", expanded=False):
        # Read current thresholds
        t_cit = int(st.session_state.get("threshold_citations_max", 500))
        t_h = int(st.session_state.get("threshold_h_index_max", 40))
        t_inst = int(st.session_state.get("threshold_institutions_max", 8))
        t_works = int(st.session_state.get("threshold_author_works_max", 200))

        st.markdown(f"""
**Comment est calculé le score de notoriété ?**

Le score global est une moyenne pondérée de **5 dimensions**, chacune normalisée entre 0 et 1 :

| Dimension | Poids | Justification |
|---|---|---|
| 📄 **Article** | **35 %** | Le nombre de citations est l'indicateur le plus objectif et universel de l'impact d'un article. C'est le signal principal car il reflète directement l'engagement de la communauté scientifique. |
| 🏷️ **Norme de champ** | **20 %** | Les citations brutes sont trompeuses entre disciplines (un article très cité en sociologie peut avoir 50 citations, un article moyen en biologie 200). La normalisation par domaine corrige ce biais. |
| 📰 **Revue** | **20 %** | Le prestige de la revue (SJR, quartile SCImago) fournit un signal de qualité indépendant du nombre de citations. Les revues de haut rang ont des processus de relecture plus exigeants. |
| 👤 **Auteur** | **15 %** | Le parcours de l'auteur (indice h) apporte du contexte, mais ne doit pas être surpondéré — un jeune chercheur peut produire un travail excellent. C'est un signal de soutien. |
| 🌐 **Réseau** | **10 %** | La diversité institutionnelle et la productivité collaborative ajoutent un bonus — les études multi-institutionnelles tendent à être plus robustes — mais c'est le prédicteur le plus faible individuellement. |

---

**Grille de notation détaillée :**

**📄 Article** — Formule : `0.5 × percentile + 0.5 × ln(1+citations) / ln(1+{t_cit})`

| Citations | Score ≈ | Interprétation |
|---|---|---|
| 0 | 0.00 | Aucun impact mesurable |
| 10 | ~0.19 | Impact limité |
| 50 | ~0.32 | Impact modéré |
| 100 | ~0.37 | Bon impact |
| 200 | ~0.43 | Très bon impact |
| {t_cit}+ | **1.00** | Impact exceptionnel |

**📰 Revue** — Moyenne de : citedness OpenAlex (log/2.5) + score SCImago (quartile + bonus SJR, /5)

| Quartile | SJR | Score ≈ | Exemple |
|---|---|---|---|
| Q1 | >5 | **0.90-1.00** | Nature, Lancet |
| Q1 | 1-5 | 0.70-0.90 | Bonne revue Q1 |
| Q2 | 0.5-1.5 | 0.50-0.70 | Revue spécialisée solide |
| Q3 | 0.3-0.5 | 0.30-0.50 | Revue de niche |
| Q4 | <0.3 | 0.10-0.25 | Revue à faible impact |

**👤 Auteur** — Formule : `min(h-index, {t_h}) / {t_h}` (h-index max parmi premier + dernier auteur)

| Indice h | Score | Profil type |
|---|---|---|
| 5 | {5/t_h:.2f} | Jeune chercheur |
| 10 | {10/t_h:.2f} | Chercheur en début de carrière |
| 20 | {20/t_h:.2f} | Chercheur confirmé |
| 30 | {30/t_h:.2f} | Chercheur senior reconnu |
| {t_h}+ | **1.00** | Expert de renommée mondiale |

**🏷️ Norme de champ** — Percentile dans le domaine si stats disponibles, sinon même heuristique log que l'article.

**🌐 Réseau** — `0.5 × ln(1+moy_publis)/ln(1+{t_works}) + 0.5 × min(nb_institutions, {t_inst})/{t_inst}`

| Scénario | Score ≈ |
|---|---|
| Chercheur isolé, 1 labo | ~0.30 |
| Équipe de 3 labos, productifs | ~0.65 |
| Grande collaboration internationale ({t_inst}+ institutions) | **1.00** |

---

**Formule globale :**

$$\\text{{Score}} = 0.35 \\times Article + 0.20 \\times Revue + 0.15 \\times Auteur + 0.20 \\times Champ + 0.10 \\times Réseau$$

> 💡 Les seuils (citations max = {t_cit}, h-index max = {t_h}, institutions max = {t_inst}, publications max = {t_works}) sont ajustables dans la barre latérale → **📊 Seuils de notation**.
""")
        weights = global_score.get("weights", {}) if isinstance(global_score, dict) else {}
        if weights:
            st.markdown("**Poids utilisés pour cette analyse :**")
            weight_labels = {
                "article": "📄 Article",
                "journal": "📰 Revue",
                "author": "👤 Auteur",
                "field_norm": "🏷️ Norme de champ",
                "network": "🌐 Réseau",
            }
            wcols = st.columns(5)
            for i, (k, v) in enumerate(weights.items()):
                label = weight_labels.get(k, k)
                wcols[i % 5].metric(label, f"{v:.0%}")

    # Notes
    notes = eval_data.get("notes", [])
    if notes:
        with st.expander("Notes de traitement", expanded=False):
            for n in notes:
                st.caption(f"• {n}")


def _render_score_gauge(value: float, label: str) -> None:
    """Render a large circular score gauge."""
    pct = max(0, min(100, int(value * 100)))
    if value >= 0.7:
        color = PALETTE["success"]
        level = "Élevé"
    elif value >= 0.45:
        color = PALETTE["warning"]
        level = "Modéré"
    else:
        color = PALETTE["danger"]
        level = "Faible"

    st.markdown(
        f"""
        <div class="score-gauge-container">
            <div class="score-gauge" style="--pct:{pct};--color:{color};">
                <div class="score-gauge-inner">
                    <span class="score-gauge-value">{value:.2f}</span>
                    <span class="score-gauge-label">{level}</span>
                </div>
            </div>
            <div class="score-gauge-title">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sub_score_card(label: str, value: float, description: str) -> None:
    pct = max(0, min(100, int(value * 100)))
    if value >= 0.7:
        color = PALETTE["success"]
    elif value >= 0.45:
        color = PALETTE["warning"]
    else:
        color = PALETTE["danger"]

    st.markdown(
        f"""
        <div class="sub-score-card">
            <div class="sub-score-label">{label}</div>
            <div class="sub-score-value" style="color:{color};">{value:.2f}</div>
            <div class="sub-score-bar">
                <div class="sub-score-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <div class="sub-score-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────── Effects Tab ─────────────────────────────

def _render_effects_tab() -> None:
    """Tab focused purely on extracted effect sizes."""
    output_dir = _get_last_output_dir()
    if not output_dir:
        st.info("Lancez le pipeline d'abord pour voir les effets.")
        _render_folder_picker(key_suffix="effects")
        return

    effects_data = _load_json(output_dir / "04_effects.json")

    if not effects_data or not isinstance(effects_data, dict):
        st.warning("Aucune donnée d'effets trouvée.")
        return

    effects = effects_data.get("effects", [])
    if not isinstance(effects, list) or not effects:
        st.info("Aucun effet extrait.")
        return

    study_rows, cited_rows, model_rows = _categorize_effects(effects)

    # ── KPI row ──
    n_study = len(study_rows)
    n_cited = len(cited_rows)
    n_model = len(model_rows)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Effets de l'étude", n_study)
    k2.metric("Effets cités", n_cited)
    k3.metric("Stats de modèle", n_model)
    k4.metric("Total effets", n_study + n_cited + n_model)

    # ── Study effects ──
    st.markdown("### Effets de l'étude")
    if study_rows:
        st.dataframe(study_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun effet de l'étude extrait.")

    # ── Cited effects ──
    st.markdown("### Effets cités dans la littérature")
    if cited_rows:
        st.dataframe(cited_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun effet cité extrait.")

    # ── Model stats ──
    st.markdown("### Statistiques de modèle")
    if model_rows:
        st.dataframe(model_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucune statistique de modèle détectée.")

    # ── Notes d'extraction ──
    extraction_notes = effects_data.get("notes", [])
    if extraction_notes:
        with st.expander("Notes d'extraction", expanded=False):
            for note in extraction_notes:
                st.caption(f"• {note}")


def _render_study_tab() -> None:
    """Tab focused on study design and internal methodological quality only."""
    output_dir = _get_last_output_dir()
    if not output_dir:
        st.info("Lancez le pipeline d'abord pour voir l'analyse de l'étude.")
        _render_folder_picker(key_suffix="study")
        return

    quality_data = _load_json(output_dir / "05_quality_quick.json")

    # ── Section 1 : Design de l'étude ──
    st.markdown("## Design de l'étude")
    if quality_data and isinstance(quality_data, dict):
        design = quality_data.get("study_design", "unknown")
        design_just = quality_data.get("study_design_justification", "")
        sample_n = quality_data.get("sample_size_n")
        internal_score = quality_data.get("internal_quality_score", 0)

        design_labels = {
            "RCT": ("Essai contrôlé randomisé", PALETTE["success"]),
            "quasi_experimental": ("Quasi-expérimental", PALETTE["warning"]),
            "observational_longitudinal": ("Observationnel longitudinal", PALETTE["warning"]),
            "observational_cross_sectional": ("Observationnel transversal", PALETTE["warning"]),
            "meta_analysis": ("Méta-analyse", PALETTE["success"]),
            "case_study": ("Étude de cas", PALETTE["danger"]),
            "unknown": ("Inconnu", PALETTE["text_secondary"]),
        }
        design_label, design_color = design_labels.get(design, ("Inconnu", PALETTE["text_secondary"]))

        d1, d2, d3 = st.columns(3)
        d1.markdown(
            f'<div style="padding:1rem;border-radius:8px;border-left:4px solid {design_color};background:white;">'
            f'<div style="font-size:0.8rem;color:{PALETTE["text_secondary"]};">Type de design</div>'
            f'<div style="font-size:1.3rem;font-weight:600;color:{design_color};">{design_label}</div>'
            f'<div style="font-size:0.78rem;color:{PALETTE["text_secondary"]};margin-top:0.3rem;">{design_just[:150]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        n_str = f"N = {sample_n:,}" if sample_n else "Non rapporté"
        n_color = PALETTE["success"] if sample_n and sample_n >= 30 else (
            PALETTE["warning"] if sample_n else PALETTE["text_secondary"]
        )
        d2.markdown(
            f'<div style="padding:1rem;border-radius:8px;border-left:4px solid {n_color};background:white;">'
            f'<div style="font-size:0.8rem;color:{PALETTE["text_secondary"]};">Taille d\'échantillon</div>'
            f'<div style="font-size:1.3rem;font-weight:600;color:{n_color};">{n_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        d3.markdown(
            f'<div style="padding:1rem;border-radius:8px;border-left:4px solid {_score_color(internal_score)};background:white;">'
            f'<div style="font-size:0.8rem;color:{PALETTE["text_secondary"]};">Score qualité interne</div>'
            f'<div style="font-size:1.3rem;font-weight:600;color:{_score_color(internal_score)};">{internal_score:.2f}/1</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Données de design non disponibles.")

    st.markdown("---")

    # ── Section 2 : Indicateurs méthodologiques ──
    st.markdown("## Indicateurs méthodologiques")
    if quality_data and isinstance(quality_data, dict):
        flags = [
            ("Randomisation", quality_data.get("randomization", "unclear"), quality_data.get("randomization_justification", "")),
            ("Groupe contrôle", quality_data.get("control_group", "unclear"), quality_data.get("control_group_justification", "")),
            ("Taille d'échantillon", quality_data.get("sample_size_reported", "unclear"), quality_data.get("sample_size_justification", "")),
            ("Attrition", quality_data.get("attrition_reported", "unclear"), quality_data.get("attrition_justification", "")),
            ("Aveugle", quality_data.get("blinding_reported", "unclear"), quality_data.get("blinding_justification", "")),
        ]
        qcols = st.columns(5)
        for col, (name, val, justif) in zip(qcols, flags):
            if val == "yes":
                icon, badge_color = "✓", PALETTE["success"]
            elif val == "no":
                icon, badge_color = "✗", PALETTE["danger"]
            else:
                icon, badge_color = "?", PALETTE["warning"]
            col.markdown(
                f'<div style="padding:0.7rem;border-radius:8px;background:white;text-align:center;min-height:120px;">'
                f'<div style="display:inline-block;width:32px;height:32px;border-radius:50%;'
                f'background:{badge_color};color:white;font-weight:700;line-height:32px;font-size:1rem;">{icon}</div>'
                f'<div style="font-size:0.82rem;font-weight:600;margin-top:0.3rem;">{name}</div>'
                f'<div style="font-size:0.72rem;color:{PALETTE["text_secondary"]};margin-top:0.2rem;line-height:1.3;">'
                f'{justif[:100] if justif else "Aucune justification"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Overall justification
        overall_just = quality_data.get("justification", "")
        if overall_just and overall_just != "unknown":
            st.markdown(
                f'<div style="padding:0.6rem 1rem;margin-top:0.5rem;border-radius:6px;background:#F1F5F9;'
                f'font-size:0.82rem;color:{PALETTE["text_secondary"]};">'
                f'<strong>Justification générale :</strong> {overall_just}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Données de qualité non disponibles.")

    st.markdown("---")

    # ── Section 3 : Limites & prochaines étapes ──
    st.markdown("## Limites de l'analyse")
    st.info(
        "L'analyse méthodologique actuelle se base sur la détection du design d'étude "
        "et 5 indicateurs (randomisation, groupe contrôle, taille d'échantillon, attrition, aveugle). "
        "Une analyse plus approfondie — grilles RoB2 pour les RCT, ROBINS-I pour les quasi-expérimentaux, "
        "évaluation des biais de publication, etc. — sera implémentée dans une prochaine version."
    )


# ──────────────────────── Report Tab ──────────────────────────────

def _render_report_tab() -> None:
    """Tab showing a unified summary: scores, key effects, and LLM narrative."""
    output_dir = _get_last_output_dir()
    if not output_dir:
        st.info("Lancez le pipeline d'abord pour voir le rapport.")
        _render_folder_picker(key_suffix="report")
        return

    # Load all data
    effects_data = _load_json(output_dir / "04_effects.json")
    quality_data = _load_json(output_dir / "05_quality_quick.json")
    credibility_data = _load_json(output_dir / "06_external_credibility.json")
    reliability_data = _load_json(output_dir / "07_summary_score.json")
    metadata = _load_json(output_dir / "00_metadata.json")

    # Try to load article evaluation
    article_eval_path = output_dir / "08_article_evaluation.json"
    article_eval = _load_json(article_eval_path)

    # ── Header : titre de l'article ──
    title = "Article analysé"
    if metadata and isinstance(metadata, dict):
        title = metadata.get("title") or metadata.get("filename", "Article analysé")
    st.markdown(f"## {title}")

    st.markdown("---")

    # ── Section 1 : Scores côte à côte ──
    st.markdown("### Scores")
    s1, s2 = st.columns(2)

    # Article score
    article_score = 0.0
    if article_eval and isinstance(article_eval, dict):
        scores = article_eval.get("scores", {})
        if isinstance(scores, dict):
            g = scores.get("global", {})
            if isinstance(g, dict):
                article_score = float(g.get("value", 0))
            elif isinstance(g, (int, float)):
                article_score = float(g)
    s1.markdown(
        f'<div style="padding:1rem;border-radius:8px;border-left:4px solid {_score_color(article_score)};background:white;text-align:center;">'
        f'<div style="font-size:0.8rem;color:{PALETTE["text_secondary"]};">Score Article</div>'
        f'<div style="font-size:2rem;font-weight:700;color:{_score_color(article_score)};">{article_score:.2f}</div>'
        f'<div style="font-size:0.75rem;color:{PALETTE["text_secondary"]};">Revue, citations, auteurs</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Study quality score
    study_score = 0.0
    study_design_label = "—"
    if quality_data and isinstance(quality_data, dict):
        study_score = quality_data.get("internal_quality_score", 0)
        design = quality_data.get("study_design", "unknown")
        design_map = {
            "RCT": "RCT", "quasi_experimental": "Quasi-exp.", "meta_analysis": "Méta-analyse",
            "observational_longitudinal": "Longitudinal", "observational_cross_sectional": "Transversal",
            "case_study": "Étude de cas",
        }
        study_design_label = design_map.get(design, "Inconnu")
    s2.markdown(
        f'<div style="padding:1rem;border-radius:8px;border-left:4px solid {_score_color(study_score)};background:white;text-align:center;">'
        f'<div style="font-size:0.8rem;color:{PALETTE["text_secondary"]};">Score Étude</div>'
        f'<div style="font-size:2rem;font-weight:700;color:{_score_color(study_score)};">{study_score:.2f}</div>'
        f'<div style="font-size:0.75rem;color:{PALETTE["text_secondary"]};">Design : {study_design_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Section 2 : Effets clés ──
    st.markdown("### Effets clés")
    if effects_data and isinstance(effects_data, dict):
        effects = effects_data.get("effects", [])
        study_effects = [e for e in effects if isinstance(e, dict) and e.get("effect_scope") == "study_effect"]
        if study_effects:
            # Summary table: just predictor + value + type
            summary_rows = []
            for e in study_effects[:15]:
                spec = e.get("result_spec", {}) or {}
                group, domain, predictor = derive_group_domain_predictor(
                    group_raw=str(e.get("grouping_label") or spec.get("groups") or ""),
                    predictor_raw=str(spec.get("outcome") or ""),
                    domain_raw=str(e.get("outcome_label_normalized") or ""),
                    context=str(e.get("quote", "")),
                )
                val = e.get("value")
                et = str(e.get("effect_type", "?"))
                ci_l = e.get("ci_low")
                ci_h = e.get("ci_high")
                summary_rows.append({
                    "Groupe": group,
                    "Prédicteur": predictor,
                    "Domaine": domain,
                    "Taille d'effet": _fmt_effect(et, val, ci_l, ci_h),
                    "Page": e.get("source_page", "—"),
                })
            summary_rows.sort(key=lambda r: _abs_val(r.get("Taille d'effet", "")), reverse=True)
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)
            if len(study_effects) > 15:
                st.caption(f"... et {len(study_effects) - 15} autres effets (voir onglet Effets).")
        else:
            st.caption("Aucun effet de l'étude extrait.")
    else:
        st.caption("Aucune donnée d'effets.")

    st.markdown("---")

    # ── Section 3 : Résumé narratif (LLM) ──
    st.markdown("### Résumé")
    _render_llm_summary(output_dir, metadata, quality_data, effects_data, article_eval, reliability_data)

    # ── Markdown report if available ──
    report_path = output_dir / "report.md"
    if report_path.exists():
        with st.expander("Rapport technique complet", expanded=False):
            st.markdown(report_path.read_text(encoding="utf-8", errors="replace"))


def _render_llm_summary(
    output_dir: Path,
    metadata: Optional[Dict],
    quality_data: Optional[Dict],
    effects_data: Optional[Dict],
    article_eval: Optional[Dict],
    reliability_data: Optional[Dict],
) -> None:
    """Generate and display an LLM narrative summary, cached to disk."""
    import os

    cache_path = output_dir / "report_summary.md"

    # If already generated, show it
    if cache_path.exists():
        summary_text = cache_path.read_text(encoding="utf-8", errors="replace")
        st.markdown(summary_text)
        if st.button("Regénérer le résumé", key="regen_summary"):
            cache_path.unlink(missing_ok=True)
            st.rerun()
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.warning("Clé OPENAI_API_KEY manquante — le résumé narratif nécessite un LLM.")
        return

    # Build context for the LLM
    context_parts: List[str] = []

    if metadata and isinstance(metadata, dict):
        context_parts.append(f"Titre: {metadata.get('title', 'inconnu')}")
        context_parts.append(f"Fichier: {metadata.get('filename', 'inconnu')}")

    if article_eval and isinstance(article_eval, dict):
        scores = article_eval.get("scores", {})
        context_parts.append(f"Score article global: {scores.get('global', {}).get('value', '?')}")
        for key in ["revue", "citations", "auteurs", "doi", "open_access"]:
            sub = scores.get(key, {})
            if isinstance(sub, dict) and "value" in sub:
                context_parts.append(f"  {key}: {sub['value']}")

    if quality_data and isinstance(quality_data, dict):
        context_parts.append(f"Design d'étude: {quality_data.get('study_design', 'inconnu')}")
        context_parts.append(f"Justification design: {quality_data.get('study_design_justification', '')}")
        context_parts.append(f"N: {quality_data.get('sample_size_n', 'non rapporté')}")
        context_parts.append(f"Score qualité interne: {quality_data.get('internal_quality_score', '?')}")
        for flag in ["randomization", "control_group", "sample_size_reported", "attrition_reported", "blinding_reported"]:
            val = quality_data.get(flag, "unclear")
            justif = quality_data.get(f"{flag}_justification", "")
            context_parts.append(f"  {flag}: {val} — {justif}")

    if effects_data and isinstance(effects_data, dict):
        effects = effects_data.get("effects", [])
        study_fx = [e for e in effects if isinstance(e, dict) and e.get("effect_scope") == "study_effect"]
        context_parts.append(f"Nombre d'effets de l'étude: {len(study_fx)}")
        for e in study_fx[:10]:
            spec = e.get("result_spec", {}) or {}
            outcome = spec.get("outcome", "?")
            val = e.get("value", "?")
            et = e.get("effect_type", "?")
            context_parts.append(f"  Effet: {outcome} — {et}={val}")

    if reliability_data and isinstance(reliability_data, dict):
        context_parts.append(f"Fiabilité globale: {reliability_data.get('global_score', '?')}")
        context_parts.append(f"Conclusion: {reliability_data.get('conclusion', '?')}")

    context_text = "\n".join(context_parts)

    with st.spinner("Génération du résumé..."):
        try:
            import requests as req_module
            response = req_module.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4.1-mini",
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": (
                            "Tu es un assistant scientifique. Tu rédiges un résumé clair et structuré en français "
                            "d'une analyse d'article scientifique. Le résumé doit être accessible, factuel et "
                            "bien organisé. Utilise des sections avec des titres en markdown (####). "
                            "Sections attendues :\n"
                            "1. **Présentation** - De quoi parle l'article\n"
                            "2. **Qualité de l'article** - Score, revue, citations\n"
                            "3. **Méthodologie de l'étude** - Design, échantillon, indicateurs\n"
                            "4. **Effets observés** - Principaux résultats avec valeurs de d\n"
                            "5. **Conclusion** - Synthèse de la fiabilité et utilisabilité\n"
                            "Sois concis (300-500 mots max). Ne fabrique aucune donnée."
                        )},
                        {"role": "user", "content": f"Voici les données de l'analyse :\n\n{context_text}"},
                    ],
                },
                timeout=60,
            )
            response.raise_for_status()
            summary_text = response.json()["choices"][0]["message"]["content"]
            # Cache to disk
            cache_path.write_text(summary_text, encoding="utf-8")
            st.markdown(summary_text)
        except Exception as exc:
            st.error(f"Erreur lors de la génération du résumé : {exc}")


def _categorize_effects(effects: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    study: List[Dict] = []
    cited: List[Dict] = []
    model: List[Dict] = []
    for effect in effects:
        if not isinstance(effect, dict):
            continue
        spec = effect.get("result_spec", {}) or {}
        group, domain, predictor = derive_group_domain_predictor(
            group_raw=str(effect.get("grouping_label") or spec.get("groups") or ""),
            predictor_raw=str(spec.get("outcome") or ""),
            domain_raw=str(effect.get("outcome_label_normalized") or ""),
            context=str(effect.get("quote", "")),
        )
        value = effect.get("value")
        effect_type = str(effect.get("effect_type", "?"))
        ci_low = effect.get("ci_low")
        ci_high = effect.get("ci_high")
        p_value = effect.get("p_value")
        sample_size = effect.get("sample_size")
        effect_size = _fmt_effect(effect_type, value, ci_low, ci_high)
        duration = str(effect.get("timepoint_label_normalized") or spec.get("timepoint") or "—")

        # P-value formatting
        p_str = "—"
        if p_value is not None:
            try:
                pf = float(p_value)
                p_str = f"{pf:.4f}" if pf >= 0.001 else f"<0.001"
            except (ValueError, TypeError):
                pass

        # Sample size
        n_str = str(sample_size) if sample_size is not None else "—"

        # Quote / passage élargi
        quote_raw = str(effect.get("quote", ""))
        quote_display = quote_raw[:200] if quote_raw else "—"

        row = {
            "Groupe": group,
            "Domaine": domain,
            "Prédicteur": predictor,
            "Taille d'effet": effect_size,
            "p-value": p_str,
            "N": n_str,
            "Durée": duration,
            "Passage source": quote_display,
            "Source": str(effect.get("source_kind", "—")),
            "Page": effect.get("source_page", "—"),
        }
        scope = str(effect.get("effect_scope", "study_effect"))
        if scope == "literature_cited":
            cited.append(row)
        elif scope == "model_stat":
            model.append(row)
        else:
            study.append(row)

    study.sort(key=lambda r: _abs_val(r.get("Taille d'effet", "")), reverse=True)
    cited.sort(key=lambda r: _abs_val(r.get("Taille d'effet", "")), reverse=True)
    model.sort(key=lambda r: _abs_val(r.get("Taille d'effet", "")), reverse=True)
    return study, cited, model


# ──────────────────────── Helpers ─────────────────────────────────

def _get_last_output_dir() -> Optional[Path]:
    """Get the output directory from session state or let user pick."""
    dir_str = st.session_state.get("last_output_dir")
    if dir_str:
        p = Path(dir_str)
        if p.exists():
            return p
    return None


def _render_folder_picker(key_suffix: str = "default") -> None:
    """Let user type a path to a previous analysis folder."""
    folder = st.text_input(
        "Ou saisissez le chemin d'une analyse précédente :",
        help="ex. outputs/1771338302-nihms-100109-1-8037d1af",
        key=f"folder_picker_path_{key_suffix}",
    )
    if folder and Path(folder).exists():
        st.session_state["last_output_dir"] = folder
        st.rerun()


def _persist_uploaded_pdf(filename: str, content: bytes) -> Path:
    upload_dir = Path("outputs") / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename).strip("_")
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    safe_name = safe_name or "uploaded.pdf"
    out = upload_dir / f"{timestamp}_{safe_name}"
    out.write_bytes(content)
    return out.resolve()


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fmt_effect(effect_type: str, value: Any, ci_low: Any, ci_high: Any) -> str:
    v = _as_float(value)
    if v is None:
        return f"{effect_type}=n/a"
    cl = _as_float(ci_low)
    ch = _as_float(ci_high)
    if cl is not None and ch is not None:
        return f"{effect_type}={v:.3f} [{cl:.3f}; {ch:.3f}]"
    return f"{effect_type}={v:.3f}"


def _abs_val(effect_str: Any) -> float:
    m = re.search(r"=\s*([+-]?\d+(?:\.\d+)?)", str(effect_str))
    if m:
        try:
            return abs(float(m.group(1)))
        except Exception:
            pass
    return 0.0


def _score_color(value: float) -> str:
    """Return a color from PALETTE based on a 0-1 score."""
    if value >= 0.7:
        return PALETTE["success"]
    if value >= 0.45:
        return PALETTE["warning"]
    return PALETTE["danger"]


def _as_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val))
    except Exception:
        return None


def _render_output_json_expanders(output_dir: Path) -> None:
    json_files = sorted(output_dir.glob("*.json"))
    for jf in json_files:
        with st.expander(jf.name, expanded=False):
            size_kb = jf.stat().st_size / 1024
            st.caption(f"Taille : {size_kb:.1f} Ko")
            if size_kb > 1800:
                st.warning("Fichier volumineux — aperçu seulement.")
                text = jf.read_text(encoding="utf-8", errors="replace")[:12000]
                st.code(text, language="json")
            else:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    st.json(data)
                except Exception as exc:
                    st.error(f"Erreur de lecture JSON : {exc}")


# ──────────────────────── CSS ─────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --primary: #1B2A4A;
            --accent: #3B82F6;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --bg: #F8FAFC;
            --card: #FFFFFF;
            --text: #0F172A;
            --text2: #64748B;
            --border: #E2E8F0;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: #F1F5F9 !important;
            border-right: 1px solid var(--border) !important;
        }

        /* ── Header ── */
        .app-header {
            padding: 1.2rem 0 0.8rem;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 1.5rem;
        }
        .app-logo {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .app-logo span { color: var(--accent); }
        .app-subtitle {
            color: var(--text2);
            font-size: 0.95rem;
            margin-top: 0.15rem;
        }

        /* ── Sidebar ── */
        .sidebar-title {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
            margin-bottom: 0.5rem;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.2rem;
            color: var(--text2);
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 3px solid var(--accent);
            background: rgba(59,130,246,0.05);
        }

        /* ── Article card ── */
        .article-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .article-title {
            color: var(--primary) !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.4rem !important;
            line-height: 1.3 !important;
        }
        .article-authors {
            color: var(--text2) !important;
            font-size: 0.92rem !important;
            margin: 0 0 0.6rem !important;
        }
        .article-meta {
            display: flex;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text2);
            flex-wrap: wrap;
        }
        .article-keywords {
            margin-top: 0.6rem;
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }
        .keyword-tag {
            background: rgba(59,130,246,0.08);
            color: var(--accent);
            font-size: 0.75rem;
            padding: 0.15rem 0.55rem;
            border-radius: 20px;
            font-weight: 500;
        }

        /* ── Score gauge ── */
        .score-gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 1.5rem 0;
        }
        .score-gauge {
            width: 140px; height: 140px;
            border-radius: 50%;
            background: conic-gradient(
                var(--color) calc(var(--pct) * 1%),
                #E2E8F0 calc(var(--pct) * 1%)
            );
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .score-gauge-inner {
            width: 110px; height: 110px;
            border-radius: 50%;
            background: var(--card);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .score-gauge-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text);
        }
        .score-gauge-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text2);
        }
        .score-gauge-title {
            margin-top: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            color: var(--primary);
        }

        /* ── Sub-score cards ── */
        .sub-score-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .sub-score-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.3rem;
        }
        .sub-score-value {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .sub-score-bar {
            height: 6px;
            background: #E2E8F0;
            border-radius: 3px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        .sub-score-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .sub-score-desc {
            font-size: 0.72rem;
            color: var(--text2);
        }

        /* ── Buttons ── */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--accent) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: #2563EB !important;
        }

        /* ── Metrics ── */
        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.8rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--text2) !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-weight: 700 !important;
        }

        /* ── Expanders ── */
        [data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            background: var(--card) !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--card) !important;
            border: 2px dashed var(--accent) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--success) !important;
            background: #F0FDF4 !important;
        }

        /* ── Dataframe ── */
        .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }

        /* ── Spinner animation ── */
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* ── Typography overrides ── */
        h1, h2, h3 { color: var(--primary) !important; }
        h1 { font-size: 1.6rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
