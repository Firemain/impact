"""demo_validation.py — Script de validation pour la présentation.

Charge les résultats d'une analyse existante et affiche un résumé
structuré qui peut servir de support visuel pendant la démo.

Usage:
    python demo_validation.py outputs/1772018049-nihms-100109-8037d1af
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_score(val: float | None) -> str:
    if val is None:
        return "—"
    pct = int(val * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return f"{val:.2f} [{bar}]"


def main(output_dir: str) -> None:
    d = Path(output_dir)
    if not d.exists():
        print(f"Dossier introuvable : {d}")
        sys.exit(1)

    meta = load(d / "00_metadata.json")
    effects = load(d / "04_effects.json")
    quality = load(d / "05_quality_quick.json")
    credibility = load(d / "06_external_credibility.json")
    article_eval = load(d / "08_article_evaluation.json")
    reliability = load(d / "07_summary_score.json")

    print("=" * 70)
    print("  IMPACT — Résumé de l'analyse")
    print("=" * 70)

    # ── Métadonnées ──
    print("\n📄 MÉTADONNÉES")
    print("-" * 50)
    if article_eval:
        print(f"  Titre    : {article_eval.get('title', '?')}")
        authors = article_eval.get("authors_extracted", [])
        names = [a.get("full_name", "?") if isinstance(a, dict) else str(a) for a in authors[:4]]
        print(f"  Auteurs  : {', '.join(names)}")
        journal = article_eval.get("journal_extracted", {})
        print(f"  Revue    : {journal.get('name', '—') if isinstance(journal, dict) else '—'}")
        print(f"  Année    : {article_eval.get('year', '—')}")
        print(f"  DOI      : {article_eval.get('doi', '—')}")
        print(f"  Type     : {article_eval.get('document_type', 'journal_article')}")
        org = article_eval.get("organization")
        if org:
            print(f"  Org.     : {org}")
    elif meta:
        print(f"  Titre    : {meta.get('title', '?')}")
        print(f"  Auteurs  : {', '.join(meta.get('authors', ['?']))}")

    # ── Score Article ──
    print("\n📊 SCORE DE RENOMMÉE")
    print("-" * 50)
    if article_eval and "scores" in article_eval:
        scores = article_eval["scores"]
        g = scores.get("global", {})
        global_val = g.get("value", 0) if isinstance(g, dict) else 0
        print(f"  Global           : {fmt_score(global_val)}")
        for dim in ["article", "journal", "author", "field_norm", "network"]:
            sub = scores.get(dim, {})
            val = sub.get("score", 0) if isinstance(sub, dict) else 0
            label = {
                "article": "Article (citations)",
                "journal": "Revue (SCImago)",
                "author": "Auteur (h-index)",
                "field_norm": "Champ (normalisation)",
                "network": "Réseau (institutions)",
            }.get(dim, dim)
            print(f"  {label:<22}: {fmt_score(val)}")

        # Details
        art = scores.get("article", {})
        if isinstance(art, dict) and "raw_citations" in art:
            print(f"\n  → Citations : {art['raw_citations']}")
        auth = scores.get("author", {})
        if isinstance(auth, dict) and "aggregated_h_index" in auth:
            print(f"  → h-index max : {auth['aggregated_h_index']}")
        j = scores.get("journal", {})
        if isinstance(j, dict) and "scimago_quartile" in j:
            print(f"  → SCImago : {j['scimago_quartile']} (SJR={j.get('scimago_sjr', '?')})")
    else:
        print("  [Données non disponibles]")

    # ── Effets ──
    print("\n🔬 EFFETS EXTRAITS")
    print("-" * 50)
    if effects and isinstance(effects, dict):
        eff_list = effects.get("effects", [])
        study_fx = [e for e in eff_list if isinstance(e, dict) and e.get("effect_scope") == "study_effect"]
        cited_fx = [e for e in eff_list if isinstance(e, dict) and e.get("effect_scope") == "cited_effect"]
        model_fx = [e for e in eff_list if isinstance(e, dict) and e.get("effect_scope") == "model_stat"]
        print(f"  Total         : {len(eff_list)}")
        print(f"  Étude         : {len(study_fx)}")
        print(f"  Cités         : {len(cited_fx)}")
        print(f"  Stats modèle  : {len(model_fx)}")

        if study_fx:
            print(f"\n  {'Groupe':<12} {'Outcome':<25} {'Type':<5} {'Valeur':<8} {'Source':<20} {'Page':<5}")
            print(f"  {'─'*12} {'─'*25} {'─'*5} {'─'*8} {'─'*20} {'─'*5}")
            for e in study_fx[:10]:
                spec = e.get("result_spec", {}) or {}
                group = str(e.get("grouping_label") or spec.get("groups") or "—")[:12]
                outcome = str(spec.get("outcome") or "—")[:25]
                et = str(e.get("effect_type", "?"))
                val = e.get("value")
                val_str = f"{val:+.2f}" if val is not None else "—"
                source = str(e.get("source_kind", "—"))[:20]
                page = str(e.get("source_page", "—"))
                print(f"  {group:<12} {outcome:<25} {et:<5} {val_str:<8} {source:<20} {page:<5}")

            # Quotes
            print(f"\n  📝 Passages source :")
            for i, e in enumerate(study_fx[:3], 1):
                quote = e.get("quote", "")
                if quote:
                    print(f"  [{i}] \"{quote[:120]}{'…' if len(quote) > 120 else ''}\"")
    else:
        print("  [Aucun effet extrait]")

    # ── Qualité méthodologique ──
    print("\n🧪 QUALITÉ MÉTHODOLOGIQUE")
    print("-" * 50)
    if quality and isinstance(quality, dict):
        design = quality.get("study_design", "unknown")
        design_labels = {
            "RCT": "Essai contrôlé randomisé",
            "quasi_experimental": "Quasi-expérimental",
            "observational_longitudinal": "Longitudinal",
            "observational_cross_sectional": "Transversal",
            "meta_analysis": "Méta-analyse",
            "case_study": "Étude de cas",
        }
        print(f"  Design         : {design_labels.get(design, design)}")
        print(f"  Justification  : {(quality.get('study_design_justification') or '—')[:100]}")
        n = quality.get("sample_size_n")
        print(f"  N              : {n if n else '—'}")
        print(f"  Score interne  : {fmt_score(quality.get('internal_quality_score', 0))}")

        flags = [
            ("Randomisation", "randomization"),
            ("Groupe contrôle", "control_group"),
            ("Taille échant.", "sample_size_reported"),
            ("Attrition", "attrition_reported"),
            ("Aveugle", "blinding_reported"),
        ]
        print(f"\n  {'Indicateur':<18} {'Statut':<10} {'Justification':<50}")
        print(f"  {'─'*18} {'─'*10} {'─'*50}")
        for label, key in flags:
            val = quality.get(key, "unclear")
            icon = "✅ oui" if val == "yes" else ("❌ non" if val == "no" else "❔ ??")
            justif = (quality.get(f"{key}_justification") or "—")[:50]
            print(f"  {label:<18} {icon:<10} {justif:<50}")
    else:
        print("  [Données non disponibles]")

    # ── Crédibilité externe ──
    print("\n🌐 CRÉDIBILITÉ EXTERNE")
    print("-" * 50)
    if credibility and isinstance(credibility, dict):
        print(f"  Score          : {fmt_score(credibility.get('external_score', 0))}")
        print(f"  Niveau         : {credibility.get('credibility_level', '—')}")
        print(f"  Venue          : {credibility.get('venue', '—')}")
        print(f"  Publisher      : {credibility.get('publisher', '—')}")
        cit = credibility.get("citation_count")
        print(f"  Citations      : {cit if cit is not None else '—'}")
    else:
        print("  [Données non disponibles]")

    # ── Fiabilité globale ──
    print("\n📋 SCORE GLOBAL")
    print("-" * 50)
    if reliability and isinstance(reliability, dict):
        print(f"  Score global   : {fmt_score(reliability.get('global_score', 0))}")
        print(f"  Conclusion     : {reliability.get('conclusion', '—')}")
        items = reliability.get("items", [])
        if items:
            print(f"  Effets notés   : {len(items)}")
    else:
        print("  [Données non disponibles]")

    # ── Coûts estimés ──
    print("\n💰 ESTIMATION DE COÛT")
    print("-" * 50)
    if article_eval:
        notes = article_eval.get("notes", [])
        n_steps = len([n for n in notes if "=" in n])
        print(f"  Steps exécutés : {n_steps}")
    print(f"  Modèle         : gpt-4.1-mini")
    print(f"  Calls API est. : ~8-15 par PDF")
    print(f"  Coût estimé    : ~0.02–0.08 € / PDF")

    print("\n" + "=" * 70)
    print("  Fin du résumé")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to latest analysis
        outputs = Path("outputs")
        dirs = sorted(
            [d for d in outputs.iterdir() if d.is_dir() and not d.name.startswith("_")],
            key=lambda x: x.name,
            reverse=True,
        )
        if dirs:
            print(f"[Auto-sélection du dernier dossier : {dirs[0].name}]\n")
            main(str(dirs[0]))
        else:
            print("Usage: python demo_validation.py <dossier_analyse>")
            sys.exit(1)
    else:
        main(sys.argv[1])
