#!/usr/bin/env python3
"""Generate presentation charts for slides 8 (Effect Reliability) and 9 (Article Reliability).

Outputs PNGs in outputs/_slides/ ready to paste into the presentation.
Usage:
    python generate_slides_charts.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("outputs/_slides")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Couleurs cohérentes avec la présentation
GOLD = "#F5B731"
GOLD_LIGHT = "#FDE68A"
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
GREEN = "#16A34A"
GREEN_LIGHT = "#86EFAC"
RED = "#DC2626"
GRAY = "#6B7280"
DARK = "#1F2937"
WHITE = "#FFFFFF"
BG = "#F8FAFC"

# =====================================================================
# SLIDE 8 — FIABILITE EFFECT
# =====================================================================

def slide8_calc_score_breakdown():
    """Waterfall chart showing how calc_score is built for one effect."""
    
    # Components of calc_score (from reliability.py _calc_score)
    components = [
        ("Base\n(valeur présente)", 0.35, GREEN),
        ("Type reconnu\n(d / g / SMD)", 0.15, BLUE),
        ("Intervalle de\nconfiance (CI)", 0.10, BLUE),
        ("CI étroit\n(largeur ≤ 1.0)", 0.05, BLUE_LIGHT),
        ("p-value\nrapportée", 0.05, GOLD),
        ("p < 0.05", 0.05, GOLD),
        ("Taille échant.\nrapportée", 0.05, GOLD_LIGHT),
        ("N ≥ 30", 0.05, GOLD_LIGHT),
        ("Consistance\nstat. (pass)", 0.10, GREEN_LIGHT),
    ]
    
    penalties = [
        ("Non dérivable\n(confidence)", -0.20, RED),
        ("N < 15\n(échantillon)", -0.10, RED),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    
    # Draw waterfall
    x_pos = 0
    running = 0
    bar_width = 0.7
    
    labels = []
    for name, val, color in components:
        ax.bar(x_pos, val, bottom=running, width=bar_width, color=color, 
               edgecolor=WHITE, linewidth=1.5, zorder=3)
        # Value label inside bar
        ax.text(x_pos, running + val/2, f"+{val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=DARK)
        running += val
        labels.append(name)
        x_pos += 1
    
    # Dotted line at running total
    ax.axhline(y=running, color=GREEN, linestyle="--", linewidth=1, alpha=0.5, zorder=1)
    ax.text(x_pos - 0.5, running + 0.02, f"Max = {running:.2f}", ha="center",
            fontsize=10, fontweight="bold", color=GREEN)
    
    # Penalties
    x_pos += 0.5  # gap
    for name, val, color in penalties:
        ax.bar(x_pos, abs(val), bottom=running + val, width=bar_width, color=color,
               edgecolor=WHITE, linewidth=1.5, alpha=0.4, zorder=3,
               hatch="//")
        ax.text(x_pos, running + val/2, f"{val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=RED)
        labels.append(name)
        x_pos += 1
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Calcul du score de qualité d'un effet (calc_score)", 
                 fontsize=14, fontweight="bold", pad=15, color=DARK)
    ax.set_ylim(-0.05, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend
    bonus_patch = mpatches.Patch(color=BLUE, label="Bonus (présent)")
    penalty_patch = mpatches.Patch(color=RED, alpha=0.4, hatch="//", label="Pénalité (si applicable)")
    ax.legend(handles=[bonus_patch, penalty_patch], loc="upper left", fontsize=10)
    
    plt.tight_layout()
    path = OUT_DIR / "slide8_calc_score_waterfall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def slide8_reliability_formula():
    """Horizontal stacked bar showing the reliability formula weights."""
    
    # Real data from Doss et al.
    components = {
        "Qualité de l'effet\n(calc_score)": (0.40, 0.75, BLUE),
        "Qualité de l'étude\n(internal)": (0.20, 0.60, GOLD),
        "Crédibilité externe\n(external)": (0.15, 1.00, GREEN),
        "Évaluation article\n(article_eval)": (0.25, 0.91, "#8B5CF6"),
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor(WHITE)
    
    # Left: Pie chart of weights
    ax1.set_facecolor(WHITE)
    labels = [k.replace("\n", " ") for k in components.keys()]
    weights = [v[0] for v in components.values()]
    colors = [v[2] for v in components.values()]
    
    wedges, texts, autotexts = ax1.pie(
        weights, labels=None, autopct="%1.0f%%", colors=colors,
        startangle=90, textprops={"fontsize": 11, "fontweight": "bold", "color": WHITE},
        wedgeprops={"edgecolor": WHITE, "linewidth": 2},
        pctdistance=0.65
    )
    ax1.legend(labels, loc="lower center", fontsize=8, ncol=2, 
               bbox_to_anchor=(0.5, -0.12))
    ax1.set_title("Poids dans le score final", fontsize=13, fontweight="bold", 
                   pad=15, color=DARK)
    
    # Right: Horizontal bars with values for Doss et al.
    ax2.set_facecolor(WHITE)
    y_labels = list(components.keys())
    y_pos = range(len(y_labels))
    values = [v[1] for v in components.values()]
    bar_colors = [v[2] for v in components.values()]
    weighted = [v[0] * v[1] for v in components.values()]
    
    bars = ax2.barh(y_pos, values, color=bar_colors, edgecolor=WHITE, 
                     linewidth=1.5, height=0.6, zorder=3)
    
    for i, (bar, val, w, wv) in enumerate(zip(bars, values, weights, weighted)):
        ax2.text(val + 0.02, i, f"{val:.2f}  (×{w:.0%} = {wv:.3f})", 
                 va="center", fontsize=10, fontweight="bold", color=DARK)
    
    total = sum(weighted)
    ax2.axvline(x=total, color=RED, linewidth=2, linestyle="--", zorder=2)
    ax2.text(total + 0.02, len(y_labels) - 0.5, 
             f"Total = {total:.3f}\n> Verdict: High",
             fontsize=11, fontweight="bold", color=GREEN)
    
    ax2.set_xlim(0, 1.35)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_labels, fontsize=10)
    ax2.set_xlabel("Score (0-1)", fontsize=11)
    ax2.set_title("Exemple : Doss et al. (2009)", fontsize=13, fontweight="bold",
                   pad=15, color=DARK)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    path = OUT_DIR / "slide8_reliability_formula.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def slide8_verdict_thresholds():
    """Visual scale showing the verdict thresholds."""
    
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    
    # Zones
    zones = [
        (0.0, 0.40, RED, "Not usable", 0.15),
        (0.40, 0.55, "#F59E0B", "Low", 0.10),
        (0.55, 0.75, GOLD, "Moderate", 0.15),
        (0.75, 1.00, GREEN, "High", 0.20),
    ]
    
    for x0, x1, color, label, alpha_val in zones:
        ax.barh(0, x1 - x0, left=x0, height=0.6, color=color, alpha=0.7,
                edgecolor=WHITE, linewidth=2)
        ax.text((x0 + x1) / 2, 0, f"{label}\n[{x0:.2f} – {x1:.2f}]", ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Score de fiabilité totale", fontsize=12, fontweight="bold")
    ax.set_title("Échelle des verdicts par effet", fontsize=13, fontweight="bold", 
                  pad=10, color=DARK)
    
    plt.tight_layout()
    path = OUT_DIR / "slide8_verdict_scale.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# =====================================================================
# SLIDE 9 — FIABILITE ARTICLE (exploration data SCImago / OpenAlex)
# =====================================================================

def slide9_scimago_quartile_distribution():
    """Bar chart of Q1/Q2/Q3/Q4 distribution from SCImago 2024."""
    
    csv_path = Path("data/scimagojr 2024.csv")
    if not csv_path.exists():
        print("  ✗ SCImago 2024 CSV not found, skipping")
        return
    
    quartile_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "-": 0}
    total = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            q = row.get("SJR Best Quartile", "-").strip()
            if q in quartile_counts:
                quartile_counts[q] += 1
            else:
                quartile_counts["-"] += 1
            total += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(f"SCImago 2024 — {total:,} revues dans la base", 
                 fontsize=14, fontweight="bold", color=DARK, y=1.02)
    
    # Left: Quartile distribution pie
    ax1.set_facecolor(WHITE)
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    q_values = [quartile_counts[q] for q in q_labels]
    q_colors = [GREEN, BLUE, GOLD, RED]
    
    wedges, texts, autotexts = ax1.pie(
        q_values, labels=q_labels, autopct=lambda p: f"{p:.0f}%\n({int(p*total/100):,})",
        colors=q_colors, startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops={"edgecolor": WHITE, "linewidth": 2},
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_color(WHITE)
        t.set_fontsize(8)
    ax1.set_title("Distribution des quartiles", fontsize=12, fontweight="bold", 
                   color=DARK, pad=10)
    
    # Highlight: JPSP is Q1
    ax1.annotate("JPSP (Doss et al.) → Q1", xy=(0.15, -0.05), fontsize=9,
                 fontweight="bold", color=GREEN, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN_LIGHT, alpha=0.5))
    
    # Right: SJR distribution histogram
    ax2.set_facecolor(WHITE)
    sjr_values = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                sjr = float(row.get("SJR", "0").replace(",", "."))
                if 0 < sjr < 50:  # filter outliers for readability
                    sjr_values.append(sjr)
            except (ValueError, TypeError):
                pass
    
    ax2.hist(sjr_values, bins=100, color=BLUE, alpha=0.7, edgecolor=BLUE_LIGHT, linewidth=0.3)
    
    # Mark JPSP
    jpsp_sjr = 5.157
    ax2.axvline(x=jpsp_sjr, color=GREEN, linewidth=2, linestyle="--")
    ax2.annotate(f"JPSP\nSJR = {jpsp_sjr}", xy=(jpsp_sjr, ax2.get_ylim()[1] * 0.7),
                 xytext=(jpsp_sjr + 3, ax2.get_ylim()[1] * 0.7),
                 fontsize=10, fontweight="bold", color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN))
    
    # Mark median
    median_sjr = np.median(sjr_values)
    ax2.axvline(x=median_sjr, color=GOLD, linewidth=1.5, linestyle=":")
    ax2.text(median_sjr + 0.3, ax2.get_ylim()[1] * 0.9, f"Médiane = {median_sjr:.2f}",
             fontsize=9, color=GOLD, fontweight="bold")
    
    ax2.set_xlabel("SJR (SCImago Journal Rank)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Nombre de revues", fontsize=11, fontweight="bold")
    ax2.set_title("Distribution du SJR (0-50)", fontsize=12, fontweight="bold",
                   color=DARK, pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    plt.tight_layout()
    path = OUT_DIR / "slide9_scimago_quartiles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def slide9_scimago_evolution():
    """Line chart: evolution of JPSP's SJR over the years from SCImago CSVs."""
    
    data_dir = Path("data")
    years = []
    sjr_vals = []
    quartiles = []
    
    for year in range(1999, 2025):
        csv_path = data_dir / f"scimagojr {year}.csv"
        if not csv_path.exists():
            continue
        
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                title = row.get("Title", "").strip()
                if "Journal of Personality and Social Psychology" in title:
                    try:
                        sjr = float(row.get("SJR", "0").replace(",", "."))
                        q = row.get("SJR Best Quartile", "-").strip()
                        years.append(year)
                        sjr_vals.append(sjr)
                        quartiles.append(q)
                    except (ValueError, TypeError):
                        pass
                    break
    
    if not years:
        print("  ✗ JPSP not found in SCImago CSVs, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    
    # Color markers by quartile
    q_color_map = {"Q1": GREEN, "Q2": BLUE, "Q3": GOLD, "Q4": RED}
    colors = [q_color_map.get(q, GRAY) for q in quartiles]
    
    ax.plot(years, sjr_vals, color=BLUE, linewidth=2, zorder=2, alpha=0.7)
    ax.scatter(years, sjr_vals, c=colors, s=60, zorder=3, edgecolors=DARK, linewidth=0.5)
    
    # Fill area
    ax.fill_between(years, sjr_vals, alpha=0.1, color=BLUE)
    
    # Annotate last value
    if sjr_vals:
        ax.annotate(f"SJR = {sjr_vals[-1]:.2f}\n({quartiles[-1]})", 
                     xy=(years[-1], sjr_vals[-1]),
                     xytext=(years[-1] - 3, sjr_vals[-1] + 1),
                     fontsize=10, fontweight="bold", color=GREEN,
                     arrowprops=dict(arrowstyle="->", color=GREEN))
    
    ax.set_xlabel("Année", fontsize=11, fontweight="bold")
    ax.set_ylabel("SJR", fontsize=11, fontweight="bold")
    ax.set_title("Évolution du SJR — Journal of Personality and Social Psychology (1999-2024)",
                  fontsize=13, fontweight="bold", color=DARK, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(range(1999, 2025, 2))
    ax.tick_params(axis="x", rotation=45)
    
    # Legend for quartiles
    for q, c in [("Q1", GREEN), ("Q2", BLUE)]:
        ax.scatter([], [], c=c, s=50, label=q, edgecolors=DARK, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=10)
    
    plt.tight_layout()
    path = OUT_DIR / "slide9_jpsp_evolution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def slide9_openalex_data_retrieved():
    """Infographic showing what data OpenAlex returns for Doss et al."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Title
    ax.text(5, 9.5, "Données récupérées d'OpenAlex — Doss et al. (2009)", 
            fontsize=15, fontweight="bold", ha="center", color=DARK)
    
    # Box 1: Work (article)
    _draw_box(ax, 0.3, 6.0, 3.0, 3.0, "WORK (Article)", [
        "cited_by_count = 636",
        "type = journal-article",
        "concepts = Attachment...",
        "topics = T10677",
        "authorships = 4 auteurs",
        "open_access = false",
    ], BLUE, BLUE_LIGHT)
    
    # Box 2: Source (journal)
    _draw_box(ax, 3.6, 6.0, 3.0, 3.0, "SOURCE (Journal)", [
        "JPSP",
        "2yr_citedness = 2.15",
        "publisher = APA",
        "issn = 0022-3514",
        "type = journal",
        "works_count = 15,827",
    ], GREEN, GREEN_LIGHT)
    
    # Box 3: Authors (2 selected)
    _draw_box(ax, 6.9, 6.0, 3.0, 3.0, "AUTHORS (1er + dernier)", [
        "Doss: h=35, cited=4823",
        "Markman: h=74, cited=17365",
        "works: 158 / 246",
        "affiliations: 2 univ.",
        "→ max(h) = 74",
        "→ avg_works = 202",
    ], "#8B5CF6", "#DDD6FE")
    
    # Box 4: SCImago (offline)
    _draw_box(ax, 0.3, 2.5, 3.0, 3.0, "SCIMAGO (CSV local)", [
        "Quartile = Q1",
        "SJR = 5.157",
        "H-index revue = 378",
        "Année = 2009",
        "26 ans de données",
        "(1999-2024)",
    ], GOLD, GOLD_LIGHT)
    
    # Box 5: Crossref
    _draw_box(ax, 3.6, 2.5, 3.0, 3.0, "CROSSREF", [
        "DOI validé ✓",
        "10.1037/a0013969",
        "Matching score > 0.40",
        "Titre vérifié",
        "",
        "",
    ], GRAY, "#E5E7EB")
    
    # Box 6: Score final
    _draw_box(ax, 6.9, 2.5, 3.0, 3.0, "SCORE RENOMMEE", [
        "Article = 1.00 (35%)",
        "Revue = 0.73 (20%)",
        "Auteur = 1.00 (15%)",
        "Champ = 1.00 (20%)",
        "Réseau = 0.63 (10%)",
        "GLOBAL = 0.91",
    ], "#059669", "#A7F3D0")
    
    # Arrows
    _draw_arrow(ax, 1.8, 6.0, 1.8, 5.8)
    _draw_arrow(ax, 5.1, 6.0, 5.1, 5.8)
    _draw_arrow(ax, 8.4, 6.0, 8.4, 5.8)
    
    plt.tight_layout()
    path = OUT_DIR / "slide9_openalex_data_map.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def _draw_box(ax, x, y, w, h, title, lines, border_color, bg_color):
    """Draw a rounded box with title and text lines."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=bg_color, edgecolor=border_color, linewidth=2, alpha=0.85
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=border_color)
    for i, line in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.7 - i * 0.38, line, ha="left", va="top",
                fontsize=8, color=DARK, family="monospace")


def _draw_arrow(ax, x1, y1, x2, y2):
    """Draw a simple arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=1.5))


def slide9_article_score_radar():
    """Radar chart of the 5 scoring dimensions for Doss et al."""
    
    categories = ["Article\n(citations)", "Revue\n(SCImago)", "Auteur\n(h-index)", 
                   "Champ\n(normalisation)", "Réseau\n(institutions)"]
    values = [1.00, 0.73, 1.00, 1.00, 0.625]
    weights = [0.35, 0.20, 0.15, 0.20, 0.10]
    
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(WHITE)
    
    # Fill
    ax.fill(angles, values_plot, color=BLUE, alpha=0.15)
    ax.plot(angles, values_plot, color=BLUE, linewidth=2.5, marker="o", markersize=8)
    
    # Labels with scores
    ax.set_xticks(angles[:-1])
    labels_with_scores = [f"{cat}\n{val:.2f} ({w:.0%})" for cat, val, w in zip(categories, values, weights)]
    ax.set_xticklabels(labels_with_scores, fontsize=10, fontweight="bold", color=DARK)
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=GRAY)
    ax.yaxis.grid(True, color=GRAY, alpha=0.3)
    ax.xaxis.grid(True, color=GRAY, alpha=0.3)
    
    # Center score
    ax.text(0, 0, "0.91", fontsize=24, fontweight="bold", ha="center", va="center",
            color=GREEN, bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=GREEN, linewidth=2))
    
    ax.set_title("Score de renommée — Doss et al. (2009)", fontsize=14, fontweight="bold",
                  color=DARK, pad=30)
    
    plt.tight_layout()
    path = OUT_DIR / "slide9_radar_scores.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Génération des graphiques pour la présentation")
    print("=" * 60)
    
    print("\n📊 SLIDE 8 — Fiabilité Effect")
    slide8_calc_score_breakdown()
    slide8_reliability_formula()
    slide8_verdict_thresholds()
    
    print("\n📊 SLIDE 9 — Fiabilité Article (data exploration)")
    slide9_scimago_quartile_distribution()
    slide9_scimago_evolution()
    slide9_openalex_data_retrieved()
    slide9_article_score_radar()
    
    print(f"\n✅ Tous les graphiques sont dans : {OUT_DIR.resolve()}")
    print("   → Copie-les dans ta présentation !")
