"""Generate charts for presentation slides:
- OpenAlex: diagram of what fields are available vs what we use
- SCImago: stats analysis (journals per year, fields available, distribution)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import re

matplotlib.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"  
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "slides_charts"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================================
# 1. SCImago stats
# =====================================================================

def load_all_scimago():
    """Load all SCImago CSVs and return a combined DataFrame."""
    all_frames = []
    for f in sorted(DATA_DIR.glob("scimagojr *.csv")):
        m = re.search(r"scimagojr\s+(\d{4})\.csv$", f.name)
        if not m:
            continue
        year = int(m.group(1))
        df = pd.read_csv(f, sep=";", dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["_year"] = year
        all_frames.append(df)
    if not all_frames:
        raise RuntimeError("No SCImago CSVs found")
    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def scimago_journals_per_year(df):
    """Bar chart: number of journals per year."""
    counts = df.groupby("_year").size().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#F4B400" if y < 2024 else "#0F9D58" for y in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    
    # Add value labels on top
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}".replace(",", " "), ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel("Nombre de revues", fontsize=12)
    ax.set_title("SCImago : nombre de revues indexees par annee (1999-2024)", fontsize=14, fontweight="bold")
    ax.set_xticks(counts.index)
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    total = counts.sum()
    ax.annotate(f"Total : {total:,} entrees sur {len(counts)} annees".replace(",", " "),
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#0F9D58"))
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scimago_journals_per_year.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> scimago_journals_per_year.png ({len(counts)} years, total={total})")
    return counts


def scimago_columns_table(df):
    """Print the columns available in SCImago CSVs."""
    # Use 2024 as reference
    df_2024 = df[df["_year"] == 2024]
    cols = [c for c in df_2024.columns if not c.startswith("_")]
    
    print("\n  SCImago 2024 - Colonnes disponibles :")
    for c in cols:
        non_null = df_2024[c].notna().sum()
        total = len(df_2024)
        pct = non_null / total * 100 if total > 0 else 0
        print(f"    {c:40s}  {non_null:>6} / {total}  ({pct:.0f}%)")
    return cols


def scimago_quartile_distribution(df):
    """Pie chart: Q1/Q2/Q3/Q4 distribution for latest year."""
    df_2024 = df[df["_year"] == 2024].copy()
    
    # Find quartile column
    q_col = None
    for col in ("SJR Best Quartile", "Quartile"):
        if col in df_2024.columns:
            q_col = col
            break
    if not q_col:
        print("  [WARN] No quartile column found")
        return
    
    df_2024["q_clean"] = df_2024[q_col].fillna("").str.strip().str.upper()
    dist = df_2024["q_clean"].value_counts()
    
    labels = ["Q1", "Q2", "Q3", "Q4"]
    values = [dist.get(q, 0) for q in labels]
    other = len(df_2024) - sum(values)
    if other > 0:
        labels.append("Non classe")
        values.append(other)
    
    colors_q = ["#0F9D58", "#F4B400", "#FF6D00", "#DB4437", "#BDBDBD"]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(p*sum(values)/100)})",
        colors=colors_q[:len(labels)], startangle=90,
        textprops={"fontsize": 10}
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("SCImago 2024 : distribution des quartiles", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scimago_quartile_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> scimago_quartile_dist.png (Q1={values[0]}, Q2={values[1]}, Q3={values[2]}, Q4={values[3]})")


def scimago_sjr_histogram(df):
    """Histogram of SJR values for 2024."""
    df_2024 = df[df["_year"] == 2024].copy()
    
    sjr_col = None
    for col in ("SJR", "sjr"):
        if col in df_2024.columns:
            sjr_col = col
            break
    if not sjr_col:
        return
    
    df_2024["sjr_float"] = pd.to_numeric(
        df_2024[sjr_col].str.replace(",", ".", regex=False), errors="coerce"
    )
    sjr_vals = df_2024["sjr_float"].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Cap at 10 for visibility
    sjr_capped = sjr_vals.clip(upper=10)
    ax.hist(sjr_capped, bins=50, color="#4285F4", edgecolor="white", alpha=0.85)
    ax.axvline(sjr_vals.median(), color="#DB4437", linestyle="--", linewidth=2, label=f"Mediane = {sjr_vals.median():.2f}")
    ax.axvline(sjr_vals.mean(), color="#F4B400", linestyle="--", linewidth=2, label=f"Moyenne = {sjr_vals.mean():.2f}")
    
    ax.set_xlabel("SJR (tronque a 10)", fontsize=11)
    ax.set_ylabel("Nombre de revues", fontsize=11)
    ax.set_title("SCImago 2024 : distribution du SJR", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    stats_text = (f"N = {len(sjr_vals):,}\n"
                  f"Min = {sjr_vals.min():.3f}\n"
                  f"Max = {sjr_vals.max():.1f}\n"
                  f"P75 = {sjr_vals.quantile(0.75):.2f}\n"
                  f"P95 = {sjr_vals.quantile(0.95):.2f}").replace(",", " ")
    ax.annotate(stats_text, xy=(0.75, 0.70), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F4B400"))
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scimago_sjr_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> scimago_sjr_distribution.png (N={len(sjr_vals)}, median={sjr_vals.median():.2f})")


def scimago_top_areas(df):
    """Bar chart: top areas by journal count (2024)."""
    df_2024 = df[df["_year"] == 2024].copy()
    
    areas_col = None
    for col in ("Areas", "areas"):
        if col in df_2024.columns:
            areas_col = col
            break
    if not areas_col:
        return
    
    # Each row can have multiple areas separated by ;
    all_areas = []
    for val in df_2024[areas_col].dropna():
        for a in str(val).split(";"):
            a = a.strip()
            if a:
                all_areas.append(a)
    
    area_counts = pd.Series(all_areas).value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(area_counts)), area_counts.values, color="#4285F4", edgecolor="white")
    ax.set_yticks(range(len(area_counts)))
    ax.set_yticklabels(area_counts.index, fontsize=9)
    ax.invert_yaxis()
    
    for bar, val in zip(bars, area_counts.values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:,}".replace(",", " "),
                va="center", fontsize=8, fontweight="bold")
    
    ax.set_xlabel("Nombre de revues", fontsize=11)
    ax.set_title("SCImago 2024 : top 15 domaines scientifiques", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scimago_top_areas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> scimago_top_areas.png ({len(area_counts)} areas)")


# =====================================================================
# 2. OpenAlex data map
# =====================================================================

def openalex_data_map():
    """Generate a visual table showing OpenAlex endpoints and what we extract."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    
    # Title
    ax.text(8, 10.5, "OpenAlex API : donnees disponibles et exploitees", fontsize=16,
            fontweight="bold", ha="center", va="center", color="#1A237E")
    
    # 3 endpoint blocks
    blocks = [
        {
            "title": "WORKS  /works/{doi}",
            "subtitle": "Entite article",
            "x": 0.3, "y": 1.0, "w": 4.8, "h": 8.5,
            "color": "#E3F2FD", "border": "#1565C0",
            "fields": [
                ("display_name", "Titre de l'article", True),
                ("publication_year", "Annee de publication", True),
                ("cited_by_count", "Nb citations", True),
                ("doi", "DOI", True),
                ("type", "Type (article, book...)", False),
                ("authorships[]", "Liste auteurs + affiliations", True),
                ("authorships[].institutions", "Institutions par auteur", True),
                ("primary_location.source", "Revue (ID source)", True),
                ("concepts[] / topics[]", "Concepts / Sujets", True),
                ("open_access.is_oa", "Acces ouvert ?", False),
                ("biblio.volume/issue", "Volume, numero, pages", False),
                ("abstract_inverted_index", "Resume (index inverse)", False),
                ("referenced_works[]", "Articles cites", False),
                ("related_works[]", "Articles similaires", False),
                ("grants[]", "Financements", False),
                ("sustainable_dev_goals", "ODD / SDG", False),
            ]
        },
        {
            "title": "SOURCES  /sources/{id}",
            "subtitle": "Entite revue/journal",
            "x": 5.5, "y": 3.5, "w": 4.8, "h": 6.0,
            "color": "#FFF3E0", "border": "#E65100",
            "fields": [
                ("display_name", "Nom de la revue", True),
                ("issn / issn_l", "ISSN (print + online)", True),
                ("publisher", "Editeur", False),
                ("type", "Type (journal, repo...)", False),
                ("summary_stats.2yr_mean_citedness", "Citedness moyen 2 ans", True),
                ("summary_stats.h_index", "h-index du journal", False),
                ("works_count", "Nb articles publies", False),
                ("cited_by_count", "Citations totales", False),
                ("is_oa / is_in_doaj", "Open Access ?", False),
                ("apc_usd", "Cout publication APC", False),
            ]
        },
        {
            "title": "AUTHORS  /authors/{id}",
            "subtitle": "Entite auteur",
            "x": 10.8, "y": 3.5, "w": 4.8, "h": 6.0,
            "color": "#E8F5E9", "border": "#2E7D32",
            "fields": [
                ("display_name", "Nom complet", True),
                ("summary_stats.h_index", "h-index", True),
                ("cited_by_count", "Citations totales", True),
                ("works_count", "Nb publications", True),
                ("last_known_institutions", "Institutions actuelles", False),
                ("affiliations[]", "Historique affiliations", False),
                ("topics[]", "Sujets de recherche", False),
                ("counts_by_year[]", "Citations/an", False),
            ]
        }
    ]
    
    for block in blocks:
        # Background rect
        rect = plt.Rectangle((block["x"], block["y"]), block["w"], block["h"],
                             facecolor=block["color"], edgecolor=block["border"],
                             linewidth=2, zorder=1, alpha=0.7)
        ax.add_patch(rect)
        
        # Title
        ax.text(block["x"] + block["w"]/2, block["y"] + block["h"] - 0.3,
                block["title"], fontsize=11, fontweight="bold", ha="center",
                color=block["border"])
        ax.text(block["x"] + block["w"]/2, block["y"] + block["h"] - 0.7,
                block["subtitle"], fontsize=9, ha="center", color="#666666", style="italic")
        
        # Fields
        for i, (field_name, desc, used) in enumerate(block["fields"]):
            y_pos = block["y"] + block["h"] - 1.2 - i * 0.45
            if y_pos < block["y"] + 0.1:
                break
            
            marker = "[U]" if used else "[ ]"
            color = block["border"] if used else "#999999"
            weight = "bold" if used else "normal"
            
            ax.text(block["x"] + 0.2, y_pos, marker, fontsize=7, color=color, fontweight=weight,
                    fontfamily="monospace", va="center")
            ax.text(block["x"] + 0.7, y_pos, field_name, fontsize=7.5, color=color,
                    fontweight=weight, fontfamily="monospace", va="center")
            ax.text(block["x"] + 0.7, y_pos - 0.18, desc, fontsize=6.5, color="#888888",
                    va="center")
    
    # Legend
    ax.text(5.5, 0.5, "[U] = Utilise par Impact", fontsize=10, fontweight="bold", color="#1565C0")
    ax.text(5.5, 0.15, "[ ] = Disponible mais non exploite (amelioration possible)", fontsize=9, color="#999999")
    
    # API info
    ax.text(0.3, 0.5, "API gratuite\nPolite pool (email)\nPas de cle requise", fontsize=8,
            color="#2E7D32", bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32"))
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "openalex_data_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> openalex_data_map.png")


def scimago_fields_summary():
    """Generate a visual summary of SCImago CSV fields and what we use."""
    
    fields = [
        ("Rank", "Rang du journal (par SJR)", False, "int"),
        ("Sourceid", "Identifiant unique SCImago", False, "int"),
        ("Title", "Nom du journal", True, "str"),
        ("Type", "journal / book series / conf.", False, "str"),
        ("Issn", "ISSN (print + online)", True, "str"),
        ("Publisher", "Editeur", True, "str"),
        ("Open Access", "Yes / No", True, "str"),
        ("SJR", "SCImago Journal Rank (score)", True, "float"),
        ("SJR Best Quartile", "Q1 / Q2 / Q3 / Q4", True, "str"),
        ("H index", "h-index du journal", True, "int"),
        ("Total Docs. (year)", "Articles publies cette annee", False, "int"),
        ("Total Docs. (3years)", "Articles sur 3 ans", False, "int"),
        ("Total Refs.", "References totales", False, "int"),
        ("Total Citations (3years)", "Citations recues sur 3 ans", False, "int"),
        ("Citable Docs. (3years)", "Docs citables sur 3 ans", False, "int"),
        ("Citations/Doc. (2years)", "Impact moyen 2 ans", False, "float"),
        ("Ref./Doc.", "References par article", False, "float"),
        ("%Female", "% auteures femmes", False, "float"),
        ("Overton", "Score Overton (policy impact)", False, "int"),
        ("SDG", "Objectifs Developpement Durable", False, "str"),
        ("Country", "Pays d'edition", True, "str"),
        ("Region", "Region geographique", False, "str"),
        ("Coverage", "Annees couvertes", False, "str"),
        ("Categories", "Sous-domaines (ex: Oncology Q1)", True, "str"),
        ("Areas", "Domaines (ex: Medicine)", True, "str"),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, len(fields) + 2)
    
    ax.text(7, len(fields) + 1.3, "SCImago CSV : 25 champs disponibles par revue et par annee",
            fontsize=14, fontweight="bold", ha="center", color="#E65100")
    
    # Header
    y_header = len(fields) + 0.5
    ax.text(0.2, y_header, "Champ", fontsize=10, fontweight="bold", color="#333")
    ax.text(4.5, y_header, "Description", fontsize=10, fontweight="bold", color="#333")
    ax.text(10.0, y_header, "Type", fontsize=10, fontweight="bold", color="#333")
    ax.text(11.5, y_header, "Utilise ?", fontsize=10, fontweight="bold", color="#333")
    ax.axhline(y=y_header - 0.2, xmin=0.01, xmax=0.95, color="#333", linewidth=1)
    
    for i, (name, desc, used, dtype) in enumerate(fields):
        y = len(fields) - i - 0.3
        bg_color = "#FFF3E0" if used else "white"
        rect = plt.Rectangle((0.1, y - 0.2), 13.5, 0.45, facecolor=bg_color,
                             edgecolor="#EEEEEE", linewidth=0.5, zorder=0)
        ax.add_patch(rect)
        
        color = "#E65100" if used else "#666666"
        weight = "bold" if used else "normal"
        
        ax.text(0.2, y, name, fontsize=8.5, color=color, fontweight=weight, fontfamily="monospace", va="center")
        ax.text(4.5, y, desc, fontsize=8.5, color="#444444", va="center")
        ax.text(10.0, y, dtype, fontsize=8, color="#999999", fontfamily="monospace", va="center")
        marker = "OUI" if used else "---"
        marker_color = "#0F9D58" if used else "#CCCCCC"
        ax.text(11.8, y, marker, fontsize=8.5, color=marker_color, fontweight="bold", va="center")
    
    # Summary
    used_count = sum(1 for _, _, u, _ in fields if u)
    ax.text(7, -0.5, f"{used_count} champs exploites sur {len(fields)} disponibles  |  26 annees (1999-2024)  |  ~{30000}+ revues/an",
            fontsize=10, ha="center", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scimago_fields_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> scimago_fields_summary.png ({used_count}/{len(fields)} fields used)")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Generation des graphiques pour slides")
    print("=" * 60)
    
    # --- SCImago ---
    print("\n[1] Chargement SCImago...")
    df = load_all_scimago()
    print(f"    {len(df)} lignes, {df['_year'].nunique()} annees ({df['_year'].min()}-{df['_year'].max()})")
    
    print("\n[2] Journals par annee...")
    counts = scimago_journals_per_year(df)
    
    print("\n[3] Colonnes SCImago 2024...")
    cols = scimago_columns_table(df)
    
    print("\n[4] Distribution quartiles...")
    scimago_quartile_distribution(df)
    
    print("\n[5] Distribution SJR...")
    scimago_sjr_histogram(df)
    
    print("\n[6] Top areas...")
    scimago_top_areas(df)
    
    print("\n[7] Resume champs SCImago...")
    scimago_fields_summary()
    
    # --- OpenAlex ---
    print("\n[8] Data map OpenAlex...")
    openalex_data_map()
    
    print("\n" + "=" * 60)
    print(f"  Tous les graphiques dans : {OUT_DIR}")
    print("=" * 60)
