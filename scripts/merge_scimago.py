"""Merge all yearly ``scimagojr YYYY.csv`` files into one
``data/scimago_merged.csv`` with an added ``Year`` column.

Usage::

    python scripts/merge_scimago.py
"""
from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_FILE = DATA_DIR / "scimago_merged.csv"


def main() -> None:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(DATA_DIR.glob("scimagojr *.csv")):
        m = re.search(r"scimagojr\s+(\d{4})\.csv$", csv_path.name, re.IGNORECASE)
        if not m:
            continue
        year = int(m.group(1))
        print(f"  Loading {csv_path.name}  ({year}) …", end=" ")
        df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Rename year-specific column  Total Docs. (YYYY)  → Total Docs.
        year_col = f"Total Docs. ({year})"
        if year_col in df.columns:
            df.rename(columns={year_col: "Total Docs."}, inplace=True)

        df.insert(0, "Year", str(year))
        print(f"{len(df)} rows")
        frames.append(df)

    if not frames:
        print("No scimagojr YYYY.csv found in", DATA_DIR)
        return

    merged = pd.concat(frames, ignore_index=True)

    # Ensure consistent column order
    cols_first = ["Year", "Rank", "Sourceid", "Title", "Type", "Issn"]
    rest = [c for c in merged.columns if c not in cols_first]
    merged = merged[cols_first + rest]

    merged.to_csv(OUT_FILE, index=False, sep=";", encoding="utf-8")
    print(f"\n✓ Wrote {OUT_FILE}  ({len(merged)} rows, {len(merged.columns)} cols)")


if __name__ == "__main__":
    main()
