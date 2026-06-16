"""Load and normalize the medicine CSV into a list of clean records.

The Kaggle medicine dataset has inconsistent column names across versions, so
we map a bunch of likely names onto our canonical schema and skip anything
missing. Each record gets a `doc_text` field that is what we embed.
"""
import os
import re
import pandas as pd

from . import config

# Map many possible source column names -> our canonical field.
_COLUMN_ALIASES = {
    config.COL_NAME: ["name", "medicine name", "drug name", "product_name"],
    config.COL_USES: ["uses", "use", "indication", "indications", "therapeutic_class"],
    config.COL_SIDE_EFFECTS: ["side_effects", "side effects", "sideeffect", "adverse_effects"],
    config.COL_SUBSTITUTES: ["substitutes", "substitute", "alternatives", "substitute0"],
    config.COL_COMPOSITION: ["composition", "salt_composition", "ingredients", "short_composition1"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _split_list(value: str) -> list[str]:
    """Split a 'substitutes' / 'side effects' cell into individual items."""
    if not value:
        return []
    parts = re.split(r"[|,;]", value)
    return [p.strip() for p in parts if p.strip()]


def _gather_numbered(df: pd.DataFrame, prefix: str) -> list[str]:
    """Find spread-out columns like sideEffect0..sideEffect41 -> ordered list.

    The Kaggle '250k Medicines' dataset spreads multi-valued fields across
    numbered columns (substitute0..4, sideEffect0..41, use0..4). This collects
    every column whose name (lowercased) starts with `prefix` and ends in digits.
    """
    matches = []
    for c in df.columns:
        cl = c.lower().strip()
        if cl.startswith(prefix.lower()) and cl[len(prefix):].isdigit():
            matches.append((int(cl[len(prefix):]), c))
    matches.sort()
    return [c for _, c in matches]


def _collect_cells(row, columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        v = _clean(row[c])
        if v and v.upper() != "NA":
            out.append(v)
    return out


def load_records(csv_path: str | None = None, max_rows: int | None = None) -> list[dict]:
    csv_path = csv_path or config.DATA_CSV
    max_rows = max_rows or config.MAX_ROWS

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'. Download the dataset (see README) "
            f"or set DATA_CSV in your .env."
        )

    # Read the WHOLE file, then sample evenly across it — otherwise nrows=N
    # only gives the first N rows, which in this dataset are all alphabetical
    # ('a*' medicines) and unrepresentative.
    df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8", low_memory=False)

    if len(df) > max_rows:
        # Evenly spaced sample across the full dataset (deterministic, no RNG
        # needed) so we get a*, b*, ... z* medicines, not just a*.
        step = len(df) / max_rows
        keep_idx = [int(i * step) for i in range(max_rows)]
        df = df.iloc[keep_idx].reset_index(drop=True)

    # Resolve single-column matches (works for the simple sample CSV).
    resolved = {
        canonical: _find_column(df, aliases)
        for canonical, aliases in _COLUMN_ALIASES.items()
    }
    if resolved[config.COL_NAME] is None:
        raise ValueError(
            f"Could not find a medicine-name column in {list(df.columns)}. "
            f"Edit _COLUMN_ALIASES in data_loader.py to match your file."
        )

    # Detect the Kaggle "numbered columns" layout (substitute0, sideEffect0, use0).
    sub_cols = _gather_numbered(df, "substitute")
    side_cols = _gather_numbered(df, "sideeffect")
    use_cols = _gather_numbered(df, "use")
    # Composition lives in 'Chemical Class' / 'Therapeutic Class' in this dataset.
    chem_col = _find_column(df, ["chemical class", "composition", "salt_composition"])
    ther_col = _find_column(df, ["therapeutic class"])

    records = []
    for idx, row in df.iterrows():
        name = _clean(row[resolved[config.COL_NAME]])
        if not name:
            continue

        # Prefer numbered columns (real dataset); fall back to single column (sample).
        if use_cols:
            uses = ", ".join(_collect_cells(row, use_cols))
        else:
            uses = _clean(row[resolved[config.COL_USES]]) if resolved[config.COL_USES] else ""

        if side_cols:
            side_list = _collect_cells(row, side_cols)
            side = ", ".join(side_list)
        else:
            side = _clean(row[resolved[config.COL_SIDE_EFFECTS]]) if resolved[config.COL_SIDE_EFFECTS] else ""

        if sub_cols:
            subs = ", ".join(_collect_cells(row, sub_cols))
        else:
            subs = _clean(row[resolved[config.COL_SUBSTITUTES]]) if resolved[config.COL_SUBSTITUTES] else ""

        if chem_col or ther_col:
            comp_parts = _collect_cells(row, [c for c in (chem_col, ther_col) if c])
            comp = " | ".join(comp_parts)
        else:
            comp = _clean(row[resolved[config.COL_COMPOSITION]]) if resolved[config.COL_COMPOSITION] else ""

        # Text we embed for semantic search. We REPEAT the indication/uses so the
        # vector leans toward *what the medicine treats* (e.g. "fever") rather
        # than being dominated by the brand name. This makes condition/symptom
        # queries match far better.
        doc_text = (
            f"{name}. "
            f"This medicine is used for: {uses}. "
            f"It treats {uses}. "
            f"Indication: {uses}. "
            f"Drug class: {comp}. "
            f"Side effects: {side}."
        )

        records.append({
            "id": int(idx),
            "name": name,
            "uses": uses,
            "side_effects": _split_list(side),
            "substitutes": _split_list(subs),
            "composition": comp,
            "doc_text": doc_text,
        })

    if not records:
        raise ValueError("No valid records loaded from the CSV.")

    # Drop duplicate medicines (the dataset repeats some rows verbatim), keeping
    # the first occurrence so retrieval slots aren't wasted on copies.
    seen = set()
    deduped = []
    for r in records:
        key = r["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


if __name__ == "__main__":
    recs = load_records()
    print(f"Loaded {len(recs)} medicine records.")
    print("Sample:", recs[0])
