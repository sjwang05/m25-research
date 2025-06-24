import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, List, Dict, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Mapping and column indices
# ---------------------------------------------------------------------------
MEAL_MAP = {"朝": "breakfast", "昼": "lunch", "夕": "dinner"}

COL_TIME = 1       # "朝" / "昼" / "夕" or blank
COL_DISH = 3       # Dish name (料理名)
COL_SUBTOTAL = 7   # Label like "朝小計"
COL_ENERGY = 10    # kcal
COL_PROTEIN = 11   # g
COL_FAT = 14       # g
COL_CARB = 15      # g

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Canonicalise *text* for matching & tidy output.

    • half‑width kana → full‑width
    • full‑width roman/ASCII variants → ASCII
    • any "space separator" char → ASCII space
    • collapse multiple whitespace → one space
    """
    if not isinstance(text, str):
        text = str(text or "")

    s = unicodedata.normalize("NFKC", text)  # width & punctuation normalisation
    s = "".join(" " if unicodedata.category(c) == "Zs" else c for c in s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Numeric conversion helper
# ---------------------------------------------------------------------------

def _to_float(cell: str | Any):
    try:
        return float(str(cell).replace(",", "")) if str(cell).strip() else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core parsing routine
# ---------------------------------------------------------------------------

def parse_kyoto(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, header=None, dtype=str).fillna("")

    records: List[Dict[str, Any]] = []
    current_meal_en: str | None = None
    current_meal_ja: str | None = None
    items: List[str] = []
    energy_values: List[float | None] = []
    protein_values: List[float | None] = []
    fat_values: List[float | None] = []
    carb_values: List[float | None] = []

    for _, row in df.iterrows():
        time_marker = normalize_text(row[COL_TIME])
        dish_name = normalize_text(row[COL_DISH])
        subtotal_label = normalize_text(row[COL_SUBTOTAL])

        # --- start of a new meal ------------------------------------------
        if current_meal_en is None:
            if time_marker in MEAL_MAP and dish_name:
                current_meal_en = MEAL_MAP[time_marker]
                current_meal_ja = time_marker
                items = [dish_name]
                # Capture nutritional values for the first item
                energy_values = [_to_float(row[COL_ENERGY])]
                protein_values = [_to_float(row[COL_PROTEIN])]
                fat_values = [_to_float(row[COL_FAT])]
                carb_values = [_to_float(row[COL_CARB])]
            continue

        # --- inside a meal -------------------------------------------------
        if (subtotal_label.endswith("小計") and subtotal_label.startswith(current_meal_ja)):
            # Check if we have any valid nutritional values
            has_values = any(
                v is not None for values in [energy_values, protein_values, fat_values, carb_values] 
                for v in values
            )
            
            if has_values and items:  # Only add if we have both items and values
                records.append({
                    "meal_time": current_meal_en,
                    "meal_time_local": current_meal_ja,
                    "description_local": items.copy(),
                    "energy": energy_values.copy(),
                    "protein": protein_values.copy(),
                    "fat": fat_values.copy(),
                    "carb": carb_values.copy(),
                })
            
            # Reset for next meal
            current_meal_en = current_meal_ja = None
            items = []
            energy_values = []
            protein_values = []
            fat_values = []
            carb_values = []
            continue

        # --- add food item and its nutritional values ----------------------
        if dish_name and current_meal_en is not None:
            # Get nutritional values for this item
            energy = _to_float(row[COL_ENERGY])
            protein = _to_float(row[COL_PROTEIN])
            fat = _to_float(row[COL_FAT])
            carb = _to_float(row[COL_CARB])
            
            # Only add if at least one nutritional value is present
            if any(v is not None for v in [energy, protein, fat, carb]):
                items.append(dish_name)
                energy_values.append(energy)
                protein_values.append(protein)
                fat_values.append(fat)
                carb_values.append(carb)

    return records


# ---------------------------------------------------------------------------
# NutriBench DataFrame builder
# ---------------------------------------------------------------------------

def build_nutribench_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    # Convert arrays to JSON strings
    for record in records:
        record["description_local"] = json.dumps(record["description_local"], ensure_ascii=False)
        record["carb"] = json.dumps(record["carb"], ensure_ascii=False)
        record["energy"] = json.dumps(record["energy"], ensure_ascii=False)
        record["protein"] = json.dumps(record["protein"], ensure_ascii=False)
        record["fat"] = json.dumps(record["fat"], ensure_ascii=False)
    
    df = pd.DataFrame(records)

    # Add empty arrays for fields not in the original data
    df["description"] = "[]"
    df["unit"] = "[]"
    df["unit_local"] = "[]"
    df["local_language"] = "Japanese"

    order = [
        "description", "description_local", "unit", "unit_local",
        "local_language", "carb", "energy", "protein", "fat", "meal_time", "meal_time_local"
    ]
    return df[order]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert NHO-Kyoto CSV to NutriBench Multilingual format")
    parser.add_argument("input_csv", type=Path, help="Path to NHO-Kyoto.csv")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output file; default is *.csv or *.jsonl based on format")
    parser.add_argument("--jsonl", action="store_true",
                        help="Write JSON Lines instead of CSV")
    args = parser.parse_args()

    # Choose default output name if none supplied
    ext = "jsonl" if args.jsonl else "csv"
    base_name = str(args.input_csv).replace('.csv', '')
    if args.output is None:
        args.output = Path(f"{base_name}-NutriBench.{ext}")

    records = parse_kyoto(args.input_csv)
    if not records:
        raise SystemExit("No meals detected.")

    df = build_nutribench_df(records)

    if args.jsonl:
        df.to_json(args.output, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(args.output, index=False)

    fmt = "JSONL" if args.jsonl else "CSV"
    print(f"✅ Converted {len(df)} meals → {args.output} ({fmt})")


if __name__ == "__main__":
    main()
