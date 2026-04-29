from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

try:
    import pyreadstat
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install with: pip install pandas pyreadstat openpyxl"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "原始数据+问卷2011~2020" / "2020" / "CHARLS2020r" / "Work_Retirement.dta"
OUTPUT_DIR = ROOT / "outputs"
DICT_PATH = OUTPUT_DIR / "2020_work_retirement_variable_dictionary.xlsx"
CANDIDATES_PATH = OUTPUT_DIR / "2020_labor_variable_candidates.xlsx"

KEYWORDS = [
    "work",
    "working",
    "job",
    "labor",
    "labour",
    "employ",
    "employment",
    "retire",
    "retired",
    "farm",
    "farming",
    "agriculture",
    "wage",
    "salary",
    "self",
    "business",
    "hours",
    "month",
    "income",
    "pension",
    "工作",
    "劳动",
    "就业",
    "退休",
    "农业",
    "务农",
    "受雇",
    "自雇",
    "经营",
    "工时",
    "收入",
    "养老金",
]

ID_HINTS = [
    "id",
    "householdid",
    "household_id",
    "hhid",
    "communityid",
    "community_id",
    "commid",
]


def summarize_value_labels(value_label_name: str, value_labels: dict[str, dict]) -> str:
    if not value_label_name:
        return ""
    mapping = value_labels.get(value_label_name, {})
    if not mapping:
        return value_label_name

    pairs = []
    for key, value in list(mapping.items())[:8]:
        pairs.append(f"{key}={value}")

    summary = "; ".join(pairs)
    if len(mapping) > 8:
        summary += f"; ... ({len(mapping)} values)"
    return summary


def has_keyword(text: str) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in KEYWORDS)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, meta = pyreadstat.read_dta(DATA_PATH, apply_value_formats=False)

    print(f"Data file: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("First 10 variables:")
    for name in df.columns[:10]:
        print(f"  - {name}")

    normalized_columns = {col.casefold(): col for col in df.columns}
    print("Identifier variables found:")
    found_any_id = False
    for hint in ID_HINTS:
        if hint in normalized_columns:
            print(f"  - {normalized_columns[hint]}")
            found_any_id = True
    if not found_any_id:
        print("  - None matched the common ID hints")

    variable_to_label = meta.column_names_to_labels or {}
    variable_to_value_label = getattr(meta, "variable_to_label", {}) or {}
    value_labels = meta.value_labels or {}

    records = []
    for variable in df.columns:
        variable_label = variable_to_label.get(variable, "") or ""
        value_label_name = variable_to_value_label.get(variable, "") or ""
        value_label_summary = summarize_value_labels(value_label_name, value_labels)
        records.append(
            {
                "variable_name": variable,
                "variable_label": variable_label,
                "value_label_name": value_label_name,
                "value_label_summary": value_label_summary,
            }
        )

    dictionary_df = pd.DataFrame(records)
    dictionary_df.to_excel(DICT_PATH, index=False)

    candidate_mask = dictionary_df.apply(
        lambda row: has_keyword(
            " ".join(
                [
                    str(row["variable_name"]),
                    str(row["variable_label"]),
                    str(row["value_label_name"]),
                    str(row["value_label_summary"]),
                ]
            )
        ),
        axis=1,
    )
    candidates_df = dictionary_df.loc[candidate_mask].copy()
    candidates_df.to_excel(CANDIDATES_PATH, index=False)

    print()
    print("Candidate variables:")
    if candidates_df.empty:
        print("No candidates matched the keyword list.")
    else:
        for row in candidates_df.itertuples(index=False):
            print(
                f"{row.variable_name} | {row.variable_label} | {row.value_label_summary}"
            )

    print()
    print(f"Variable dictionary saved to: {DICT_PATH}")
    print(f"Candidate list saved to: {CANDIDATES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
