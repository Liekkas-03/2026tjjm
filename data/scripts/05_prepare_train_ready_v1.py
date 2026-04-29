from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and (path / "labels.csv").exists())
OUTPUT_DIR = ROOT / "outputs"

SOURCE_MODEL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v2_model_age90.csv"
TRAIN_READY = BASE_DIR / "CHARLS_labor_panel_2018_2020_v3_train_ready.csv"
FEATURES_XLSX = OUTPUT_DIR / "train_ready_feature_roles_v1.xlsx"
REPORT_XLSX = OUTPUT_DIR / "train_ready_report_v1.xlsx"

ID_COLS = ["ID", "wave", "year"]
WEIGHT_COLS = ["INDV_weight", "HH_weight"]
TARGET_COL = "labor_participation"

FEATURE_GROUPS = {
    "demographic": [
        "age",
        "female",
        "married",
        "rural",
        "edu",
        "year",
    ],
    "health": [
        "poor_health",
        "chronic_count",
        "adl_limit",
        "iadl_limit",
        "depression_high",
        "total_cognition",
        "hibpe",
        "diabe",
        "hearte",
        "stroke",
        "arthre",
        "kidneye",
        "digeste",
        "ins",
    ],
    "family": [
        "hchild",
        "family_size",
        "co_reside_child",
        "care_elder_or_disabled",
        "intergen_support_in",
        "intergen_support_out",
        "family_care_index_v1",
    ],
    "economic": [
        "log_hhcperc_v1",
        "medical_expense",
        "medical_burden",
        "economic_pressure_index_v1",
    ],
    "behavior": [
        "smokev",
        "drinkl",
        "exercise",
        "totmet",
    ],
}

IMPUTE_BY_YEAR = {
    "poor_health",
    "chronic_count",
    "adl_limit",
    "iadl_limit",
    "depression_high",
    "total_cognition",
    "co_reside_child",
    "intergen_support_in",
    "intergen_support_out",
    "log_hhcperc_v1",
    "medical_expense",
    "medical_burden",
    "economic_pressure_index_v1",
    "drinkl",
    "smokev",
    "exercise",
    "totmet",
}

BINARY_FEATURES = {
    "female",
    "married",
    "rural",
    "poor_health",
    "adl_limit",
    "iadl_limit",
    "depression_high",
    "hibpe",
    "diabe",
    "hearte",
    "stroke",
    "arthre",
    "kidneye",
    "digeste",
    "ins",
    "co_reside_child",
    "care_elder_or_disabled",
    "smokev",
    "drinkl",
    "exercise",
}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def flatten_feature_groups() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            pairs.append((group, column))
    return pairs


def year_group_fill(series: pd.Series, years: pd.Series) -> tuple[pd.Series, str]:
    numeric = safe_numeric(series)
    if series.name in BINARY_FEATURES:
        fill_method = "mode_by_year_then_global_mode"
        filled = numeric.copy()
        for year, idx in years.groupby(years).groups.items():
            year_values = numeric.loc[idx].dropna()
            if not year_values.empty:
                filled.loc[idx] = numeric.loc[idx].fillna(year_values.mode().iloc[0])
        global_mode = numeric.dropna().mode()
        if not global_mode.empty:
            filled = filled.fillna(global_mode.iloc[0])
        return filled, fill_method

    fill_method = "median_by_year_then_global_median"
    filled = numeric.copy()
    for year, idx in years.groupby(years).groups.items():
        year_values = numeric.loc[idx].dropna()
        if not year_values.empty:
            filled.loc[idx] = numeric.loc[idx].fillna(year_values.median())
    global_median = numeric.dropna().median()
    if not np.isnan(global_median):
        filled = filled.fillna(global_median)
    return filled, fill_method


def build_feature_roles() -> pd.DataFrame:
    rows = []
    for group, feature in flatten_feature_groups():
        rows.append({"feature": feature, "group": group, "role": "model_feature"})
    rows.append({"feature": TARGET_COL, "group": "target", "role": "target"})
    for column in ID_COLS:
        rows.append({"feature": column, "group": "id", "role": "traceability_only"})
    for column in WEIGHT_COLS:
        rows.append({"feature": column, "group": "weight", "role": "analysis_weight"})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_MODEL, low_memory=False)
    feature_cols = []
    for feature, _group in [(feature, group) for group, feature in flatten_feature_groups()]:
        if feature not in feature_cols:
            feature_cols.append(feature)
    keep_cols = []
    for column in ID_COLS + WEIGHT_COLS + [TARGET_COL, "labor_anomaly_flag"] + feature_cols:
        if column in df.columns and column not in keep_cols:
            keep_cols.append(column)

    base = df[keep_cols].copy()
    base[TARGET_COL] = safe_numeric(base[TARGET_COL])
    base["labor_anomaly_flag"] = safe_numeric(base["labor_anomaly_flag"])

    sample_flow = [
        {"step": "source_v2_model_age90", "n": len(base)},
        {"step": "drop_missing_target", "n": int(base[TARGET_COL].notna().sum())},
    ]

    base = base.loc[base[TARGET_COL].notna()].copy()
    base = base.loc[base["labor_anomaly_flag"] != 1].copy()
    sample_flow.append({"step": "drop_labor_anomalies", "n": len(base)})

    base = base.drop(columns=["labor_anomaly_flag"])

    feature_roles = build_feature_roles()
    missing_before = (
        base[feature_cols]
        .isna()
        .mean()
        .rename("missing_rate_before")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    imputation_rows = []
    year_series = safe_numeric(base["year"])

    for feature in feature_cols:
        if feature not in base.columns:
            continue
        missing_rate = float(base[feature].isna().mean())
        missing_count = int(base[feature].isna().sum())
        if missing_rate > 0:
            base[f"{feature}_missing"] = np.where(base[feature].isna(), 1.0, 0.0)
            feature_roles = pd.concat(
                [
                    feature_roles,
                    pd.DataFrame(
                        [{"feature": f"{feature}_missing", "group": "missing_flag", "role": "missing_indicator"}]
                    ),
                ],
                ignore_index=True,
            )
        if feature in IMPUTE_BY_YEAR:
            filled, method = year_group_fill(base[feature], year_series)
        else:
            filled = safe_numeric(base[feature])
            global_median = filled.dropna().median()
            if not np.isnan(global_median):
                filled = filled.fillna(global_median)
            method = "global_median"
            if feature in BINARY_FEATURES:
                global_mode = safe_numeric(base[feature]).dropna().mode()
                if not global_mode.empty:
                    filled = safe_numeric(base[feature]).fillna(global_mode.iloc[0])
                    method = "global_mode"
        base[feature] = filled
        imputation_rows.append(
            {
                "feature": feature,
                "missing_rate_before": round(missing_rate, 6),
                "imputation_method": method,
                "missing_count_before": missing_count,
            }
        )

    for feature in feature_cols:
        if feature in BINARY_FEATURES and feature in base.columns:
            base[feature] = safe_numeric(base[feature]).round().clip(0, 1)

    for weight_col in WEIGHT_COLS:
        if weight_col not in base.columns:
            continue
        base[weight_col] = safe_numeric(base[weight_col])
        if base[weight_col].isna().any():
            base[f"{weight_col}_missing"] = np.where(base[weight_col].isna(), 1.0, 0.0)
            feature_roles = pd.concat(
                [
                    feature_roles,
                    pd.DataFrame(
                        [{"feature": f"{weight_col}_missing", "group": "missing_flag", "role": "missing_indicator"}]
                    ),
                ],
                ignore_index=True,
            )
            filled, _method = year_group_fill(base[weight_col], year_series)
            base[weight_col] = filled

    ordered_cols = []
    for column in (
        [column for column in ID_COLS if column in base.columns]
        + [TARGET_COL]
        + feature_cols
        + sorted([column for column in base.columns if column.endswith("_missing")])
        + [column for column in WEIGHT_COLS if column in base.columns]
    ):
        if column not in ordered_cols:
            ordered_cols.append(column)
    train_ready = base[ordered_cols].copy()

    missing_after = (
        train_ready.isna()
        .mean()
        .rename("missing_rate_after")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    summary = pd.DataFrame(
        [
            {"metric": "final_rows", "value": len(train_ready)},
            {"metric": "feature_count", "value": len(feature_cols)},
            {"metric": "missing_indicator_count", "value": len([c for c in train_ready.columns if c.endswith("_missing")])},
            {"metric": "target_positive_rate", "value": float(train_ready[TARGET_COL].mean())},
            {"metric": "target_negative_rate", "value": float(1 - train_ready[TARGET_COL].mean())},
        ]
    )

    train_ready.to_csv(TRAIN_READY, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(FEATURES_XLSX, engine="openpyxl") as writer:
        feature_roles.to_excel(writer, sheet_name="feature_roles", index=False)

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(sample_flow).to_excel(writer, sheet_name="sample_flow", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        missing_before.to_excel(writer, sheet_name="missing_before", index=False)
        pd.DataFrame(imputation_rows).to_excel(writer, sheet_name="imputation", index=False)
        missing_after.to_excel(writer, sheet_name="missing_after", index=False)

    print(f"source rows: {len(df)}")
    print(f"train_ready rows: {len(train_ready)}")
    print(f"target positive rate: {train_ready[TARGET_COL].mean():.4f}")
    print(f"output: {TRAIN_READY}")


if __name__ == "__main__":
    main()
