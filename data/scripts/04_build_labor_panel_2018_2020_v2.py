from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "整理完的-charls数据"
OUTPUT_DIR = ROOT / "outputs"

V1_FULL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_full.csv"
V1_MODEL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_model.csv"
V1_MODEL_AGE90 = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_model_age90.csv"
ANOMALIES_V1 = OUTPUT_DIR / "labor_logic_anomalies_v1.csv"

V2_FULL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v2_full.csv"
V2_MODEL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v2_model.csv"
V2_MODEL_AGE90 = BASE_DIR / "CHARLS_labor_panel_2018_2020_v2_model_age90.csv"
V2_ML = BASE_DIR / "CHARLS_labor_panel_2018_2020_v2_ml.csv"
V2_REPORT = BASE_DIR / "CHARLS_labor_cleaning_report_2018_2020_v2.xlsx"

ANOMALIES_V2 = OUTPUT_DIR / "labor_logic_anomalies_v2.csv"
ANOMALIES_SUMMARY_V2 = OUTPUT_DIR / "labor_logic_anomalies_summary_v2.xlsx"
PENSION_CHECK_V2 = OUTPUT_DIR / "pension_check_v2.xlsx"
FINAL_MODEL_VARLIST_V2 = OUTPUT_DIR / "final_model_variable_list_v2.xlsx"
DESCRIPTIVE_TABLES_V2 = OUTPUT_DIR / "descriptive_tables_for_paper_v2.xlsx"

HEALTH_CORE = [
    "srh",
    "poor_health",
    "chronic_count",
    "adlab_c",
    "adl_limit",
    "iadl",
    "iadl_limit",
    "disability",
    "cesd10",
    "depression_high",
    "total_cognition",
    "bmi",
    "wspeed",
    "lgrip",
    "rgrip",
    "sleep",
]
FAMILY_CORE_V1 = [
    "hchild",
    "family_size",
    "intergen_support_out",
    "care_elder_or_disabled",
    "co_reside_child",
]
ECON_CORE_V1 = [
    "pension",
    "income_total",
    "hhcperc",
    "intergen_support_in",
    "intergen_support_out",
    "medical_expense",
]

MODEL_VARIABLE_GROUPS = {
    "dependent_variable": ["labor_participation"],
    "core_explanatory_variables": [
        "poor_health",
        "chronic_count",
        "adl_limit",
        "iadl_limit",
        "depression_high",
        "total_cognition",
        "family_care_index_v1",
        "co_reside_child",
        "intergen_support_out",
        "economic_pressure_index_v1",
        "log_hhcperc_v1",
        "medical_burden",
        "pension",
        "ins",
    ],
    "control_variables": [
        "age",
        "female",
        "married",
        "rural",
        "edu",
        "year",
        "province",
        "smokev",
        "drinkl",
        "exercise",
    ],
    "heterogeneity_variables": [
        "female",
        "rural",
        "age_group",
        "poor_health",
        "family_care_index_v1",
        "economic_pressure_index_v1",
    ],
    "stratification_variables": ["suitability_group_v2"],
    "ml_candidate_variables": [
        "labor_participation",
        "age",
        "female",
        "married",
        "rural",
        "edu",
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
        "hchild",
        "family_size",
        "co_reside_child",
        "intergen_support_in",
        "intergen_support_out",
        "family_care_index_v1",
        "log_hhcperc_v1",
        "pension",
        "ins",
        "medical_expense",
        "medical_burden",
        "economic_pressure_index_v1",
        "smokev",
        "drinkl",
        "exercise",
        "totmet",
        "year",
    ],
}

ANOMALY_TYPES = [
    "labor0_hours_positive",
    "labor0_days_positive",
    "labor1_no_work_flags",
    "excessive_hours",
    "excessive_days",
    "age_outlier",
    "pension_labor_conflict",
]


def normalize_key_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].round().astype("Int64").astype("string")
    return out


def normalize_keys(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for key in keys:
        out[key] = normalize_key_series(out[key])
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_v1_full() -> pd.DataFrame:
    df = pd.read_csv(V1_FULL)
    df = normalize_keys(df, ["ID", "householdID", "communityID"])
    df["wave"] = safe_numeric(df["wave"]).astype("Int64")
    df["year"] = safe_numeric(df["year"]).astype("Int64")
    return df


def build_family_high_flag(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float")
    for year, idx in df.groupby("year").groups.items():
        median_value = safe_numeric(df.loc[idx, "family_care_index_v1"]).median(skipna=True)
        current = safe_numeric(df.loc[idx, "family_care_index_v1"])
        out.loc[idx] = np.where(
            current.notna(),
            np.where((current > median_value) | (current >= 3), 1.0, 0.0),
            np.nan,
        )
    return out


def build_econ_high_flag(df: pd.DataFrame) -> pd.Series:
    current = safe_numeric(df["economic_pressure_index_v1"])
    return np.where(current.notna(), np.where(current >= 3, 1.0, 0.0), np.nan)


def add_suitability_group_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["family_care_index_v1_high_v2"] = build_family_high_flag(out)
    out["economic_pressure_index_v1_high_v2"] = build_econ_high_flag(out)

    labor = safe_numeric(out["labor_participation"])
    poor = safe_numeric(out["poor_health"]) == 1
    adl = safe_numeric(out["adl_limit"]) == 1
    iadl = safe_numeric(out["iadl_limit"]) == 1
    health_constraints = poor | adl | iadl
    family_high = safe_numeric(out["family_care_index_v1_high_v2"]) == 1
    econ_high = safe_numeric(out["economic_pressure_index_v1_high_v2"]) == 1
    healthy = (
        (safe_numeric(out["poor_health"]) == 0)
        & (safe_numeric(out["adl_limit"]) == 0)
        & (safe_numeric(out["iadl_limit"]) == 0)
    )
    age = safe_numeric(out["age"])

    out["suitability_group_v2"] = "unknown"

    cond_f = (labor == 1) & (health_constraints | family_high)
    cond_e = health_constraints & econ_high & ((labor == 0) | age.gt(65))
    cond_b = (labor == 1) & econ_high & ~cond_e
    cond_a = (labor == 1) & healthy & (~family_high) & (~econ_high)
    cond_c = (labor == 0) & (health_constraints | family_high)
    cond_d = (labor == 0) & healthy & (~family_high) & age.le(65)

    out.loc[cond_f, "suitability_group_v2"] = "F"
    out.loc[(out["suitability_group_v2"] == "unknown") & cond_e, "suitability_group_v2"] = "E"
    out.loc[(out["suitability_group_v2"] == "unknown") & cond_b, "suitability_group_v2"] = "B"
    out.loc[(out["suitability_group_v2"] == "unknown") & cond_a, "suitability_group_v2"] = "A"
    out.loc[(out["suitability_group_v2"] == "unknown") & cond_c, "suitability_group_v2"] = "C"
    out.loc[(out["suitability_group_v2"] == "unknown") & cond_d, "suitability_group_v2"] = "D"

    unknown_reason = pd.Series("other_conflict_or_missing", index=out.index, dtype="object")
    unknown_reason.loc[labor.isna()] = "missing_labor_participation"
    unknown_reason.loc[
        (out["suitability_group_v2"] == "unknown")
        & (safe_numeric(out["poor_health"]).isna() | safe_numeric(out["adl_limit"]).isna() | safe_numeric(out["iadl_limit"]).isna())
    ] = "missing_core_health"
    unknown_reason.loc[
        (out["suitability_group_v2"] == "unknown")
        & safe_numeric(out["family_care_index_v1_high_v2"]).isna()
    ] = "missing_family_care_index_v1"
    unknown_reason.loc[
        (out["suitability_group_v2"] == "unknown")
        & safe_numeric(out["economic_pressure_index_v1_high_v2"]).isna()
    ] = "missing_economic_pressure_index_v1"
    unknown_reason.loc[
        (out["suitability_group_v2"] == "unknown")
        & (labor == 0)
        & healthy
        & (~family_high)
        & age.gt(65)
    ] = "healthy_nonworking_age_gt65"
    unknown_reason.loc[
        (out["suitability_group_v2"] == "unknown")
        & (labor == 1)
        & healthy
        & (~family_high)
        & (~econ_high)
    ] = "working_but_other_conflict"
    unknown_reason.loc[out["suitability_group_v2"] != "unknown"] = np.nan
    out["suitability_group_v2_unknown_reason"] = unknown_reason

    total_dist = (
        out["suitability_group_v2"]
        .value_counts(dropna=False)
        .rename_axis("suitability_group_v2")
        .reset_index(name="n")
    )
    by_year = (
        out.groupby(["year", "suitability_group_v2"])
        .size()
        .rename("n")
        .reset_index()
    )
    return out, total_dist, by_year


def load_anomalies_v1() -> pd.DataFrame:
    df = pd.read_csv(ANOMALIES_V1)
    df = normalize_keys(df, ["ID", "householdID", "communityID"])
    df["wave"] = safe_numeric(df["wave"]).astype("Int64")
    df["year"] = safe_numeric(df["year"]).astype("Int64")
    return df


def build_anomalies_v2(full: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    base = full.copy()
    labor = safe_numeric(base["labor_participation"])
    hours = safe_numeric(base["work_hours_weekly"])
    days = safe_numeric(base["work_days_yearly"])
    age = safe_numeric(base["age"])
    pension = safe_numeric(base["pension"])
    flag_sum = (
        safe_numeric(base["farm_work_flag"]).fillna(0)
        + safe_numeric(base["nonfarm_work_flag"]).fillna(0)
        + safe_numeric(base["employed_flag"]).fillna(0)
        + safe_numeric(base["self_employed_flag"]).fillna(0)
        + safe_numeric(base["side_job_flag"]).fillna(0)
    )

    pension_conflict_mask = (
        safe_numeric(base["year"]) == 2018
    ) & (pension == 0) & (labor == 1)

    condition_map = {
        "labor0_hours_positive": (labor == 0) & hours.gt(0),
        "labor0_days_positive": (labor == 0) & days.gt(0),
        "labor1_no_work_flags": (labor == 1) & (flag_sum == 0),
        "excessive_hours": hours.gt(100),
        "excessive_days": days.gt(365),
        "age_outlier": age.gt(90) | age.lt(45),
        "pension_labor_conflict": pension_conflict_mask,
    }

    anomaly_rows = []
    for anomaly_type, mask in condition_map.items():
        if mask.any():
            subset = base.loc[mask].copy()
            subset["anomaly_type"] = anomaly_type
            anomaly_rows.append(subset)
    anomalies = pd.concat(anomaly_rows, ignore_index=True) if anomaly_rows else pd.DataFrame(columns=list(base.columns) + ["anomaly_type"])

    anomalies["labor_anomaly_flag"] = 1
    anomalies["labor_anomaly_type"] = anomalies["anomaly_type"]
    anomalies = anomalies.sort_values(["year", "ID", "anomaly_type"], ignore_index=True)
    anomalies.to_csv(ANOMALIES_V2, index=False, encoding="utf-8-sig")

    type_counts = (
        anomalies["anomaly_type"]
        .value_counts(dropna=False)
        .reindex(ANOMALY_TYPES, fill_value=0)
        .rename_axis("anomaly_type")
        .reset_index(name="n")
    )
    by_year = (
        anomalies.groupby(["year", "anomaly_type"])
        .size()
        .rename("n")
        .reset_index()
    )
    by_labor_type = (
        anomalies.groupby(["labor_type", "anomaly_type"])
        .size()
        .rename("n")
        .reset_index()
    )
    summary_tables = {
        "anomaly_type_counts": type_counts,
        "anomaly_by_year": by_year,
        "anomaly_by_labor_type": by_labor_type,
    }
    with pd.ExcelWriter(ANOMALIES_SUMMARY_V2, engine="openpyxl") as writer:
        for sheet, table in summary_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    # collapse to record-level flags on full dataset
    anomaly_map = (
        anomalies.groupby(["ID", "wave"])["anomaly_type"]
        .agg(lambda vals: "; ".join(sorted(set(vals))))
        .reset_index()
    )
    anomaly_map = normalize_keys(anomaly_map, ["ID"])
    anomaly_map["wave"] = safe_numeric(anomaly_map["wave"]).astype("Int64")
    return anomalies, summary_tables, anomaly_map


def build_pension_checks(full: pd.DataFrame) -> dict[str, pd.DataFrame]:
    year_pension = (
        full.groupby(["year", "pension"])
        .size()
        .rename("n")
        .reset_index()
    )
    year_pension_labor = (
        full.groupby(["year", "pension", "labor_participation"])
        .size()
        .rename("n")
        .reset_index()
    )
    year_pension_age = (
        full.groupby(["year", "pension", "age_group"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    pension_missing = (
        full.groupby("year")["pension"]
        .apply(lambda s: s.isna().mean())
        .rename("missing_rate")
        .reset_index()
    )
    pension_retire = (
        full.groupby(["year", "pension", "retired_flag_raw"])
        .size()
        .rename("n")
        .reset_index()
    )
    pension_age = (
        full.groupby(["year", "pension", "age_group"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )

    notes = pd.DataFrame(
        [
            {
                "issue": "2018 pension=0 and labor=0 count",
                "value": int(
                    (
                        (safe_numeric(full["year"]) == 2018)
                        & (safe_numeric(full["pension"]) == 0)
                        & (safe_numeric(full["labor_participation"]) == 0)
                    ).sum()
                ),
                "interpretation": "If this is 0, keep it as an observed structure and treat pension mainly as part of economic pressure rather than a standalone core explanatory variable.",
            }
        ]
    )

    tables = {
        "year_pension": year_pension,
        "year_pension_labor": year_pension_labor,
        "year_pension_age_group": year_pension_age,
        "pension_missing_rate": pension_missing,
        "pension_retired_crosstab": pension_retire,
        "pension_age_group_crosstab": pension_age,
        "notes": notes,
    }
    with pd.ExcelWriter(PENSION_CHECK_V2, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)
    return tables


def build_final_model_variable_list() -> pd.DataFrame:
    rows = []
    for group_name, variables in MODEL_VARIABLE_GROUPS.items():
        for variable in variables:
            rows.append({"group": group_name, "variable": variable})
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(FINAL_MODEL_VARLIST_V2, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="variables", index=False)
    return df


def build_model_outputs(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_mask = (
        safe_numeric(full["age"]).ge(45)
        & safe_numeric(full["labor_participation"]).notna()
        & full[HEALTH_CORE].notna().any(axis=1)
        & full[FAMILY_CORE_V1].notna().any(axis=1)
        & full[ECON_CORE_V1].notna().any(axis=1)
        & ~full.duplicated(subset=["ID", "wave"], keep=False)
    )
    model = full.loc[model_mask].copy()
    model_age90 = model.loc[safe_numeric(model["age"]).le(90)].copy()

    ml_cols = [col for col in MODEL_VARIABLE_GROUPS["ml_candidate_variables"] if col in model.columns]
    ml = model[ml_cols].copy()
    return model, model_age90, ml


def build_descriptive_tables(full: pd.DataFrame, model: pd.DataFrame, model_age90: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sample_flow = pd.DataFrame(
        [
            {"step": "v1_full input", "n": len(full)},
            {"step": "v2_full", "n": len(full)},
            {"step": "v2_model", "n": len(model)},
            {"step": "v2_model_age90", "n": len(model_age90)},
        ]
    )
    labor_rate_by_year = (
        full.groupby("year")["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_age_group = (
        full.groupby("age_group", observed=False)["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_gender = (
        full.groupby("female")["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_rural = (
        full.groupby("rural")["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_health = (
        full.groupby(["poor_health", "adl_limit", "iadl_limit"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_family = (
        full.groupby("family_care_index_v1_high_v2")["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_rate_by_econ = (
        full.groupby("economic_pressure_index_v1_high_v2")["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    retire_labor = (
        full.groupby(["retired_flag_raw", "labor_participation"])
        .size()
        .rename("n")
        .reset_index()
    )
    suitability_dist = (
        full.groupby(["year", "suitability_group_v2"])
        .size()
        .rename("n")
        .reset_index()
    )
    return {
        "sample_flow": sample_flow,
        "labor_rate_by_year": labor_rate_by_year,
        "labor_rate_by_age_group": labor_rate_by_age_group,
        "labor_rate_by_gender": labor_rate_by_gender,
        "labor_rate_by_rural": labor_rate_by_rural,
        "labor_rate_by_health": labor_rate_by_health,
        "labor_rate_by_family_care": labor_rate_by_family,
        "labor_rate_by_economic_pressure": labor_rate_by_econ,
        "retire_labor_crosstab": retire_labor,
        "suitability_group_v2_distribution": suitability_dist,
    }


def build_report_tables(
    full: pd.DataFrame,
    model: pd.DataFrame,
    model_age90: pd.DataFrame,
    pension_tables: dict[str, pd.DataFrame],
    anomaly_tables: dict[str, pd.DataFrame],
    final_varlist: pd.DataFrame,
    descriptive_tables: dict[str, pd.DataFrame],
    suitability_total: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    wave_distribution = (
        full.groupby(["wave", "year"])
        .size()
        .rename("n")
        .reset_index()
    )
    labor_distribution = (
        full.groupby(["year", "labor_participation"])
        .size()
        .rename("n")
        .reset_index()
    )
    labor_by_age_gender_rural = (
        full.groupby(["year", "age_group", "female", "rural"], observed=False)["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_health = (
        full.groupby(["year", "poor_health", "adl_limit", "iadl_limit"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_family = (
        full.groupby(["year", "family_care_index_v1_high_v2"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_econ = (
        full.groupby(["year", "economic_pressure_index_v1_high_v2"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    retire_labor = (
        full.groupby(["year", "retired_flag_raw", "labor_participation"])
        .size()
        .rename("n")
        .reset_index()
    )
    missing_rate = (
        full.isna().mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "variable"})
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )
    age_check = (
        full.groupby("year")["age"]
        .agg(["count", "min", "median", "max", "mean"])
        .reset_index()
    )
    suit_by_year = (
        full.groupby(["year", "suitability_group_v2"])
        .size()
        .rename("n")
        .reset_index()
    )
    overall_block = suitability_total.copy()
    overall_block["year"] = pd.Series([pd.NA] * len(overall_block), dtype="Int64")
    overall_block["section"] = "overall"
    by_year_block = suit_by_year.copy()
    by_year_block["section"] = "by_year"
    suit_sheet = pd.concat(
        [
            by_year_block[["year", "suitability_group_v2", "n", "section"]],
            overall_block[["year", "suitability_group_v2", "n", "section"]],
        ],
        ignore_index=True,
    )
    return {
        "sample_flow": descriptive_tables["sample_flow"],
        "wave_distribution": wave_distribution,
        "labor_distribution": labor_distribution,
        "labor_by_age_gender_rural": labor_by_age_gender_rural,
        "labor_by_health": labor_by_health,
        "labor_by_family_care_v1": labor_by_family,
        "labor_by_economic_pressure_v1": labor_by_econ,
        "retire_labor_crosstab": retire_labor,
        "pension_labor_crosstab": pension_tables["year_pension_labor"],
        "pension_check_v2": pension_tables["notes"],
        "labor_logic_anomalies_summary_v2": anomaly_tables["anomaly_type_counts"],
        "missing_rate": missing_rate,
        "final_model_variable_list_v2": final_varlist,
        "age_check": age_check,
        "suitability_group_v2_distribution": suit_sheet,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full = load_v1_full()
    full, suitability_total, suitability_by_year = add_suitability_group_v2(full)
    anomalies_v2, anomaly_tables, anomaly_map = build_anomalies_v2(full)

    full = full.merge(anomaly_map, on=["ID", "wave"], how="left")
    full["labor_anomaly_flag"] = np.where(full["anomaly_type"].notna(), 1, 0)
    full["labor_anomaly_type"] = full["anomaly_type"]
    full = full.drop(columns=["anomaly_type"])

    pension_tables = build_pension_checks(full)
    final_varlist = build_final_model_variable_list()
    model, model_age90, ml = build_model_outputs(full)

    full.to_csv(V2_FULL, index=False, encoding="utf-8-sig")
    model.to_csv(V2_MODEL, index=False, encoding="utf-8-sig")
    model_age90.to_csv(V2_MODEL_AGE90, index=False, encoding="utf-8-sig")
    ml.to_csv(V2_ML, index=False, encoding="utf-8-sig")

    descriptive_tables = build_descriptive_tables(full, model, model_age90)
    with pd.ExcelWriter(DESCRIPTIVE_TABLES_V2, engine="openpyxl") as writer:
        for sheet, table in descriptive_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    report_tables = build_report_tables(
        full,
        model,
        model_age90,
        pension_tables,
        anomaly_tables,
        final_varlist,
        descriptive_tables,
        suitability_total,
    )
    with pd.ExcelWriter(V2_REPORT, engine="openpyxl") as writer:
        for sheet, table in report_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    labor_rate = full.groupby("year")["labor_participation"].mean()
    pension_zero_nonwork_2018 = int(
        (
            (safe_numeric(full["year"]) == 2018)
            & (safe_numeric(full["pension"]) == 0)
            & (safe_numeric(full["labor_participation"]) == 0)
        ).sum()
    )

    print(f"v2_full sample n: {len(full)}")
    print(f"v2_model sample n: {len(model)}")
    print(f"v2_model_age90 sample n: {len(model_age90)}")
    print(f"v2_ml sample n: {len(ml)}")
    for year, rate in labor_rate.items():
        print(f"{int(year)} labor participation rate: {rate:.4f}")
    print("suitability_group_v2 distribution:")
    for row in suitability_total.itertuples(index=False):
        print(f"  {row.suitability_group_v2}: {row.n}")
    print("labor_logic_anomalies type counts:")
    for row in anomaly_tables["anomaly_type_counts"].itertuples(index=False):
        print(f"  {row.anomaly_type}: {row.n}")
    print("pension variable recommendation:")
    if pension_zero_nonwork_2018 == 0:
        print("  Not recommended as a standalone core explanatory variable.")
        print("  Reason: in 2018 the cell pension=0 and labor_participation=0 remains 0, suggesting strong structure or coding concentration.")
        print("  Better use pension inside economic_pressure_index_v1 and as a descriptive auxiliary variable.")
    else:
        print("  Can remain as an auxiliary explanatory variable, but still better interpreted jointly with economic pressure.")
    print("family pressure wording recommendation:")
    print("  Yes, recommend the paper wording '家庭照料与代际支持压力'.")
    print("  Reason: the stable comparable signals are co-residence, elder/disabled care, and intergenerational support outflows, not only direct caregiving.")
    print("next step recommendation:")
    print("  Yes, after reviewing the anomaly summaries and pension check table, you can start modeling.")


if __name__ == "__main__":
    main()
