from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyreadstat
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install with: pip install pandas pyreadstat openpyxl"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "整理完的-charls数据"
RAW_DIR = ROOT / "原始数据+问卷2011~2020"
OUTPUT_DIR = ROOT / "outputs"

BASE_CSV = BASE_DIR / "CHARLS.csv"
BASE_DTA = BASE_DIR / "charls.dta"

FULL_CSV = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_full.csv"
MODEL_CSV = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_model.csv"
FULL_DTA = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_full.dta"
MODEL_DTA = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_model.dta"
REPORT_XLSX = BASE_DIR / "CHARLS_labor_cleaning_report_2018_2020_v0.xlsx"

VARIABLE_SOURCE_CSV = OUTPUT_DIR / "variable_source_2018_2020_v0.csv"
LABOR_CANDIDATES_XLSX = OUTPUT_DIR / "labor_variable_candidates_2018_2020.xlsx"
FAMILY_CANDIDATES_XLSX = OUTPUT_DIR / "family_care_variable_candidates_2018_2020.xlsx"
ECON_CANDIDATES_XLSX = OUTPUT_DIR / "economic_pressure_variable_candidates_2018_2020.xlsx"
HEALTH_CANDIDATES_XLSX = OUTPUT_DIR / "health_variable_candidates_2018_2020.xlsx"
DUPLICATE_CSV = OUTPUT_DIR / "duplicate_id_wave_records_2018_2020.csv"

WAVE_YEAR_MAP = {4: 2018, 5: 2020}

LABOR_KEYWORDS = [
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
    "self-employed",
    "self employed",
    "business",
    "hours",
    "income",
    "pension",
]
FAMILY_KEYWORDS = [
    "child",
    "children",
    "grandchild",
    "grandchildren",
    "care",
    "self-care",
    "self care",
    "support",
    "transfer",
    "parent",
    "spouse",
    "living regularly",
    "live with",
    "disabled",
]
ECON_KEYWORDS = [
    "income",
    "wage",
    "salary",
    "pension",
    "medical",
    "doctor",
    "hospital",
    "insurance",
    "expense",
    "support",
    "transfer",
    "consumption",
    "subsidy",
]
HEALTH_KEYWORDS = [
    "health",
    "disease",
    "disabled",
    "disability",
    "adl",
    "iadl",
    "depression",
    "cognition",
    "memory",
    "sleep",
    "grip",
    "walk",
    "speed",
    "bmi",
    "chronic",
    "stroke",
    "heart",
    "diabetes",
    "hypertension",
]

MODULES_TO_SCAN = {
    2018: {
        "Work_Retirement": RAW_DIR / "2018" / "CHARLS2018r" / "Work_Retirement.dta",
        "Demographic_Background": RAW_DIR / "2018" / "CHARLS2018r" / "Demographic_Background.dta",
        "Health_Status_and_Functioning": RAW_DIR / "2018" / "CHARLS2018r" / "Health_Status_and_Functioning.dta",
        "Family_Information": RAW_DIR / "2018" / "CHARLS2018r" / "Family_Information.dta",
        "Household_Income": RAW_DIR / "2018" / "CHARLS2018r" / "Household_Income.dta",
        "Individual_Income": RAW_DIR / "2018" / "CHARLS2018r" / "Individual_Income.dta",
        "Sample_Infor": RAW_DIR / "2018" / "CHARLS2018r" / "Sample_Infor.dta",
        "Weights": RAW_DIR / "2018" / "CHARLS2018r" / "Weights.dta",
    },
    2020: {
        "Work_Retirement": RAW_DIR / "2020" / "CHARLS2020r" / "Work_Retirement.dta",
        "Demographic_Background": RAW_DIR / "2020" / "CHARLS2020r" / "Demographic_Background.dta",
        "Health_Status_and_Functioning": RAW_DIR / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta",
        "Family_Information": RAW_DIR / "2020" / "CHARLS2020r" / "Family_Information.dta",
        "Household_Income": RAW_DIR / "2020" / "CHARLS2020r" / "Household_Income.dta",
        "Individual_Income": RAW_DIR / "2020" / "CHARLS2020r" / "Individual_Income.dta",
        "Sample_Infor": RAW_DIR / "2020" / "CHARLS2020r" / "Sample_Infor.dta",
        "Weights": RAW_DIR / "2020" / "CHARLS2020r" / "Weights.dta",
    },
}

BASE_KEEP_COLS = [
    "ID",
    "wave",
    "householdID",
    "communityID",
    "iwy",
    "iwm",
    "gender",
    "marry",
    "rural",
    "rural2",
    "srh",
    "adlab_c",
    "hibpe",
    "diabe",
    "cancre",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "dyslipe",
    "livere",
    "kidneye",
    "digeste",
    "asthmae",
    "memrye",
    "drinkev",
    "drinkl",
    "smokev",
    "smoken",
    "hospital",
    "doctor",
    "oophos1y",
    "tothos1y",
    "oopdoc1m",
    "totdoc1m",
    "income_total",
    "hhcperc",
    "family_size",
    "hchild",
    "fcamt",
    "tcamt",
    "retire",
    "wspeed",
    "lgrip",
    "rgrip",
    "bmi",
    "cesd10",
    "sleep",
    "disability",
    "social7",
    "nation",
    "province",
    "city",
    "age",
    "edu",
    "exercise",
    "totmet",
    "total_cognition",
    "pension",
    "ins",
    "iadl",
]

DISEASE_VARS = [
    "hibpe",
    "diabe",
    "cancre",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "dyslipe",
    "livere",
    "kidneye",
    "digeste",
    "asthmae",
    "memrye",
]

FAMILY_CORE_VARS = [
    "hchild",
    "family_size",
    "fcamt",
    "tcamt",
    "care_grandchild",
    "care_elder_or_disabled",
    "co_reside_child",
    "co_reside_parent",
]
ECON_CORE_VARS = [
    "income_total",
    "hhcperc",
    "pension",
    "medical_expense",
    "medical_burden",
    "fcamt",
    "tcamt",
]
HEALTH_CORE_VARS = [
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


def load_base_labels() -> dict[str, dict]:
    meta = pyreadstat.read_dta(BASE_DTA, metadataonly=True)[1]
    variable_to_label = meta.variable_to_label or {}
    value_labels = meta.value_labels or {}
    return {var: value_labels.get(label_name, {}) for var, label_name in variable_to_label.items()}


def scan_module_candidates(keywords: list[str], output_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    lowered_keywords = [k.lower() for k in keywords]

    for year, modules in MODULES_TO_SCAN.items():
        for module_name, module_path in modules.items():
            meta = pyreadstat.read_dta(module_path, metadataonly=True)[1]
            labels_map = meta.column_names_to_labels or {}
            value_label_map = meta.variable_to_label or {}
            value_labels = meta.value_labels or {}

            for variable in meta.column_names:
                label = labels_map.get(variable, "") or ""
                haystack = f"{variable} {label}".lower()
                matched = [keyword for keyword in lowered_keywords if keyword in haystack]
                if not matched:
                    continue

                labelset = value_label_map.get(variable, "")
                label_values = value_labels.get(labelset, {})
                value_summary = summarize_value_labels(label_values)
                rows.append(
                    {
                        "year": year,
                        "module": module_name,
                        "source_file": str(module_path.relative_to(ROOT)),
                        "variable_name": variable,
                        "variable_label": label,
                        "matched_keywords": ", ".join(matched),
                        "value_label_summary": value_summary,
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["year", "module", "variable_name"], ignore_index=True
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="candidates")
    return result


def summarize_value_labels(label_values: dict) -> str:
    if not label_values:
        return ""
    pairs = [f"{key}={value}" for key, value in list(label_values.items())[:8]]
    summary = "; ".join(pairs)
    if len(label_values) > 8:
        summary += f"; ... ({len(label_values)} values)"
    return summary


def read_base_sample() -> pd.DataFrame:
    base = pd.read_csv(BASE_CSV, usecols=BASE_KEEP_COLS)
    base = base.loc[base["wave"].isin([4, 5])].copy()
    base["year"] = base["wave"].map(WAVE_YEAR_MAP)
    base = base.loc[base["age"].ge(45)].copy()
    base = base.loc[base["ID"].notna() & base["wave"].notna()].copy()
    return base


def standardize_yes_no_from_labels(series: pd.Series, yes_values: Iterable, no_values: Iterable) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[series.isin(list(yes_values))] = 1.0
    out.loc[series.isin(list(no_values))] = 0.0
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_key_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    mask = numeric.notna()
    out.loc[mask] = numeric.loc[mask].round().astype("Int64").astype("string")
    return out


def normalize_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for key in keys:
        out[key] = normalize_key_series(out[key])
    return out


def set_special_missing(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    return numeric.mask(numeric.isin([993, 995, 997, 999]))


def clip_plausible(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
    series = safe_numeric(series)
    return series.where(series.between(min_value, max_value))


def sum_with_min_count(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].sum(axis=1, min_count=1)


def first_non_missing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].bfill(axis=1).iloc[:, 0]


def max_with_min_count(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    values = df[cols]
    out = values.max(axis=1, skipna=True)
    return out.where(values.notna().any(axis=1))


def read_work_2018() -> pd.DataFrame:
    path = MODULES_TO_SCAN[2018]["Work_Retirement"]
    usecols = [
        "ID",
        "householdID",
        "communityID",
        "xf1",
        "xf4",
        "xf5",
        "fa002_w4",
        "fa003",
        "fb011_w4",
        "fc001",
        "fc008",
        "fc009",
        "fc010",
        "fc011",
        "fc020_w4_a",
        "fc020_w4_c",
        "fc020_w4_d",
        "fc020_w4_f",
        "fc021_w4_c",
        "fc021_w4_f",
        "fe001",
        "fe002",
        "fe003",
        "fh001",
        "fh002",
        "fh003",
        "fj001_w4",
        "fj002_w4",
    ]
    df, _ = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)

    coded_cols = [
        "xf1",
        "xf4",
        "xf5",
        "fa002_w4",
        "fa003",
        "fb011_w4",
        "fc001",
        "fc008",
        "fc020_w4_a",
        "fc020_w4_c",
        "fc020_w4_d",
        "fc020_w4_f",
        "fc021_w4_c",
        "fc021_w4_f",
    ]
    for col in coded_cols:
        df[col] = set_special_missing(df[col])

    df["fe001"] = clip_plausible(df["fe001"], 0, 12)
    df["fe002"] = clip_plausible(df["fe002"], 0, 7)
    df["fe003"] = clip_plausible(df["fe003"], 0, 24)
    df["fh001"] = clip_plausible(df["fh001"], 0, 12)
    df["fh002"] = clip_plausible(df["fh002"], 0, 7)
    df["fh003"] = clip_plausible(df["fh003"], 0, 24)
    df["fj001_w4"] = clip_plausible(df["fj001_w4"], 0, 20)
    df["fj002_w4"] = clip_plausible(df["fj002_w4"], 0, 168)
    df["fc009"] = clip_plausible(df["fc009"], 0, 12)
    df["fc010"] = clip_plausible(df["fc010"], 0, 31)
    df["fc011"] = clip_plausible(df["fc011"], 0, 24)

    df["farm_work_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "xf4": standardize_yes_no_from_labels(df["xf4"], [1], [2]),
                "fc001": standardize_yes_no_from_labels(df["fc001"], [1], [2]),
                "fc008": standardize_yes_no_from_labels(df["fc008"], [1], [2]),
            }
        ),
        ["xf4", "fc001", "fc008"],
    )
    df["nonfarm_work_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fa002_w4": standardize_yes_no_from_labels(df["fa002_w4"], [1], [2]),
                "fa003": standardize_yes_no_from_labels(df["fa003"], [1], [2]),
                "fe_section": df["fe001"].notna().astype(float),
                "fh_section": df["fh001"].notna().astype(float),
            }
        ),
        ["fa002_w4", "fa003", "fe_section", "fh_section"],
    )

    job_type_cols = ["fc020_w4_a", "fc020_w4_c", "fc020_w4_d", "fc020_w4_f", "fc021_w4_c", "fc021_w4_f"]
    employed_sources = [(df[col].isin([1, 4])).astype(float).where(df[col].notna()) for col in job_type_cols]
    self_sources = [(df[col].isin([2, 3])).astype(float).where(df[col].notna()) for col in job_type_cols]

    df["employed_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fc001": standardize_yes_no_from_labels(df["fc001"], [1], [2]),
                "job_type": pd.concat(employed_sources, axis=1).max(axis=1, skipna=True),
                "fe_section": df["fe001"].notna().astype(float),
            }
        ),
        ["fc001", "job_type", "fe_section"],
    )
    df["self_employed_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fc008": standardize_yes_no_from_labels(df["fc008"], [1], [2]),
                "job_type": pd.concat(self_sources, axis=1).max(axis=1, skipna=True),
                "fh_section": df["fh001"].notna().astype(float),
            }
        ),
        ["fc008", "job_type", "fh_section"],
    )
    df["side_job_flag"] = (df["fj001_w4"].fillna(0) > 0).astype(float)
    df.loc[df["fj001_w4"].isna() & df["fj002_w4"].isna(), "side_job_flag"] = np.nan
    df.loc[df["fj002_w4"].fillna(0) > 0, "side_job_flag"] = 1.0

    df["retired_flag_raw"] = standardize_yes_no_from_labels(df["fb011_w4"], [1], [2])

    df["farm_hours_weekly"] = df["fc010"] * df["fc011"]
    df["employed_hours_weekly"] = df["fe002"] * df["fe003"]
    df["self_hours_weekly"] = df["fh002"] * df["fh003"]
    df["side_hours_weekly"] = df["fj002_w4"]
    df["work_hours_weekly"] = sum_with_min_count(
        df,
        ["farm_hours_weekly", "employed_hours_weekly", "self_hours_weekly", "side_hours_weekly"],
    )

    df["farm_days_yearly"] = df["fc009"] * df["fc010"]
    df["employed_days_yearly"] = df["fe001"] * 4.345 * df["fe002"]
    df["self_days_yearly"] = df["fh001"] * 4.345 * df["fh002"]
    df["work_days_yearly"] = sum_with_min_count(
        df, ["farm_days_yearly", "employed_days_yearly", "self_days_yearly"]
    )

    working_status = df["xf1"]
    yes_evidence = (
        (working_status == 3)
        | (df["farm_work_flag"] == 1)
        | (df["nonfarm_work_flag"] == 1)
        | (df["side_job_flag"] == 1)
        | (df["work_days_yearly"].fillna(0) > 0)
        | (df["work_hours_weekly"].fillna(0) > 0)
    )
    no_evidence = (working_status.isin([1, 2])) & ~yes_evidence
    df["labor_participation"] = np.nan
    df.loc[yes_evidence, "labor_participation"] = 1.0
    df.loc[no_evidence, "labor_participation"] = 0.0

    df["labor_type"] = derive_labor_type(
        labor=df["labor_participation"],
        farm=df["farm_work_flag"],
        nonfarm=df["nonfarm_work_flag"],
        employed=df["employed_flag"],
        self_emp=df["self_employed_flag"],
        side=df["side_job_flag"],
    )

    keep_cols = [
        "ID",
        "householdID",
        "communityID",
        "farm_work_flag",
        "nonfarm_work_flag",
        "employed_flag",
        "self_employed_flag",
        "side_job_flag",
        "retired_flag_raw",
        "work_hours_weekly",
        "work_days_yearly",
        "labor_participation",
        "labor_type",
    ]
    result = df[keep_cols].copy()
    result["wave"] = 4
    return normalize_keys(result, ["ID", "householdID", "communityID"])


def read_work_2020() -> pd.DataFrame:
    path = MODULES_TO_SCAN[2020]["Work_Retirement"]
    usecols = [
        "ID",
        "householdID",
        "communityID",
        "fa001",
        "fa002_s1",
        "fa002_s2",
        "fa004",
        "fa009",
        "fa010",
        "fa011",
        "fb005",
        "fb006",
        "fb007",
        "fc025",
        "fc026",
        "fc027",
        "fd007",
        "fd008",
        "fd009",
        "fe001",
        "fe002",
        "ff001",
        "fh001",
        "fh002",
        "xworking",
        "xemployed",
    ]
    df, _ = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)

    coded_cols = [
        "fa001",
        "fa002_s1",
        "fa002_s2",
        "fa004",
        "fa009",
        "fa010",
        "fa011",
        "fh001",
        "fh002",
        "xworking",
        "xemployed",
    ]
    for col in coded_cols:
        df[col] = set_special_missing(df[col])

    for col in ["fb005", "fc025", "fd007"]:
        df[col] = clip_plausible(df[col], 0, 12)
    for col in ["fc026", "fd008", "fe001"]:
        df[col] = clip_plausible(df[col], 0, 7)
    for col in ["fb006"]:
        df[col] = clip_plausible(df[col], 0, 31)
    for col in ["fb007", "fc027", "fd009", "fe002"]:
        df[col] = clip_plausible(df[col], 0, 24)
    df["ff001"] = clip_plausible(df["ff001"], 0, 366)

    df["farm_work_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fa001": standardize_yes_no_from_labels(df["fa001"], [1], [2]),
                "fa002_s1": standardize_yes_no_from_labels(df["fa002_s1"], [1], [0]),
                "fa002_s2": standardize_yes_no_from_labels(df["fa002_s2"], [2], [0]),
                "fb005": df["fb005"].notna().astype(float),
            }
        ),
        ["fa001", "fa002_s1", "fa002_s2", "fb005"],
    )
    df["nonfarm_work_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fa004": standardize_yes_no_from_labels(df["fa004"], [1], [2]),
                "fa010_nonfarm": df["fa010"].isin([1, 2, 3]).astype(float).where(df["fa010"].notna()),
                "fa011_nonfarm": df["fa011"].isin([1, 2, 3]).astype(float).where(df["fa011"].notna()),
                "fc025": df["fc025"].notna().astype(float),
                "fd007": df["fd007"].notna().astype(float),
            }
        ),
        ["fa004", "fa010_nonfarm", "fa011_nonfarm", "fc025", "fd007"],
    )
    df["employed_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fa002_s2": standardize_yes_no_from_labels(df["fa002_s2"], [2], [0]),
                "fa010": df["fa010"].isin([1, 4]).astype(float).where(df["fa010"].notna()),
                "fa011": df["fa011"].isin([1]).astype(float).where(df["fa011"].notna()),
                "fc025": df["fc025"].notna().astype(float),
            }
        ),
        ["fa002_s2", "fa010", "fa011", "fc025"],
    )
    df["self_employed_flag"] = max_with_min_count(
        pd.DataFrame(
            {
                "fa002_s1": standardize_yes_no_from_labels(df["fa002_s1"], [1], [0]),
                "fa010": df["fa010"].isin([2, 3]).astype(float).where(df["fa010"].notna()),
                "fa011": df["fa011"].isin([2, 3]).astype(float).where(df["fa011"].notna()),
                "fd007": df["fd007"].notna().astype(float),
            }
        ),
        ["fa002_s1", "fa010", "fa011", "fd007"],
    )
    df["side_job_flag"] = ((df["fe001"].fillna(0) > 0) | (df["fe002"].fillna(0) > 0)).astype(float)
    df.loc[df["fe001"].isna() & df["fe002"].isna(), "side_job_flag"] = np.nan

    df["retired_flag_raw"] = standardize_yes_no_from_labels(df["fh001"], [1], [2])

    df["farm_hours_weekly"] = df["fb006"] * df["fb007"]
    df["employed_hours_weekly"] = df["fc026"] * df["fc027"]
    df["self_hours_weekly"] = df["fd008"] * df["fd009"]
    df["side_hours_weekly"] = df["fe001"] * df["fe002"]
    df["work_hours_weekly"] = sum_with_min_count(
        df,
        ["farm_hours_weekly", "employed_hours_weekly", "self_hours_weekly", "side_hours_weekly"],
    )

    df["farm_days_yearly"] = df["fb005"] * df["fb006"]
    df["employed_days_yearly"] = df["fc025"] * 4.345 * df["fc026"]
    df["self_days_yearly"] = df["fd007"] * 4.345 * df["fd008"]
    annualized_days = sum_with_min_count(
        df,
        ["farm_days_yearly", "employed_days_yearly", "self_days_yearly"],
    )
    df["work_days_yearly"] = max_with_min_count(
        pd.DataFrame({"ff001": df["ff001"], "annualized": annualized_days}),
        ["ff001", "annualized"],
    )

    yes_evidence = (
        (df["xworking"] == 1)
        | (df["farm_work_flag"] == 1)
        | (df["nonfarm_work_flag"] == 1)
        | (df["side_job_flag"] == 1)
        | (df["work_days_yearly"].fillna(0) > 0)
        | (df["work_hours_weekly"].fillna(0) > 0)
    )
    no_evidence = (df["xworking"] == 0) & ~yes_evidence
    df["labor_participation"] = np.nan
    df.loc[yes_evidence, "labor_participation"] = 1.0
    df.loc[no_evidence, "labor_participation"] = 0.0

    df["labor_type"] = derive_labor_type(
        labor=df["labor_participation"],
        farm=df["farm_work_flag"],
        nonfarm=df["nonfarm_work_flag"],
        employed=df["employed_flag"],
        self_emp=df["self_employed_flag"],
        side=df["side_job_flag"],
    )

    keep_cols = [
        "ID",
        "householdID",
        "communityID",
        "farm_work_flag",
        "nonfarm_work_flag",
        "employed_flag",
        "self_employed_flag",
        "side_job_flag",
        "retired_flag_raw",
        "work_hours_weekly",
        "work_days_yearly",
        "labor_participation",
        "labor_type",
    ]
    result = df[keep_cols].copy()
    result["wave"] = 5
    return normalize_keys(result, ["ID", "householdID", "communityID"])


def derive_labor_type(
    labor: pd.Series,
    farm: pd.Series,
    nonfarm: pd.Series,
    employed: pd.Series,
    self_emp: pd.Series,
    side: pd.Series,
) -> pd.Series:
    labor_type = pd.Series("unknown", index=labor.index, dtype="object")
    labor_type.loc[labor == 0] = "not_working"
    labor_type.loc[(labor == 1) & (farm == 1) & (nonfarm != 1)] = "farm_only"
    labor_type.loc[(labor == 1) & (farm == 1) & (nonfarm == 1)] = "farm_plus_nonfarm"
    labor_type.loc[(labor == 1) & (employed == 1) & (self_emp != 1)] = "employed"
    labor_type.loc[(labor == 1) & (self_emp == 1) & (employed != 1)] = "self_employed"
    labor_type.loc[(labor == 1) & (employed == 1) & (self_emp == 1)] = "mixed_employment"
    labor_type.loc[
        (labor == 1)
        & (farm.fillna(0) != 1)
        & (nonfarm.fillna(0) != 1)
        & (side == 1),
    ] = "side_job_only"
    labor_type.loc[(labor == 1) & (labor_type == "unknown")] = "working_unknown"
    labor_type.loc[labor.isna()] = np.nan
    return labor_type


def read_family_2018() -> pd.DataFrame:
    path = MODULES_TO_SCAN[2018]["Family_Information"]
    meta = pyreadstat.read_dta(path, metadataonly=True)[1]
    cols = meta.column_names
    child_loc_cols = [col for col in cols if col.startswith("cb053_")]
    parent_loc_cols = [col for col in cols if col.startswith("ca016_")]
    grandchild_cols = [col for col in cols if col.startswith("cb067_")]
    parent_selfcare_cols = [col for col in cols if col.startswith("ca026_w3_")]

    usecols = ["ID", "householdID", "communityID"] + child_loc_cols + parent_loc_cols + grandchild_cols + parent_selfcare_cols
    df, _ = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)

    for col in child_loc_cols + parent_loc_cols + parent_selfcare_cols:
        df[col] = set_special_missing(df[col])
    for col in grandchild_cols:
        df[col] = clip_plausible(df[col], 0, 99)

    child_co_reside = df[child_loc_cols].isin([1, 2])
    parent_co_reside = df[parent_loc_cols].isin([1])
    parent_need_help = df[parent_selfcare_cols].isin([2])

    out = df[["ID", "householdID", "communityID"]].copy()
    out["co_reside_child"] = np.where(
        child_co_reside.any(axis=1),
        1.0,
        np.where(df[child_loc_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["co_reside_parent"] = np.where(
        parent_co_reside.any(axis=1),
        1.0,
        np.where(df[parent_loc_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["grandchild_number"] = df[grandchild_cols].sum(axis=1, min_count=1)
    out["parent_need_help_flag_2018"] = np.where(
        parent_need_help.any(axis=1),
        1.0,
        np.where(df[parent_selfcare_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["wave"] = 4
    return normalize_keys(out, ["ID", "householdID", "communityID"])


def read_weights_sample(year: int) -> pd.DataFrame:
    weights_path = MODULES_TO_SCAN[year]["Weights"]
    sample_path = MODULES_TO_SCAN[year]["Sample_Infor"]

    weights, _ = pyreadstat.read_dta(
        weights_path,
        usecols=["ID", "householdID", "communityID", "HH_weight", "INDV_weight"],
        apply_value_formats=False,
    )
    sample, _ = pyreadstat.read_dta(
        sample_path,
        usecols=["ID", "householdID", "communityID", "died", "crosssection", "iyear", "imonth"],
        apply_value_formats=False,
    )
    out = weights.merge(sample, on=["ID", "householdID", "communityID"], how="outer")
    out["wave"] = 4 if year == 2018 else 5
    return normalize_keys(out, ["ID", "householdID", "communityID"])


def build_variable_source_table() -> pd.DataFrame:
    rows = [
        {"final_variable": "ID", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "ID", "construct_rule": "直接保留底表 ID", "notes": ""},
        {"final_variable": "householdID", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "householdID", "construct_rule": "直接保留底表 householdID", "notes": ""},
        {"final_variable": "communityID", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "communityID", "construct_rule": "直接保留底表 communityID", "notes": ""},
        {"final_variable": "wave", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "wave", "construct_rule": "底表筛选 wave in [4,5]", "notes": ""},
        {"final_variable": "year", "source_file": "derived", "raw_variables": "wave", "construct_rule": "wave=4 映射 2018, wave=5 映射 2020", "notes": ""},
        {"final_variable": "province", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "province", "construct_rule": "直接保留底表 province", "notes": ""},
        {"final_variable": "city", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "city", "construct_rule": "直接保留底表 city", "notes": ""},
        {"final_variable": "labor_participation", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: xf1, xf4, fa002_w4, fa003, fc001, fc008, fe*, fh*, fj*; 2020: xworking, fa001, fa002_s1, fa002_s2, fa004, fc*, fd*, fe*, ff001", "construct_rule": "若存在农业/非农/受雇/自雇/副业或工作天数/工时证据则记 1；若工作状态明确为不工作且无正向劳动证据则记 0；否则缺失", "notes": "退休不直接等同退出劳动"},
        {"final_variable": "labor_type", "source_file": "derived", "raw_variables": "farm_work_flag, nonfarm_work_flag, employed_flag, self_employed_flag, side_job_flag", "construct_rule": "分为 farm_only, farm_plus_nonfarm, employed, self_employed, mixed_employment, side_job_only, not_working, working_unknown", "notes": ""},
        {"final_variable": "farm_work_flag", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: xf4, fc001, fc008; 2020: fa001, fa002_s1, fa002_s2, fb005", "construct_rule": "任一农业劳动证据为 1", "notes": ""},
        {"final_variable": "nonfarm_work_flag", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fa002_w4, fa003, fe001, fh001; 2020: fa004, fa010, fa011, fc025, fd007", "construct_rule": "任一非农劳动证据为 1", "notes": ""},
        {"final_variable": "employed_flag", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fc001, fc020*, fc021*, fe001; 2020: fa002_s2, fa010, fa011, fc025", "construct_rule": "受雇农业或受雇非农相关证据为 1", "notes": ""},
        {"final_variable": "self_employed_flag", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fc008, fc020*, fc021*, fh001; 2020: fa002_s1, fa010, fa011, fd007", "construct_rule": "自雇、家庭经营或家庭帮工相关证据为 1", "notes": ""},
        {"final_variable": "side_job_flag", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fj001_w4, fj002_w4; 2020: fe001, fe002", "construct_rule": "副业数量或副业工时大于 0 记 1", "notes": "2020 fe 模块按 side job 解释"},
        {"final_variable": "retired_flag_raw", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fb011_w4; 2020: fh001", "construct_rule": "按退休手续是否办理原始题直接映射", "notes": "不直接替代 labor_participation"},
        {"final_variable": "work_hours_weekly", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fc010*fc011 + fe002*fe003 + fh002*fh003 + fj002_w4; 2020: fb006*fb007 + fc026*fc027 + fd008*fd009 + fe001*fe002", "construct_rule": "按可用劳动模块把各工作类型周工时相加", "notes": "口径为近似周工时"},
        {"final_variable": "work_days_yearly", "source_file": "2018/2020 Work_Retirement.dta", "raw_variables": "2018: fc009*fc010 + fe001*4.345*fe002 + fh001*4.345*fh002; 2020: max(ff001, fb005*fb006 + fc025*4.345*fc026 + fd007*4.345*fd008)", "construct_rule": "按月数*周数*天数或直接总天数近似年工作天数", "notes": "近似构造"},
        {"final_variable": "srh", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "srh", "construct_rule": "直接保留", "notes": "1=很差, 5=很好"},
        {"final_variable": "poor_health", "source_file": "derived", "raw_variables": "srh", "construct_rule": "srh<=2 记 1，否则记 0", "notes": ""},
        {"final_variable": "chronic_count", "source_file": "derived", "raw_variables": ", ".join(DISEASE_VARS), "construct_rule": "14 个具体疾病 0/1 指标求和", "notes": ""},
        {"final_variable": "adl_limit", "source_file": "derived", "raw_variables": "adlab_c", "construct_rule": "adlab_c>0 记 1，否则 0", "notes": ""},
        {"final_variable": "iadl_limit", "source_file": "derived", "raw_variables": "iadl", "construct_rule": "iadl>0 记 1，否则 0", "notes": ""},
        {"final_variable": "depression_high", "source_file": "derived", "raw_variables": "cesd10", "construct_rule": "CESD10>=12 记 1，否则 0", "notes": "常见筛分阈值"},
        {"final_variable": "care_elder_or_disabled", "source_file": "整理完的-charls数据/CHARLS.csv + 2018 Family_Information.dta", "raw_variables": "social7; parent_need_help_flag_2018", "construct_rule": "以 social7 为主，若 2018 家庭模块显示同住/亲代失去自理能力则补充为 1", "notes": "更偏照料责任代理变量"},
        {"final_variable": "care_grandchild", "source_file": "candidate only", "raw_variables": "2018/2020 Family_Information 候选变量", "construct_rule": "本版未发现两期稳定可比且明确表示祖辈照看孙辈的变量，因此保留缺失", "notes": "已在 family 候选表中保留候选来源"},
        {"final_variable": "co_reside_child", "source_file": "2018 Family_Information.dta", "raw_variables": "cb053_*", "construct_rule": "任一子女居住地编码为 1 或 2 记 1；若有观测且均不为 1/2 记 0", "notes": "2020 同口径变量未明确识别，本版置缺失"},
        {"final_variable": "co_reside_parent", "source_file": "2018 Family_Information.dta", "raw_variables": "ca016_*", "construct_rule": "任一父母/公婆居住地编码为 1 记 1；若有观测且均不为 1 记 0", "notes": "2020 同口径变量未明确识别，本版置缺失"},
        {"final_variable": "grandchild_number", "source_file": "2018 Family_Information.dta", "raw_variables": "cb067_*", "construct_rule": "对子女对应孙辈数直接求和", "notes": "2020 同口径变量未明确识别，本版置缺失"},
        {"final_variable": "spouse_health", "source_file": "candidate only", "raw_variables": "无稳定直接来源", "construct_rule": "本版保留缺失", "notes": "后续如需精确构造，建议建立配偶级联结"},
        {"final_variable": "intergen_support_in", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "fcamt", "construct_rule": "直接映射为子女对父母经济支持流入", "notes": ""},
        {"final_variable": "intergen_support_out", "source_file": "整理完的-charls数据/CHARLS.csv", "raw_variables": "tcamt", "construct_rule": "直接映射为父母对子女经济支持流出", "notes": ""},
        {"final_variable": "family_care_index", "source_file": "derived", "raw_variables": "care_grandchild, care_elder_or_disabled, family_size, hchild, intergen_support_out", "construct_rule": "按题设规则逐项加总；中位数阈值按 wave 分年计算", "notes": "若部分分项缺失，则按可观测分项加总"},
        {"final_variable": "medical_expense", "source_file": "derived", "raw_variables": "oophos1y, oopdoc1m", "construct_rule": "住院自付费用 + 12*门诊月自付费用", "notes": "将月度门诊费用年化，属近似口径"},
        {"final_variable": "medical_burden", "source_file": "derived", "raw_variables": "medical_expense, income_total", "construct_rule": "medical_expense / income_total", "notes": "仅在 income_total>0 时计算"},
        {"final_variable": "log_income_total", "source_file": "derived", "raw_variables": "income_total", "construct_rule": "log1p(income_total)", "notes": ""},
        {"final_variable": "log_hhcperc", "source_file": "derived", "raw_variables": "hhcperc", "construct_rule": "log1p(hhcperc)", "notes": ""},
        {"final_variable": "economic_pressure_index", "source_file": "derived", "raw_variables": "pension, hhcperc, medical_expense, intergen_support_in, intergen_support_out", "construct_rule": "无养老金、低人均消费、高医疗自付、高代际流入、高代际流出各记 1", "notes": "中位数阈值按 wave 分年计算"},
        {"final_variable": "female", "source_file": "整理完的-charls数据/charls.dta labels", "raw_variables": "gender", "construct_rule": "根据值标签确认 gender=0 为女性，因此 gender==0 记 1", "notes": ""},
        {"final_variable": "married", "source_file": "整理完的-charls数据/charls.dta labels", "raw_variables": "marry", "construct_rule": "根据值标签确认 marry=1 为已婚，因此 marry==1 记 1", "notes": ""},
        {"final_variable": "age_group", "source_file": "derived", "raw_variables": "age", "construct_rule": "按 45-49, 50-54, 55-59, 60-64, 65-69, 70+ 分组", "notes": ""},
        {"final_variable": "suitability_group", "source_file": "derived", "raw_variables": "labor_participation, poor_health, chronic_count, adl_limit, iadl_limit, depression_high, family_care_index, economic_pressure_index, age, hhcperc, pension, medical_burden", "construct_rule": "按题设 A-E 规则分组", "notes": "无法满足判定条件者记 unknown"},
    ]
    return pd.DataFrame(rows)


def add_derived_variables(df: pd.DataFrame, base_labels: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "gender",
        "marry",
        "rural",
        "rural2",
        "retire",
        "pension",
        "ins",
        "hospital",
        "doctor",
        "exercise",
        "disability",
        "social7",
    ] + DISEASE_VARS:
        out[col] = safe_numeric(out[col])

    out["female"] = np.where(out["gender"] == 0, 1.0, np.where(out["gender"].notna(), 0.0, np.nan))
    out["married"] = np.where(out["marry"] == 1, 1.0, np.where(out["marry"].notna(), 0.0, np.nan))
    out["poor_health"] = np.where(out["srh"].le(2), 1.0, np.where(out["srh"].notna(), 0.0, np.nan))
    out["chronic_count"] = out[DISEASE_VARS].fillna(0).sum(axis=1)
    out.loc[out[DISEASE_VARS].notna().sum(axis=1) == 0, "chronic_count"] = np.nan
    out["adl_limit"] = np.where(out["adlab_c"].gt(0), 1.0, np.where(out["adlab_c"].notna(), 0.0, np.nan))
    out["iadl_limit"] = np.where(out["iadl"].gt(0), 1.0, np.where(out["iadl"].notna(), 0.0, np.nan))
    out["depression_high"] = np.where(out["cesd10"].ge(12), 1.0, np.where(out["cesd10"].notna(), 0.0, np.nan))

    out["intergen_support_in"] = out["fcamt"]
    out["intergen_support_out"] = out["tcamt"]

    care_proxy = np.where(out["social7"] == 1, 1.0, np.where(out["social7"].notna(), 0.0, np.nan))
    if "parent_need_help_flag_2018" in out.columns:
        parent_proxy = out["parent_need_help_flag_2018"]
        care_proxy = np.where(
            (pd.Series(care_proxy, index=out.index) == 1) | (parent_proxy == 1),
            1.0,
            np.where(
                pd.Series(care_proxy, index=out.index).notna() | parent_proxy.notna(),
                0.0,
                np.nan,
            ),
        )
    out["care_elder_or_disabled"] = care_proxy

    out["care_grandchild"] = np.nan
    out["spouse_health"] = np.nan

    out["medical_expense"] = out["oophos1y"].fillna(0) + out["oopdoc1m"].fillna(0) * 12
    both_missing = out["oophos1y"].isna() & out["oopdoc1m"].isna()
    out.loc[both_missing, "medical_expense"] = np.nan
    out["medical_burden"] = out["medical_expense"] / out["income_total"]
    out.loc[~out["income_total"].gt(0), "medical_burden"] = np.nan
    out["log_income_total"] = np.log1p(out["income_total"].clip(lower=0))
    out["log_hhcperc"] = np.log1p(out["hhcperc"].clip(lower=0))
    out.loc[out["income_total"].isna(), "log_income_total"] = np.nan
    out.loc[out["hhcperc"].isna(), "log_hhcperc"] = np.nan

    out["age_group"] = pd.cut(
        out["age"],
        bins=[45, 50, 55, 60, 65, 70, np.inf],
        right=False,
        labels=["45-49", "50-54", "55-59", "60-64", "65-69", "70+"],
    )

    for year, year_df in out.groupby("year"):
        idx = year_df.index
        family_size_median = year_df["family_size"].median(skipna=True)
        hchild_median = year_df["hchild"].median(skipna=True)
        support_out_median = year_df["intergen_support_out"].median(skipna=True)
        hhcperc_median = year_df["hhcperc"].median(skipna=True)
        medical_median = year_df["medical_expense"].median(skipna=True)
        support_in_median = year_df["intergen_support_in"].median(skipna=True)

        family_components = pd.DataFrame(index=idx)
        family_components["care_grandchild"] = out.loc[idx, "care_grandchild"]
        family_components["care_elder_or_disabled"] = out.loc[idx, "care_elder_or_disabled"]
        family_components["family_size_high"] = np.where(
            out.loc[idx, "family_size"].gt(family_size_median),
            1.0,
            np.where(out.loc[idx, "family_size"].notna(), 0.0, np.nan),
        )
        family_components["hchild_high"] = np.where(
            out.loc[idx, "hchild"].gt(hchild_median),
            1.0,
            np.where(out.loc[idx, "hchild"].notna(), 0.0, np.nan),
        )
        family_components["support_out_high"] = np.where(
            out.loc[idx, "intergen_support_out"].gt(support_out_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_out"].notna(), 0.0, np.nan),
        )
        out.loc[idx, "family_care_index"] = family_components.sum(axis=1, min_count=1)

        econ_components = pd.DataFrame(index=idx)
        econ_components["no_pension"] = np.where(
            out.loc[idx, "pension"] == 0,
            1.0,
            np.where(out.loc[idx, "pension"].notna(), 0.0, np.nan),
        )
        econ_components["low_hhcperc"] = np.where(
            out.loc[idx, "hhcperc"].lt(hhcperc_median),
            1.0,
            np.where(out.loc[idx, "hhcperc"].notna(), 0.0, np.nan),
        )
        econ_components["high_medical"] = np.where(
            out.loc[idx, "medical_expense"].gt(medical_median),
            1.0,
            np.where(out.loc[idx, "medical_expense"].notna(), 0.0, np.nan),
        )
        econ_components["high_support_in"] = np.where(
            out.loc[idx, "intergen_support_in"].gt(support_in_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_in"].notna(), 0.0, np.nan),
        )
        econ_components["high_support_out"] = np.where(
            out.loc[idx, "intergen_support_out"].gt(support_out_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_out"].notna(), 0.0, np.nan),
        )
        out.loc[idx, "economic_pressure_index"] = econ_components.sum(axis=1, min_count=1)

    out["health_constraint_flag"] = np.where(
        (
            (out["poor_health"] == 1)
            | (out["adl_limit"] == 1)
            | (out["iadl_limit"] == 1)
            | (out["depression_high"] == 1)
            | (out["chronic_count"] >= 2)
        ),
        1.0,
        np.where(
            out[["poor_health", "adl_limit", "iadl_limit", "depression_high", "chronic_count"]]
            .notna()
            .any(axis=1),
            0.0,
            np.nan,
        ),
    )

    out["family_care_high"] = np.where(
        out["family_care_index"].ge(2),
        1.0,
        np.where(out["family_care_index"].notna(), 0.0, np.nan),
    )
    out["economic_pressure_high"] = np.where(
        out["economic_pressure_index"].ge(2),
        1.0,
        np.where(out["economic_pressure_index"].notna(), 0.0, np.nan),
    )
    out["health_good_flag"] = np.where(
        (
            (out["poor_health"] == 0)
            & (out["adl_limit"] == 0)
            & (out["iadl_limit"] == 0)
            & ((out["chronic_count"] <= 1) | out["chronic_count"].isna())
            & ((out["depression_high"] == 0) | out["depression_high"].isna())
        ),
        1.0,
        np.where(
            out[["poor_health", "adl_limit", "iadl_limit", "chronic_count", "depression_high"]]
            .notna()
            .any(axis=1),
            0.0,
            np.nan,
        ),
    )

    out["suitability_group"] = "unknown"
    out.loc[
        (out["health_constraint_flag"] == 1)
        & (
            (out["family_care_high"] == 1)
            | (out["hhcperc"].notna() & out.groupby("year")["hhcperc"].transform("median").gt(out["hhcperc"]))
            | (out["pension"] == 0)
            | (out["medical_burden"].gt(0.2))
        ),
        "suitability_group",
    ] = "E"
    out.loc[
        (out["labor_participation"] == 1)
        & (out["economic_pressure_high"] == 1),
        "suitability_group",
    ] = "B"
    out.loc[
        (out["labor_participation"] == 1)
        & (out["health_good_flag"] == 1)
        & (out["family_care_high"] == 0)
        & (out["economic_pressure_high"] == 0),
        "suitability_group",
    ] = "A"
    out.loc[
        (out["labor_participation"] == 0)
        & ((out["health_constraint_flag"] == 1) | (out["family_care_high"] == 1)),
        "suitability_group",
    ] = "C"
    out.loc[
        (out["labor_participation"] == 0)
        & (out["health_good_flag"] == 1)
        & (out["family_care_high"] == 0)
        & (out["age"] < 60),
        "suitability_group",
    ] = "D"
    out.loc[out["labor_participation"].isna(), "suitability_group"] = "unknown"

    return out


def build_final_column_order() -> list[str]:
    return [
        "ID",
        "householdID",
        "communityID",
        "wave",
        "year",
        "province",
        "city",
        "labor_participation",
        "labor_type",
        "farm_work_flag",
        "nonfarm_work_flag",
        "employed_flag",
        "self_employed_flag",
        "side_job_flag",
        "retired_flag_raw",
        "work_hours_weekly",
        "work_days_yearly",
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
        *DISEASE_VARS,
        "hchild",
        "family_size",
        "fcamt",
        "tcamt",
        "intergen_support_in",
        "intergen_support_out",
        "care_grandchild",
        "care_elder_or_disabled",
        "co_reside_child",
        "co_reside_parent",
        "spouse_health",
        "grandchild_number",
        "family_care_index",
        "income_total",
        "hhcperc",
        "log_income_total",
        "log_hhcperc",
        "pension",
        "ins",
        "medical_expense",
        "medical_burden",
        "oophos1y",
        "tothos1y",
        "oopdoc1m",
        "totdoc1m",
        "hospital",
        "doctor",
        "economic_pressure_index",
        "age",
        "age_group",
        "gender",
        "female",
        "marry",
        "married",
        "rural",
        "rural2",
        "edu",
        "nation",
        "drinkev",
        "drinkl",
        "smokev",
        "smoken",
        "exercise",
        "totmet",
        "suitability_group",
        "HH_weight",
        "INDV_weight",
        "died",
        "crosssection",
        "iyear",
        "imonth",
    ]


def create_report_sheets(full_df: pd.DataFrame, model_df: pd.DataFrame, variable_source: pd.DataFrame, duplicates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sample_flow = pd.DataFrame(
        [
            {"step": "wave in [4,5] from base", "n": len(full_df)},
            {"step": "full sample: age>=45 and ID+wave non-missing", "n": len(full_df)},
            {"step": "model sample", "n": len(model_df)},
        ]
    )

    wave_distribution = (
        full_df.groupby(["wave", "year"]).size().rename("n").reset_index().sort_values(["wave"])
    )
    labor_distribution = (
        full_df.groupby(["wave", "labor_participation"]).size().rename("n").reset_index()
    )

    labor_by_age_gender_rural = (
        full_df.groupby(["year", "age_group", "female", "rural"], observed=False)["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_health = (
        full_df.groupby(["year", "poor_health", "adl_limit", "iadl_limit", "depression_high"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_family_care = (
        full_df.groupby(["year", "care_elder_or_disabled", "co_reside_child", "family_care_index"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_economic_pressure = (
        full_df.groupby(["year", "pension", "economic_pressure_index"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )

    retire_labor = (
        full_df.groupby(["year", "retired_flag_raw", "labor_participation"]).size().rename("n").reset_index()
    )

    missing_rate = (
        full_df.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "variable"})
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )

    duplicate_check = pd.DataFrame(
        [
            {"metric": "duplicate_rows", "value": len(duplicates)},
            {"metric": "has_duplicate_id_wave", "value": "yes" if len(duplicates) > 0 else "no"},
        ]
    )

    age_check = (
        full_df.groupby("year")["age"].agg(["count", "min", "median", "max", "mean"]).reset_index()
    )
    suitability_dist = (
        full_df.groupby(["year", "suitability_group"]).size().rename("n").reset_index()
    )

    return {
        "sample_flow": sample_flow,
        "wave_distribution": wave_distribution,
        "labor_distribution": labor_distribution,
        "labor_by_age_gender_rural": labor_by_age_gender_rural,
        "labor_by_health": labor_by_health,
        "labor_by_family_care": labor_by_family_care,
        "labor_by_economic_pressure": labor_by_economic_pressure,
        "retire_labor_crosstab": retire_labor,
        "missing_rate": missing_rate,
        "variable_source": variable_source,
        "duplicate_check": duplicate_check,
        "age_check": age_check,
        "suitability_group_distribution": suitability_dist,
    }


def write_dta(df: pd.DataFrame, path: Path) -> None:
    export_df = df.copy()
    for col in export_df.columns:
        if isinstance(export_df[col].dtype, pd.CategoricalDtype):
            export_df[col] = export_df[col].astype(str).replace("nan", np.nan)
    pyreadstat.write_dta(export_df, str(path))


def print_summary(full_df: pd.DataFrame, model_df: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    wave_counts = full_df.groupby("year").size()
    labor_rates = full_df.groupby("year")["labor_participation"].mean()
    labor_missing = full_df["labor_participation"].isna().mean()
    health_missing = full_df[HEALTH_CORE_VARS].isna().all(axis=1).mean()
    family_missing = full_df[FAMILY_CORE_VARS].isna().all(axis=1).mean()
    econ_missing = full_df[ECON_CORE_VARS].isna().all(axis=1).mean()
    suitability = full_df["suitability_group"].value_counts(dropna=False)

    print(f"Full sample n: {len(full_df)}")
    print(f"Model sample n: {len(model_df)}")
    for year in [2018, 2020]:
        print(f"{year} sample n: {int(wave_counts.get(year, 0))}")
    for year in [2018, 2020]:
        value = labor_rates.get(year)
        print(f"{year} labor participation rate: {value:.4f}" if pd.notna(value) else f"{year} labor participation rate: nan")
    print(f"labor_participation missing rate: {labor_missing:.4f}")
    print(f"Health core all-missing rate: {health_missing:.4f}")
    print(f"Family core all-missing rate: {family_missing:.4f}")
    print(f"Economic core all-missing rate: {econ_missing:.4f}")
    print("suitability_group distribution:")
    for key, value in suitability.items():
        print(f"  {key}: {value}")
    print(f"Has duplicate ID+wave: {'yes' if len(duplicates) > 0 else 'no'}")
    print("2015 extension notes:")
    print("  1. 劳动模块口径更接近 2018，但变量后缀和分支逻辑与 2020 不同，不能直接复用 2020 的 xworking/ff001 规则。")
    print("  2. 家庭信息模块的子女、父母与同住变量命名体系更接近 2018，建议优先按 roster 类题目重新定位。")
    print("  3. 2020 含疫情背景题，2015 没有，对‘劳动受疫情影响’类变量不能回推。")
    print("  4. 如果扩到 2015，建议先单独做一轮 metadata scan，再统一三期劳动参与与照料责任口径。")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_labels = load_base_labels()

    labor_candidates = scan_module_candidates(LABOR_KEYWORDS, LABOR_CANDIDATES_XLSX)
    family_candidates = scan_module_candidates(FAMILY_KEYWORDS, FAMILY_CANDIDATES_XLSX)
    econ_candidates = scan_module_candidates(ECON_KEYWORDS, ECON_CANDIDATES_XLSX)
    health_candidates = scan_module_candidates(HEALTH_KEYWORDS, HEALTH_CANDIDATES_XLSX)

    base = normalize_keys(read_base_sample(), ["ID", "householdID", "communityID"])

    duplicate_mask = base.duplicated(subset=["ID", "wave"], keep=False)
    duplicates = base.loc[duplicate_mask, ["ID", "wave", "year", "householdID", "communityID"]].sort_values(["ID", "wave"])
    duplicates.to_csv(DUPLICATE_CSV, index=False)

    work = pd.concat([read_work_2018(), read_work_2020()], ignore_index=True)
    family2018 = read_family_2018()
    family = family2018.copy()

    weights_sample = pd.concat([read_weights_sample(2018), read_weights_sample(2020)], ignore_index=True)

    full = base.merge(
        work,
        on=["ID", "householdID", "communityID", "wave"],
        how="left",
        suffixes=("", "_work"),
    )
    full = full.merge(
        family,
        on=["ID", "householdID", "communityID", "wave"],
        how="left",
        suffixes=("", "_family"),
    )
    full = full.merge(
        weights_sample,
        on=["ID", "householdID", "communityID", "wave"],
        how="left",
        suffixes=("", "_ws"),
    )

    full["wave"] = safe_numeric(full["wave"]).astype("Int64")
    full = add_derived_variables(full, base_labels)
    full["year"] = safe_numeric(full["wave"]).map(WAVE_YEAR_MAP)

    final_cols = [col for col in build_final_column_order() if col in full.columns]
    full = full[final_cols].copy()

    model_mask = (
        full["labor_participation"].notna()
        & full[HEALTH_CORE_VARS].notna().any(axis=1)
        & full[FAMILY_CORE_VARS].notna().any(axis=1)
        & full[ECON_CORE_VARS].notna().any(axis=1)
        & ~full.duplicated(subset=["ID", "wave"], keep=False)
    )
    model = full.loc[model_mask].copy()

    variable_source = build_variable_source_table()
    variable_source.to_csv(VARIABLE_SOURCE_CSV, index=False, encoding="utf-8-sig")

    report_sheets = create_report_sheets(full, model, variable_source, duplicates)
    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        for sheet_name, df in report_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    full.to_csv(FULL_CSV, index=False, encoding="utf-8-sig")
    model.to_csv(MODEL_CSV, index=False, encoding="utf-8-sig")
    write_dta(full, FULL_DTA)
    write_dta(model, MODEL_DTA)

    print_summary(full, model, duplicates)


if __name__ == "__main__":
    main()
