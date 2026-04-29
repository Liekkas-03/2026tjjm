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

V0_FULL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_full.csv"
V0_MODEL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v0_model.csv"

V1_FULL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_full.csv"
V1_MODEL = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_model.csv"
V1_MODEL_AGE90 = BASE_DIR / "CHARLS_labor_panel_2018_2020_v1_model_age90.csv"
V1_REPORT = BASE_DIR / "CHARLS_labor_cleaning_report_2018_2020_v1.xlsx"

FAMILY_MAPPING_XLSX = OUTPUT_DIR / "family_care_mapping_v1.xlsx"
FAMILY_DISTRIBUTION_XLSX = OUTPUT_DIR / "family_care_distribution_v1.xlsx"
ECON_CHECK_XLSX = OUTPUT_DIR / "economic_pressure_check_v1.xlsx"
ANOMALY_CSV = OUTPUT_DIR / "labor_logic_anomalies_v1.csv"

MODULE_PATHS = {
    2018: {
        "Family_Information": RAW_DIR / "2018" / "CHARLS2018r" / "Family_Information.dta",
        "Demographic_Background": RAW_DIR / "2018" / "CHARLS2018r" / "Demographic_Background.dta",
        "Health_Status_and_Functioning": RAW_DIR / "2018" / "CHARLS2018r" / "Health_Status_and_Functioning.dta",
        "Work_Retirement": RAW_DIR / "2018" / "CHARLS2018r" / "Work_Retirement.dta",
        "Household_Income": RAW_DIR / "2018" / "CHARLS2018r" / "Household_Income.dta",
        "Individual_Income": RAW_DIR / "2018" / "CHARLS2018r" / "Individual_Income.dta",
    },
    2020: {
        "Family_Information": RAW_DIR / "2020" / "CHARLS2020r" / "Family_Information.dta",
        "Demographic_Background": RAW_DIR / "2020" / "CHARLS2020r" / "Demographic_Background.dta",
        "Health_Status_and_Functioning": RAW_DIR / "2020" / "CHARLS2020r" / "Health_Status_and_Functioning.dta",
        "Work_Retirement": RAW_DIR / "2020" / "CHARLS2020r" / "Work_Retirement.dta",
        "Household_Income": RAW_DIR / "2020" / "CHARLS2020r" / "Household_Income.dta",
        "Individual_Income": RAW_DIR / "2020" / "CHARLS2020r" / "Individual_Income.dta",
    },
}

FAMILY_SEARCH_KEYWORDS = [
    "grandchild",
    "grandchildren",
    "grandson",
    "granddaughter",
    "care",
    "take care",
    "look after",
    "child care",
    "help child",
    "disabled",
    "parent",
    "spouse",
    "live with",
    "living regularly",
    "health status",
    "under 18",
    "raise",
]

FAMILY_CORE_V1 = [
    "hchild",
    "family_size",
    "intergen_support_out",
    "care_elder_or_disabled",
    "co_reside_child",
]
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
ECON_CORE_V1 = [
    "pension",
    "income_total",
    "hhcperc",
    "intergen_support_in",
    "intergen_support_out",
    "medical_expense",
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


def set_special_missing(series: pd.Series) -> pd.Series:
    numeric = safe_numeric(series)
    return numeric.mask(numeric.isin([993, 995, 997, 999]))


def clip_range(series: pd.Series, low: float, high: float) -> pd.Series:
    numeric = safe_numeric(series)
    return numeric.where(numeric.between(low, high))


def summarize_value_labels(labels: dict) -> str:
    if not labels:
        return ""
    pairs = [f"{k}={v}" for k, v in list(labels.items())[:8]]
    summary = "; ".join(pairs)
    if len(labels) > 8:
        summary += f"; ... ({len(labels)} values)"
    return summary


def scan_family_candidates() -> pd.DataFrame:
    rows: list[dict] = []
    keywords = [k.lower() for k in FAMILY_SEARCH_KEYWORDS]
    for year, modules in MODULE_PATHS.items():
        for module_name, module_path in modules.items():
            meta = pyreadstat.read_dta(module_path, metadataonly=True)[1]
            label_map = meta.column_names_to_labels or {}
            value_label_map = meta.variable_to_label or {}
            value_labels = meta.value_labels or {}
            for variable in meta.column_names:
                label = label_map.get(variable, "") or ""
                haystack = f"{variable} {label}".lower()
                matched = [kw for kw in keywords if kw in haystack]
                if not matched:
                    continue
                labelset = value_label_map.get(variable, "")
                rows.append(
                    {
                        "year": year,
                        "module": module_name,
                        "source_file": str(module_path.relative_to(ROOT)),
                        "variable_name": variable,
                        "variable_label": label,
                        "matched_keywords": ", ".join(matched),
                        "value_label_summary": summarize_value_labels(value_labels.get(labelset, {})),
                    }
                )
    return pd.DataFrame(rows).sort_values(["year", "module", "variable_name"], ignore_index=True)


def load_v0_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(V0_FULL)
    model = pd.read_csv(V0_MODEL)
    full = normalize_keys(full, ["ID", "householdID", "communityID"])
    model = normalize_keys(model, ["ID", "householdID", "communityID"])
    full["wave"] = safe_numeric(full["wave"]).astype("Int64")
    model["wave"] = safe_numeric(model["wave"]).astype("Int64")
    full["year"] = safe_numeric(full["year"]).astype("Int64")
    model["year"] = safe_numeric(model["year"]).astype("Int64")
    return full, model


def build_family_2018() -> pd.DataFrame:
    path = MODULE_PATHS[2018]["Family_Information"]
    child_loc_cols = [f"cb053_{i}_" for i in range(1, 16)]
    parent_loc_cols = [f"ca016_{i}_" for i in range(1, 9)]
    parent_selfcare_cols = [f"ca026_w3_{i}_" for i in range(1, 9)]
    grandchild_cols = [f"cb067_{i}_" for i in range(1, 16)]
    grandchild_u16_cols = [f"cb068_{i}_" for i in range(1, 14)]
    usecols = ["ID", "householdID", "communityID"] + child_loc_cols + parent_loc_cols + parent_selfcare_cols + grandchild_cols + grandchild_u16_cols
    df, _ = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)
    df = normalize_keys(df, ["ID", "householdID", "communityID"])

    for col in child_loc_cols + parent_loc_cols + parent_selfcare_cols:
        df[col] = set_special_missing(df[col])
    for col in grandchild_cols + grandchild_u16_cols:
        df[col] = clip_range(df[col], 0, 99)

    child_reside = df[child_loc_cols].isin([1, 2])
    parent_reside = df[parent_loc_cols].isin([1])
    parent_need_help = df[parent_selfcare_cols].isin([2])

    out = df[["ID", "householdID", "communityID"]].copy()
    out["wave"] = pd.Series([4] * len(out), dtype="Int64")
    out["co_reside_child_v1"] = np.where(
        child_reside.any(axis=1),
        1.0,
        np.where(df[child_loc_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["co_reside_parent_v1"] = np.where(
        parent_reside.any(axis=1),
        1.0,
        np.where(df[parent_loc_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["parent_need_help_flag"] = np.where(
        parent_need_help.any(axis=1),
        1.0,
        np.where(df[parent_selfcare_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["grandchild_number_v1"] = df[grandchild_cols].sum(axis=1, min_count=1)
    out["grandchild_under16_v1"] = df[grandchild_u16_cols].sum(axis=1, min_count=1)
    return out


def build_family_2020() -> pd.DataFrame:
    path = MODULE_PATHS[2020]["Family_Information"]
    child_live_cols = [f"ca014_{i}_" for i in range(1, 18)]
    child_health_cols = [f"ca013_{i}_" for i in range(1, 18)]
    child_in_cols = [f"ca017_1_{i}_" for i in range(1, 18)]
    child_out_cols = [f"ca018_1_{i}_" for i in range(1, 18)]
    usecols = ["householdID", "communityID"] + child_live_cols + child_health_cols + child_in_cols + child_out_cols + ["cc001", "cc001_1"]
    df, _ = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)
    df = normalize_keys(df, ["householdID", "communityID"])

    for col in child_live_cols + child_health_cols + child_in_cols + child_out_cols + ["cc001", "cc001_1"]:
        df[col] = set_special_missing(df[col])
    for col in child_live_cols:
        df[col] = clip_range(df[col], 0, 12)

    child_reside = df[child_live_cols].gt(0)
    out = df[["householdID", "communityID"]].copy()
    out["wave"] = pd.Series([5] * len(out), dtype="Int64")
    out["co_reside_child_v1"] = np.where(
        child_reside.any(axis=1),
        1.0,
        np.where(df[child_live_cols].notna().any(axis=1), 0.0, np.nan),
    )
    out["co_reside_parent_v1"] = np.nan
    out["parent_need_help_flag"] = np.nan
    out["grandchild_number_v1"] = np.nan
    out["grandchild_under16_v1"] = np.nan
    out["care_grandchild_candidate_2020"] = np.where(
        df["cc001"] == 1,
        1.0,
        np.where(df["cc001"].notna(), 0.0, np.nan),
    )
    return out


def build_family_mapping(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "final_variable": "care_grandchild",
            "status": "candidate_only",
            "preferred_source": "2018 Family_Information: cb067_*, cb068_*; 2020 Family_Information: cc001, cc001_1",
            "rule_or_reason": "2018 可得到孙辈数量与16岁以下孙辈数量，但没有两期稳定的“实际照料孙辈”直接题；2020 仅有疫情期间子孙未回家题，不足以等同照料责任，因此 v1 主变量保留缺失",
            "usable_in_main_index": "no",
        },
        {
            "final_variable": "care_elder_or_disabled",
            "status": "constructed_v1",
            "preferred_source": "v0 base social7 + 2018 Family_Information ca026_w3_* + ca016_*",
            "rule_or_reason": "social7=1 记照料老人/病人/残疾人；2018 若同住父母/公婆且其无自理能力，也补记 1",
            "usable_in_main_index": "yes",
        },
        {
            "final_variable": "co_reside_child",
            "status": "constructed_v1",
            "preferred_source": "2018 Family_Information cb053_*; 2020 Family_Information ca014_*",
            "rule_or_reason": "2018 按每个子女是否与受访者同住/同院；2020 按与每个子女同住时长是否大于 0 构造",
            "usable_in_main_index": "yes",
        },
        {
            "final_variable": "co_reside_parent",
            "status": "partially_constructed",
            "preferred_source": "2018 Family_Information ca016_*",
            "rule_or_reason": "2018 可按父母/公婆居住地直接构造；2020 未找到稳定同口径来源，因此 2020 保留缺失",
            "usable_in_main_index": "supplement_only",
        },
        {
            "final_variable": "grandchild_number",
            "status": "partially_constructed",
            "preferred_source": "2018 Family_Information cb067_*",
            "rule_or_reason": "2018 可对子女对应孙辈数求和；2020 未发现可比题项，因此仅作补充变量",
            "usable_in_main_index": "supplement_only",
        },
        {
            "final_variable": "spouse_health",
            "status": "candidate_only",
            "preferred_source": "2020 Family_Information ca013_* (children health only); Demographic_Background spouse living items",
            "rule_or_reason": "当前选定模块中未找到可稳定匹配到受访者配偶个体健康的标识链，v1 不强行构造",
            "usable_in_main_index": "no",
        },
        {
            "final_variable": "family_care_index_v1",
            "status": "constructed_v1",
            "preferred_source": "hchild, family_size, intergen_support_out, care_elder_or_disabled, co_reside_child",
            "rule_or_reason": "按题设优先使用两期都相对稳定的五项指标构造，不纳入 care_grandchild",
            "usable_in_main_index": "yes",
        },
    ]
    mapping = pd.DataFrame(rows)
    with pd.ExcelWriter(FAMILY_MAPPING_XLSX, engine="openpyxl") as writer:
        mapping.to_excel(writer, index=False, sheet_name="mapping")
        candidates.to_excel(writer, index=False, sheet_name="raw_candidates")
    return mapping


def merge_family_v1(full: pd.DataFrame) -> pd.DataFrame:
    out = full.copy()
    family2018 = build_family_2018()
    family2020 = build_family_2020()

    out = out.merge(
        family2018,
        on=["ID", "householdID", "communityID", "wave"],
        how="left",
        suffixes=("", "_2018f"),
    )
    out = out.merge(
        family2020,
        on=["householdID", "communityID", "wave"],
        how="left",
        suffixes=("", "_2020f"),
    )

    out["co_reside_child"] = out["co_reside_child_v1"].combine_first(out["co_reside_child_v1_2020f"])
    out["co_reside_parent"] = out["co_reside_parent_v1"].combine_first(out["co_reside_parent_v1_2020f"])
    out["grandchild_number"] = out["grandchild_number_v1"].combine_first(out["grandchild_number_v1_2020f"])
    out["grandchild_under16_v1"] = out["grandchild_under16_v1"].combine_first(out["grandchild_under16_v1_2020f"])

    existing_care = safe_numeric(out["care_elder_or_disabled"]) if "care_elder_or_disabled" in out.columns else pd.Series(np.nan, index=out.index)
    care_proxy = np.where(
        (existing_care == 1) | (safe_numeric(out["parent_need_help_flag"]) == 1),
        1.0,
        np.where(
            existing_care.notna() | safe_numeric(out["parent_need_help_flag"]).notna(),
            0.0,
            np.nan,
        ),
    )
    out["care_elder_or_disabled"] = care_proxy
    out["care_grandchild"] = np.nan
    out["spouse_health"] = np.nan

    for year, idx in out.groupby("year").groups.items():
        support_out_median = out.loc[idx, "intergen_support_out"].median(skipna=True)
        family_size_median = out.loc[idx, "family_size"].median(skipna=True)
        hchild_median = out.loc[idx, "hchild"].median(skipna=True)

        comps = pd.DataFrame(index=idx)
        comps["care_elder_or_disabled"] = np.where(
            out.loc[idx, "care_elder_or_disabled"] == 1,
            1.0,
            np.where(out.loc[idx, "care_elder_or_disabled"].notna(), 0.0, np.nan),
        )
        comps["co_reside_child"] = np.where(
            out.loc[idx, "co_reside_child"] == 1,
            1.0,
            np.where(out.loc[idx, "co_reside_child"].notna(), 0.0, np.nan),
        )
        comps["family_size_high"] = np.where(
            out.loc[idx, "family_size"].gt(family_size_median),
            1.0,
            np.where(out.loc[idx, "family_size"].notna(), 0.0, np.nan),
        )
        comps["hchild_high"] = np.where(
            out.loc[idx, "hchild"].gt(hchild_median),
            1.0,
            np.where(out.loc[idx, "hchild"].notna(), 0.0, np.nan),
        )
        comps["support_out_high"] = np.where(
            out.loc[idx, "intergen_support_out"].gt(support_out_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_out"].notna(), 0.0, np.nan),
        )
        out.loc[idx, "family_care_index_v1"] = comps.sum(axis=1, min_count=1)
        out.loc[idx, "family_care_index_v1_high"] = np.where(
            out.loc[idx, "family_care_index_v1"].ge(2),
            1.0,
            np.where(out.loc[idx, "family_care_index_v1"].notna(), 0.0, np.nan),
        )

    drop_cols = [c for c in out.columns if c.endswith("_2020f") or c in {"co_reside_child_v1", "co_reside_parent_v1", "grandchild_number_v1", "parent_need_help_flag"}]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def add_economic_v1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["income_total", "hhcperc", "intergen_support_in", "intergen_support_out", "medical_expense", "pension"]:
        out[col] = safe_numeric(out[col])

    out["income_total_nonnegative"] = out["income_total"].clip(lower=0)
    out["hhcperc_nonnegative"] = out["hhcperc"].clip(lower=0)
    out.loc[out["income_total"].isna(), "income_total_nonnegative"] = np.nan
    out.loc[out["hhcperc"].isna(), "hhcperc_nonnegative"] = np.nan
    out["log_income_total_v1"] = np.log1p(out["income_total_nonnegative"])
    out["log_hhcperc_v1"] = np.log1p(out["hhcperc_nonnegative"])
    out.loc[out["income_total_nonnegative"].isna(), "log_income_total_v1"] = np.nan
    out.loc[out["hhcperc_nonnegative"].isna(), "log_hhcperc_v1"] = np.nan

    positive_income = out["income_total_nonnegative"].replace(0, np.nan)
    out["medical_burden_v1"] = out["medical_expense"] / positive_income

    for year, idx in out.groupby("year").groups.items():
        hhcperc_median = out.loc[idx, "hhcperc_nonnegative"].median(skipna=True)
        medical_median = out.loc[idx, "medical_expense"].median(skipna=True)
        support_in_median = out.loc[idx, "intergen_support_in"].median(skipna=True)
        support_out_median = out.loc[idx, "intergen_support_out"].median(skipna=True)

        comps = pd.DataFrame(index=idx)
        comps["no_pension"] = np.where(
            out.loc[idx, "pension"] == 0,
            1.0,
            np.where(out.loc[idx, "pension"].notna(), 0.0, np.nan),
        )
        comps["low_hhcperc"] = np.where(
            out.loc[idx, "hhcperc_nonnegative"].lt(hhcperc_median),
            1.0,
            np.where(out.loc[idx, "hhcperc_nonnegative"].notna(), 0.0, np.nan),
        )
        comps["high_medical_expense"] = np.where(
            out.loc[idx, "medical_expense"].gt(medical_median),
            1.0,
            np.where(out.loc[idx, "medical_expense"].notna(), 0.0, np.nan),
        )
        comps["high_support_in"] = np.where(
            out.loc[idx, "intergen_support_in"].gt(support_in_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_in"].notna(), 0.0, np.nan),
        )
        comps["high_support_out"] = np.where(
            out.loc[idx, "intergen_support_out"].gt(support_out_median),
            1.0,
            np.where(out.loc[idx, "intergen_support_out"].notna(), 0.0, np.nan),
        )
        out.loc[idx, "economic_pressure_index_v1"] = comps.sum(axis=1, min_count=1)
        out.loc[idx, "economic_pressure_index_v1_high"] = np.where(
            out.loc[idx, "economic_pressure_index_v1"].ge(2),
            1.0,
            np.where(out.loc[idx, "economic_pressure_index_v1"].notna(), 0.0, np.nan),
        )
    return out


def build_family_distribution_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "by_year_mean": df.groupby("year")[
            ["care_elder_or_disabled", "co_reside_child", "co_reside_parent", "grandchild_number", "family_care_index_v1"]
        ]
        .mean()
        .reset_index(),
        "by_year_missing": df.groupby("year")[
            ["care_grandchild", "care_elder_or_disabled", "co_reside_child", "co_reside_parent", "grandchild_number", "family_care_index_v1"]
        ]
        .apply(lambda x: x.isna().mean())
        .reset_index(),
        "index_distribution": df.groupby(["year", "family_care_index_v1"]).size().rename("n").reset_index(),
    }


def build_economic_check_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries = []
    for col in [
        "income_total",
        "income_total_nonnegative",
        "hhcperc",
        "hhcperc_nonnegative",
        "intergen_support_in",
        "intergen_support_out",
        "medical_expense",
        "medical_burden",
        "medical_burden_v1",
        "log_income_total",
        "log_income_total_v1",
        "log_hhcperc",
        "log_hhcperc_v1",
    ]:
        s = safe_numeric(df[col])
        summaries.append(
            {
                "variable": col,
                "n": int(s.notna().sum()),
                "neg_count": int((s < 0).sum(skipna=True)),
                "zero_count": int((s == 0).sum(skipna=True)),
                "p1": s.quantile(0.01),
                "p50": s.quantile(0.50),
                "p99": s.quantile(0.99),
                "max": s.max(skipna=True),
            }
        )
    summary_df = pd.DataFrame(summaries)

    pension_labor = (
        df.groupby(["year", "pension", "labor_participation"]).size().rename("n").reset_index()
    )
    econ_labor = (
        df.groupby(["year", "economic_pressure_index_v1", "labor_participation"]).size().rename("n").reset_index()
    )
    extreme_counts = pd.DataFrame(
        [
            {"check": "income_total_negative", "n": int((safe_numeric(df["income_total"]) < 0).sum())},
            {"check": "hhcperc_negative", "n": int((safe_numeric(df["hhcperc"]) < 0).sum())},
            {"check": "income_total_zero", "n": int((safe_numeric(df["income_total"]) == 0).sum())},
            {"check": "hhcperc_zero", "n": int((safe_numeric(df["hhcperc"]) == 0).sum())},
            {"check": "medical_burden_gt_1", "n": int((safe_numeric(df["medical_burden"]) > 1).sum())},
            {"check": "medical_burden_v1_gt_1", "n": int((safe_numeric(df["medical_burden_v1"]) > 1).sum())},
        ]
    )
    top_values = pd.concat(
        [
            df.nlargest(20, "income_total")[["ID", "year", "income_total"]].assign(source="income_total"),
            df.nlargest(20, "hhcperc")[["ID", "year", "hhcperc"]].rename(columns={"hhcperc": "income_total"}).assign(source="hhcperc"),
            df.nlargest(20, "medical_burden_v1")[["ID", "year", "medical_burden_v1"]].rename(columns={"medical_burden_v1": "income_total"}).assign(source="medical_burden_v1"),
        ],
        ignore_index=True,
    )
    return {
        "summary_stats": summary_df,
        "pension_labor_crosstab": pension_labor,
        "economic_pressure_labor_crosstab": econ_labor,
        "extreme_counts": extreme_counts,
        "top_values": top_values,
    }


def build_labor_logic_outputs(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    flag_sum = (
        safe_numeric(out["farm_work_flag"]).fillna(0)
        + safe_numeric(out["nonfarm_work_flag"]).fillna(0)
        + safe_numeric(out["employed_flag"]).fillna(0)
        + safe_numeric(out["self_employed_flag"]).fillna(0)
        + safe_numeric(out["side_job_flag"]).fillna(0)
    )

    anomalies = out.loc[
        (safe_numeric(out["labor_participation"]) == 0) & (safe_numeric(out["work_hours_weekly"]) > 0)
        | (safe_numeric(out["labor_participation"]) == 0) & (safe_numeric(out["work_days_yearly"]) > 0)
        | (safe_numeric(out["labor_participation"]) == 1) & (flag_sum == 0)
        | (safe_numeric(out["age"]) > 100)
        | (safe_numeric(out["work_hours_weekly"]) > 100)
        | (safe_numeric(out["work_days_yearly"]) > 365)
    ].copy()

    anomalies["anomaly_labor0_hours"] = ((safe_numeric(anomalies["labor_participation"]) == 0) & (safe_numeric(anomalies["work_hours_weekly"]) > 0)).astype(int)
    anomalies["anomaly_labor0_days"] = ((safe_numeric(anomalies["labor_participation"]) == 0) & (safe_numeric(anomalies["work_days_yearly"]) > 0)).astype(int)
    anomalies["anomaly_labor1_no_flags"] = ((safe_numeric(anomalies["labor_participation"]) == 1) & ((safe_numeric(anomalies["farm_work_flag"]).fillna(0) + safe_numeric(anomalies["nonfarm_work_flag"]).fillna(0) + safe_numeric(anomalies["employed_flag"]).fillna(0) + safe_numeric(anomalies["self_employed_flag"]).fillna(0) + safe_numeric(anomalies["side_job_flag"]).fillna(0)) == 0)).astype(int)
    anomalies["anomaly_age_gt100"] = (safe_numeric(anomalies["age"]) > 100).astype(int)
    anomalies["anomaly_hours_gt100"] = (safe_numeric(anomalies["work_hours_weekly"]) > 100).astype(int)
    anomalies["anomaly_days_gt365"] = (safe_numeric(anomalies["work_days_yearly"]) > 365).astype(int)

    reason_cols = [
        ("anomaly_labor0_hours", "labor0_hours_positive"),
        ("anomaly_labor0_days", "labor0_days_positive"),
        ("anomaly_labor1_no_flags", "labor1_no_positive_flags"),
        ("anomaly_age_gt100", "age_gt100"),
        ("anomaly_hours_gt100", "work_hours_gt100"),
        ("anomaly_days_gt365", "work_days_gt365"),
    ]
    anomalies["anomaly_reason"] = anomalies.apply(
        lambda row: "; ".join(label for col, label in reason_cols if row[col] == 1),
        axis=1,
    )

    anomalies.to_csv(ANOMALY_CSV, index=False, encoding="utf-8-sig")

    work_hours_stats = (
        out.groupby("labor_type")["work_hours_weekly"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    work_days_stats = (
        out.groupby("labor_type")["work_days_yearly"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    tables = {
        "year_labor": out.groupby(["year", "labor_participation"]).size().rename("n").reset_index(),
        "age_labor": out.groupby(["age_group", "labor_participation"], observed=False).size().rename("n").reset_index(),
        "rural_labor": out.groupby(["rural", "labor_participation"]).size().rename("n").reset_index(),
        "female_labor": out.groupby(["female", "labor_participation"]).size().rename("n").reset_index(),
        "retire_labor": out.groupby(["retired_flag_raw", "labor_participation"]).size().rename("n").reset_index(),
        "pension_labor": out.groupby(["pension", "labor_participation"]).size().rename("n").reset_index(),
        "labor_type_age": out.groupby(["labor_type", "age_group"], observed=False).size().rename("n").reset_index(),
        "labor_type_rural": out.groupby(["labor_type", "rural"]).size().rename("n").reset_index(),
        "work_hours_stats": work_hours_stats,
        "work_days_stats": work_days_stats,
        "anomaly_summary": pd.DataFrame(
            [
                {"metric": "labor_participation_0_but_hours_positive", "n": int(((safe_numeric(out["labor_participation"]) == 0) & (safe_numeric(out["work_hours_weekly"]) > 0)).sum())},
                {"metric": "labor_participation_0_but_days_positive", "n": int(((safe_numeric(out["labor_participation"]) == 0) & (safe_numeric(out["work_days_yearly"]) > 0)).sum())},
                {"metric": "labor_participation_1_but_no_positive_flags", "n": int(((safe_numeric(out["labor_participation"]) == 1) & (flag_sum == 0)).sum())},
                {"metric": "age_gt_100", "n": int((safe_numeric(out["age"]) > 100).sum())},
                {"metric": "work_hours_weekly_gt_100", "n": int((safe_numeric(out["work_hours_weekly"]) > 100).sum())},
                {"metric": "work_days_yearly_gt_365", "n": int((safe_numeric(out["work_days_yearly"]) > 365).sum())},
                {"metric": "all_anomaly_records", "n": len(anomalies)},
            ]
        ),
    }
    return tables, anomalies, work_hours_stats


def add_age_and_suitability(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["age_valid_for_model"] = np.where(
        safe_numeric(out["age"]).between(45, 90),
        1,
        0,
    )

    poor = safe_numeric(out["poor_health"]) == 1
    adl = safe_numeric(out["adl_limit"]) == 1
    iadl = safe_numeric(out["iadl_limit"]) == 1
    family_high = safe_numeric(out["family_care_index_v1_high"]) == 1
    econ_high = safe_numeric(out["economic_pressure_index_v1_high"]) == 1
    labor = safe_numeric(out["labor_participation"])
    sufficient_health = (safe_numeric(out["poor_health"]) == 0) & (safe_numeric(out["adl_limit"]) == 0) & (safe_numeric(out["iadl_limit"]) == 0)

    out["suitability_group_v1"] = "unknown"
    out.loc[(poor | adl | iadl) & econ_high, "suitability_group_v1"] = "E"
    out.loc[(labor == 1) & econ_high & (out["suitability_group_v1"] == "unknown"), "suitability_group_v1"] = "B"
    out.loc[
        (labor == 1)
        & sufficient_health
        & (~family_high)
        & (~econ_high)
        & (out["suitability_group_v1"] == "unknown"),
        "suitability_group_v1",
    ] = "A"
    out.loc[
        (labor == 0)
        & (poor | adl | iadl | family_high)
        & (out["suitability_group_v1"] == "unknown"),
        "suitability_group_v1",
    ] = "C"
    out.loc[
        (labor == 0)
        & sufficient_health
        & (~family_high)
        & safe_numeric(out["age"]).le(65)
        & (out["suitability_group_v1"] == "unknown"),
        "suitability_group_v1",
    ] = "D"

    unknown_reason = pd.Series("other_conflict_or_missing", index=out.index, dtype="object")
    unknown_reason.loc[labor.isna()] = "missing_labor_participation"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & labor.notna()
        & (safe_numeric(out["poor_health"]).isna() | safe_numeric(out["adl_limit"]).isna() | safe_numeric(out["iadl_limit"]).isna()),
    ] = "missing_core_health"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & labor.notna()
        & safe_numeric(out["family_care_index_v1"]).isna(),
    ] = "missing_family_care_index_v1"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & labor.notna()
        & safe_numeric(out["economic_pressure_index_v1"]).isna(),
    ] = "missing_economic_pressure_index_v1"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & (labor == 1)
        & sufficient_health.notna()
        & (~econ_high)
        & ((~sufficient_health) | family_high),
    ] = "working_with_health_or_family_constraints"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & (labor == 0)
        & sufficient_health
        & (~family_high)
        & safe_numeric(out["age"]).gt(65),
    ] = "healthy_nonworking_but_age_gt65"
    unknown_reason.loc[
        (out["suitability_group_v1"] == "unknown")
        & (labor == 0)
        & sufficient_health
        & (~family_high)
        & safe_numeric(out["age"]).le(65)
        & econ_high,
    ] = "healthy_nonworking_with_high_economic_pressure"
    unknown_reason.loc[out["suitability_group_v1"] != "unknown"] = np.nan
    out["suitability_group_v1_unknown_reason"] = unknown_reason

    unknown_summary = (
        out.loc[out["suitability_group_v1"] == "unknown", "suitability_group_v1_unknown_reason"]
        .value_counts(dropna=False)
        .rename_axis("unknown_reason")
        .reset_index(name="n")
    )
    return out, unknown_summary


def build_variable_source_v1() -> pd.DataFrame:
    rows = [
        {"final_variable": "co_reside_child", "source": "2018 Family_Information cb053_*; 2020 Family_Information ca014_*", "rule": "2018 依据每个子女是否与受访者同住/同院；2020 依据与子女同住时长 > 0", "notes": "两期口径相近但变量形式不同"},
        {"final_variable": "co_reside_parent", "source": "2018 Family_Information ca016_*", "rule": "父母/公婆居住地为与受访者同住则记 1", "notes": "2020 未找到稳定同口径"},
        {"final_variable": "care_elder_or_disabled", "source": "v0 social7 + 2018 Family_Information ca026_w3_*", "rule": "照顾病人/残疾人或同住父母无自理能力", "notes": "较适合作为家庭照料与赡养压力代理"},
        {"final_variable": "care_grandchild", "source": "2018 Family_Information cb067_*, cb068_*; 2020 Family_Information cc001", "rule": "未稳定构造，v1 保留缺失", "notes": "建议作为候选概念而非主模型变量"},
        {"final_variable": "grandchild_number", "source": "2018 Family_Information cb067_*", "rule": "对子女对应孙辈数量求和", "notes": "2020 不可比，仅补充"},
        {"final_variable": "income_total_nonnegative", "source": "v0 income_total", "rule": "max(income_total, 0)", "notes": ""},
        {"final_variable": "hhcperc_nonnegative", "source": "v0 hhcperc", "rule": "max(hhcperc, 0)", "notes": ""},
        {"final_variable": "log_income_total_v1", "source": "income_total_nonnegative", "rule": "log1p(income_total_nonnegative)", "notes": "避免对负值直接 log1p"},
        {"final_variable": "log_hhcperc_v1", "source": "hhcperc_nonnegative", "rule": "log1p(hhcperc_nonnegative)", "notes": "避免对负值直接 log1p"},
        {"final_variable": "economic_pressure_index_v1", "source": "pension, hhcperc_nonnegative, medical_expense, intergen_support_in, intergen_support_out", "rule": "按题设五项分年中位数规则加总", "notes": ""},
        {"final_variable": "family_care_index_v1", "source": "hchild, family_size, intergen_support_out, care_elder_or_disabled, co_reside_child", "rule": "按题设推荐的五项稳定变量构造", "notes": "未纳入 care_grandchild"},
        {"final_variable": "age_valid_for_model", "source": "age", "rule": "45<=age<=90 记 1，否则 0", "notes": ""},
        {"final_variable": "suitability_group_v1", "source": "labor_participation, health flags, family_care_index_v1, economic_pressure_index_v1, age", "rule": "按 A/B/C/D/E/unknown 规则分组", "notes": "仅作分层标签，不建议作为主模型因变量"},
    ]
    return pd.DataFrame(rows)


def build_report_tables(
    full: pd.DataFrame,
    model: pd.DataFrame,
    model_age90: pd.DataFrame,
    family_mapping: pd.DataFrame,
    econ_checks: dict[str, pd.DataFrame],
    labor_tables: dict[str, pd.DataFrame],
    unknown_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sample_flow = pd.DataFrame(
        [
            {"step": "v0_full input", "n": len(full)},
            {"step": "v1_full", "n": len(full)},
            {"step": "v1_model", "n": len(model)},
            {"step": "v1_model_age90", "n": len(model_age90)},
            {"step": "age<=95 sensitivity sample", "n": int(model["age"].le(95).sum())},
        ]
    )
    wave_distribution = full.groupby(["wave", "year"]).size().rename("n").reset_index()
    labor_distribution = full.groupby(["year", "labor_participation"]).size().rename("n").reset_index()
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
        full.groupby(["year", "care_elder_or_disabled", "co_reside_child", "family_care_index_v1"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    labor_by_econ = (
        full.groupby(["year", "pension", "economic_pressure_index_v1"])["labor_participation"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "labor_rate"})
    )
    retire_labor = full.groupby(["year", "retired_flag_raw", "labor_participation"]).size().rename("n").reset_index()
    pension_labor = econ_checks["pension_labor_crosstab"]
    econ_labor = econ_checks["economic_pressure_labor_crosstab"]
    missing_rate = (
        full.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "variable"})
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )
    age_check = full.groupby("year")["age"].agg(["count", "min", "median", "max", "mean"]).reset_index()
    suitability_dist = full.groupby(["year", "suitability_group_v1"]).size().rename("n").reset_index()
    unknown_block = pd.DataFrame(
        {
            "year": pd.Series([pd.NA] * len(unknown_summary), dtype="Int64"),
            "suitability_group_v1": unknown_summary["unknown_reason"].astype("string"),
            "n": safe_numeric(unknown_summary["n"]).astype("Int64"),
            "section": "unknown_reason",
        }
    )
    suitability_sheet = pd.concat(
        [suitability_dist.assign(section="distribution"), unknown_block],
        ignore_index=True,
    )
    return {
        "sample_flow": sample_flow,
        "wave_distribution": wave_distribution,
        "labor_distribution": labor_distribution,
        "labor_by_age_gender_rural": labor_by_age_gender_rural,
        "labor_by_health": labor_by_health,
        "labor_by_family_care_v1": labor_by_family,
        "labor_by_economic_pressure_v1": labor_by_econ,
        "retire_labor_crosstab": retire_labor,
        "pension_labor_crosstab": pension_labor,
        "economic_pressure_labor_crosstab": econ_labor,
        "missing_rate": missing_rate,
        "variable_source_v1": build_variable_source_v1(),
        "family_care_mapping_v1": family_mapping,
        "economic_pressure_check_v1": econ_checks["summary_stats"],
        "labor_logic_anomalies_summary": labor_tables["anomaly_summary"],
        "age_check": age_check,
        "suitability_group_v1_distribution": suitability_sheet,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full, v0_model = load_v0_data()
    family_candidates = scan_family_candidates()
    family_mapping = build_family_mapping(family_candidates)

    full = merge_family_v1(full)
    full = add_economic_v1(full)
    full, unknown_summary = add_age_and_suitability(full)

    family_dist_tables = build_family_distribution_tables(full)
    with pd.ExcelWriter(FAMILY_DISTRIBUTION_XLSX, engine="openpyxl") as writer:
        for sheet, table in family_dist_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    econ_checks = build_economic_check_tables(full)
    with pd.ExcelWriter(ECON_CHECK_XLSX, engine="openpyxl") as writer:
        for sheet, table in econ_checks.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    labor_tables, anomalies, _ = build_labor_logic_outputs(full)

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

    report_tables = build_report_tables(full, model, model_age90, family_mapping, econ_checks, labor_tables, unknown_summary)
    with pd.ExcelWriter(V1_REPORT, engine="openpyxl") as writer:
        for sheet, table in report_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    full.to_csv(V1_FULL, index=False, encoding="utf-8-sig")
    model.to_csv(V1_MODEL, index=False, encoding="utf-8-sig")
    model_age90.to_csv(V1_MODEL_AGE90, index=False, encoding="utf-8-sig")

    family_missing_rate = full[FAMILY_CORE_V1].isna().all(axis=1).mean()
    econ_dist = full["economic_pressure_index_v1"].value_counts(dropna=False).sort_index()
    pension_labor_year = (
        full.groupby(["year", "pension", "labor_participation"]).size().rename("n").reset_index()
    )
    suitability_dist = (
        full["suitability_group_v1"].value_counts(dropna=False).rename_axis("suitability_group_v1").reset_index(name="n")
    )

    print(f"v1_full sample n: {len(full)}")
    print(f"v1_model sample n: {len(model)}")
    print(f"v1_model_age90 sample n: {len(model_age90)}")
    print(f"age<=95 sensitivity sample n: {int(model['age'].le(95).sum())}")
    for year, rate in full.groupby("year")["labor_participation"].mean().items():
        print(f"{int(year)} labor participation rate: {rate:.4f}")
    print(f"Family care core all-missing rate: {family_missing_rate:.4f}")
    print("economic_pressure_index_v1 distribution:")
    for key, value in econ_dist.items():
        print(f"  {key}: {value}")
    print("pension × labor_participation crosstab:")
    for row in pension_labor_year.itertuples(index=False):
        print(f"  year={int(row.year)}, pension={row.pension}, labor={row.labor_participation}, n={row.n}")
    print(f"labor_logic_anomalies n: {len(anomalies)}")
    print("suitability_group_v1 distribution:")
    for row in suitability_dist.itertuples(index=False):
        print(f"  {row.suitability_group_v1}: {row.n}")
    print("unknown main reasons:")
    for row in unknown_summary.head(5).itertuples(index=False):
        print(f"  {row.unknown_reason}: {row.n}")
    print("Recommendation on wording:")
    print("  Yes. 建议后续论文把“家庭照料责任”扩展表述为“家庭照料与代际支持压力”。")
    print("  理由 1: 当前两期最稳定可比的家庭变量不仅有照料代理变量，还有对子女经济支持与同住安排。")
    print("  理由 2: 直接“照料孙辈”在 2020 缺少稳定口径，而代际支持和共居安排更可操作。")
    print("  理由 3: 对中老年劳动供给的真实约束往往同时来自时间照料与经济支持义务。")


if __name__ == "__main__":
    main()
