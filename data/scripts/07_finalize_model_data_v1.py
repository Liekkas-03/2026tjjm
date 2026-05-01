from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "final_model_data_v1"
INPUT_FILENAME = "CHARLS_labor_panel_2018_2020_v3_train_ready.csv"
LABELS_FILENAME = "labels.csv"

LEAKAGE_KEYWORDS = [
    "labor",
    "work",
    "job",
    "retire",
    "retirement",
    "employment",
    "employed",
    "wage",
    "salary",
    "working_hour",
    "occupation",
    "income_from_work",
]
PENSION_KEYWORDS = ["pension"]
TARGET_CANDIDATES = ["labor_participation", "labor_status", "is_working", "target", "y"]
ID_CANDIDATES = ["ID", "id", "pid", "person_id", "respondent_id", "individual_id"]
YEAR_CANDIDATES = ["year", "wave"]
OTHER_ID_PATTERNS = ["household_id", "householdid", "hhid", "family_id", "community_id", "communityid"]
WEIGHT_KEYWORDS = ["weight"]

HEALTH_COLUMNS = {
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
}
FAMILY_COLUMNS = {
    "hchild",
    "family_size",
    "co_reside_child",
    "care_elder_or_disabled",
    "intergen_support_in",
    "intergen_support_out",
    "family_care_index_v1",
}
ECONOMIC_COLUMNS = {
    "log_hhcperc_v1",
    "medical_expense",
    "medical_burden",
    "economic_pressure_index_v1",
}
DEMOGRAPHIC_COLUMNS = {"age", "female", "married", "rural", "edu", "year", "wave"}
BEHAVIOR_COLUMNS = {"smokev", "drinkl", "exercise", "totmet"}
EXTENSION_COLUMNS = {"ins", "INDV_weight", "HH_weight", "INDV_weight_missing", "HH_weight_missing"}

LOGIT_PRIORITY = [
    "age",
    "age_squared",
    "female",
    "married",
    "urban",
    "edu",
    "year_2020",
    "poor_health",
    "chronic_count",
    "adl_limit",
    "iadl_limit",
    "depression_high",
    "total_cognition_w",
    "family_care_index_v1",
    "co_reside_child",
    "care_elder_or_disabled",
    "log_hhcperc_v1_w",
    "log_medical_expense_w",
    "economic_pressure_index_v1",
    "log_intergen_support_in_w",
    "log_intergen_support_out_w",
    "smokev",
    "drinkl",
    "exercise",
]
CORR_EXEMPT_PAIRS = {frozenset({"age", "age_squared"})}


def search_file(filename: str) -> Path | None:
    direct_candidates = [
        ROOT / filename,
        ROOT / "data" / filename,
        ROOT / "outputs" / filename,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    search_roots = [ROOT / "data", ROOT / "outputs", ROOT]
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in search_root.rglob(filename):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                return candidate
    return None


def load_labels_map() -> tuple[dict[str, str], Path | None]:
    path = search_file(LABELS_FILENAME)
    if path is None:
        return {}, None

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            labels = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return {}, path

    if labels.shape[1] < 2:
        return {}, path

    key_col = labels.columns[0]
    value_col = labels.columns[1]
    out: dict[str, str] = {}
    for _, row in labels.iterrows():
        key = str(row[key_col]).strip()
        value = str(row[value_col]).strip()
        if key and key.lower() != "nan":
            out[key] = value
    return out, path


def detect_target(columns: Iterable[str]) -> str | None:
    column_list = list(columns)
    for candidate in TARGET_CANDIDATES:
        if candidate in column_list:
            return candidate
    return None


def detect_id_fields(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    column_list = list(columns)
    for candidate in ID_CANDIDATES:
        if candidate in column_list and candidate not in out:
            out.append(candidate)
    for column in column_list:
        lower = column.lower()
        if any(pattern in lower for pattern in OTHER_ID_PATTERNS) and column not in out:
            out.append(column)
    return out


def detect_year_fields(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    column_list = list(columns)
    for candidate in YEAR_CANDIDATES:
        if candidate in column_list and candidate not in out:
            out.append(candidate)
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def dominant_share(series: pd.Series) -> float:
    counts = series.value_counts(dropna=False, normalize=True)
    if counts.empty:
        return 1.0
    return float(counts.iloc[0])


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> tuple[pd.Series, float | None, float | None]:
    numeric = safe_numeric(series).copy()
    non_na = numeric.dropna()
    if non_na.empty:
        return numeric, None, None
    lower = float(non_na.quantile(lower_q))
    upper = float(non_na.quantile(upper_q))
    return numeric.clip(lower=lower, upper=upper), lower, upper


def sanitize_column_name(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "col"
    if out[0].isdigit():
        out = f"f_{out}"
    return out


def is_binary(series: pd.Series) -> bool:
    values = set(safe_numeric(series).dropna().unique().tolist())
    return values.issubset({0, 1})


def as_binary(series: pd.Series) -> pd.Series:
    return safe_numeric(series).round().clip(0, 1).astype(int)


def infer_matched_keyword(column: str) -> str:
    lower = column.lower()
    for keyword in LEAKAGE_KEYWORDS:
        if keyword in lower:
            return keyword
    for keyword in PENSION_KEYWORDS:
        if keyword in lower:
            return keyword
    return ""


def generated_label(column: str, source_label: str = "") -> str:
    generated = {
        "urban": "城市居住（由 rural 反向生成，1=城市）",
        "age_squared": "年龄平方（生成变量）",
        "year_2020": "2020 年虚拟变量（生成变量）",
        "total_cognition_w": "认知总分（1%-99% winsorize）",
        "log_hhcperc_v1_w": "log_hhcperc_v1（1%-99% winsorize）",
        "medical_expense_w": "医疗支出（1%-99% winsorize）",
        "medical_burden_w": "医疗负担（1%-99% winsorize）",
        "intergen_support_in_w": "代际经济流入（1%-99% winsorize）",
        "intergen_support_out_w": "代际经济流出（1%-99% winsorize）",
        "totmet_w": "体力活动总量（1%-99% winsorize）",
        "log_medical_expense_w": "log(医疗支出 winsorize + 1)",
        "log_intergen_support_in_w": "log(代际经济流入 winsorize + 1)",
        "log_intergen_support_out_w": "log(代际经济流出 winsorize + 1)",
    }
    if column in generated:
        return generated[column]
    if column.endswith("_missing"):
        base = column[: -len("_missing")]
        return f"{source_label or base} 缺失指示变量"
    if column.startswith("edu_"):
        return f"教育水平虚拟变量 {column}"
    return source_label or column


def lookup_label(column: str, labels_map: dict[str, str]) -> str:
    if column in labels_map:
        return labels_map[column]
    if column.endswith("_missing"):
        base = column[: -len("_missing")]
        return generated_label(column, labels_map.get(base, base))
    return generated_label(column, labels_map.get(column, ""))


def infer_base_role(column: str, decision: str) -> tuple[str, str]:
    if decision == "target":
        return "target", "target"
    if decision == "id_year":
        return "id_year", "id_or_time"
    if decision == "excluded_leakage":
        return "excluded_leakage", "direct_or_proxy_work_status"
    if decision == "extension_only":
        return "extension_only", "deferred_extension"
    if decision == "excluded_high_missing_or_low_variance":
        return "excluded_high_missing_or_low_variance", "low_variance_or_non_model"

    if column in HEALTH_COLUMNS:
        return "health_constraints", "core_health"
    if column in FAMILY_COLUMNS:
        return "family_care", "core_family"
    if column in ECONOMIC_COLUMNS:
        return "economic_pressure", "core_economic"
    if column in DEMOGRAPHIC_COLUMNS:
        return "demographics_controls", "base_control"
    if column in BEHAVIOR_COLUMNS:
        return "behavior_controls", "base_control"
    if column.endswith("_missing"):
        base = column[: -len("_missing")]
        if base in HEALTH_COLUMNS:
            return "health_constraints", "missing_indicator"
        if base in FAMILY_COLUMNS:
            return "family_care", "missing_indicator"
        if base in ECONOMIC_COLUMNS:
            return "economic_pressure", "missing_indicator"
        if base in DEMOGRAPHIC_COLUMNS:
            return "demographics_controls", "missing_indicator"
        if base in BEHAVIOR_COLUMNS:
            return "behavior_controls", "missing_indicator"
    return "excluded_high_missing_or_low_variance", "not_selected_in_v1"


def build_leakage_review(
    df: pd.DataFrame,
    target_col: str,
    id_fields: list[str],
    year_fields: list[str],
    labels_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    id_year_fields = set(id_fields) | set(year_fields)
    for column in df.columns:
        series = df[column]
        missing_rate = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=False))
        matched_keyword = infer_matched_keyword(column)
        lower = column.lower()
        label = lookup_label(column, labels_map)
        dom_share = dominant_share(series)

        if column == target_col:
            decision = "target"
            reason = "Target variable is not a feature."
        elif column in id_year_fields:
            decision = "id_year"
            reason = "Identifier or time field; retained only for tracking and grouping."
        elif any(keyword in lower for keyword in PENSION_KEYWORDS):
            decision = "extension_only"
            reason = "Pension-related field deferred from first main model."
        elif any(keyword in lower for keyword in WEIGHT_KEYWORDS):
            decision = "extension_only"
            reason = "Weight field kept for extension analysis, not as a predictor."
        elif matched_keyword:
            decision = "excluded_leakage"
            reason = "Column name overlaps directly with labor or retirement status."
        elif unique_count <= 1 or dom_share > 0.99:
            decision = "excluded_high_missing_or_low_variance"
            reason = "Low-variance field; one value dominates more than 99%."
        else:
            decision = "keep_for_role_assignment"
            reason = "Passes first-round leakage and variance review."

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_rate": round(missing_rate, 6),
                "unique_count": unique_count,
                "matched_keyword": matched_keyword,
                "label_or_description": label,
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def one_hot_edu(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[str]]:
    if column not in df.columns:
        return df, []
    series = safe_numeric(df[column]).astype("Int64")
    categories = [int(value) for value in sorted(series.dropna().unique().tolist())]
    created: list[str] = []
    out = df.copy()
    for category in categories:
        name = sanitize_column_name(f"{column}_{category}")
        out[name] = np.where(series == category, 1, 0).astype(int)
        created.append(name)
    return out, created


def add_transformed_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, dict[str, object]]]:
    out = df.copy()
    generated_rows: list[dict[str, object]] = []
    transform_notes: dict[str, dict[str, object]] = {}

    if "rural" in out.columns:
        out["urban"] = (1 - safe_numeric(out["rural"])).clip(lower=0, upper=1).astype(int)
        generated_rows.append(
            {
                "column": "urban",
                "role": "demographics_controls",
                "sub_role": "generated_binary",
                "selected_for_ml": True,
                "selected_for_logit": True,
                "reason": "Generated for interpretability; 1 means urban.",
                "label_or_description": generated_label("urban"),
            }
        )

    if "age" in out.columns:
        out["age_squared"] = safe_numeric(out["age"]) ** 2
        generated_rows.append(
            {
                "column": "age_squared",
                "role": "demographics_controls",
                "sub_role": "polynomial_term",
                "selected_for_ml": True,
                "selected_for_logit": True,
                "reason": "Captures non-linear age effects.",
                "label_or_description": generated_label("age_squared"),
            }
        )

    if "year" in out.columns:
        years = sorted(safe_numeric(out["year"]).dropna().unique().tolist())
        if years:
            max_year = int(max(years))
            out["year_2020"] = np.where(safe_numeric(out["year"]) == max_year, 1, 0).astype(int)
            generated_rows.append(
                {
                    "column": "year_2020",
                    "role": "demographics_controls",
                    "sub_role": "time_dummy",
                    "selected_for_ml": True,
                    "selected_for_logit": True,
                    "reason": f"Binary control for {max_year} relative to the earlier wave.",
                    "label_or_description": generated_label("year_2020"),
                }
            )

    winsor_targets = {
        "total_cognition": "total_cognition_w",
        "log_hhcperc_v1": "log_hhcperc_v1_w",
        "medical_expense": "medical_expense_w",
        "medical_burden": "medical_burden_w",
        "intergen_support_in": "intergen_support_in_w",
        "intergen_support_out": "intergen_support_out_w",
        "totmet": "totmet_w",
    }
    for source, target in winsor_targets.items():
        if source not in out.columns:
            continue
        winsorized, lower, upper = winsorize_series(out[source])
        out[target] = winsorized
        transform_notes[target] = {
            "source": source,
            "transformation": "winsorize_1_99",
            "lower": lower,
            "upper": upper,
        }
        role, sub_role = infer_base_role(source, "keep_for_role_assignment")
        generated_rows.append(
            {
                "column": target,
                "role": role,
                "sub_role": "winsorized_continuous",
                "selected_for_ml": True,
                "selected_for_logit": target != "totmet_w",
                "reason": "Winsorized at 1% and 99% for stability.",
                "label_or_description": generated_label(target),
            }
        )

    log_targets = {
        "medical_expense_w": "log_medical_expense_w",
        "intergen_support_in_w": "log_intergen_support_in_w",
        "intergen_support_out_w": "log_intergen_support_out_w",
    }
    for source, target in log_targets.items():
        if source not in out.columns:
            continue
        out[target] = np.log1p(safe_numeric(out[source]).clip(lower=0))
        transform_notes[target] = {
            "source": source,
            "transformation": "log1p_after_winsorize",
            "lower": 0,
            "upper": None,
        }
        base_source = source[: -len("_w")] if source.endswith("_w") else source
        role, _sub_role = infer_base_role(base_source, "keep_for_role_assignment")
        generated_rows.append(
            {
                "column": target,
                "role": role,
                "sub_role": "log_transformed_continuous",
                "selected_for_ml": True,
                "selected_for_logit": True,
                "reason": "Log-transformed skewed amount after winsorization.",
                "label_or_description": generated_label(target),
            }
        )

    out, edu_dummies = one_hot_edu(out, "edu")
    for dummy in edu_dummies:
        generated_rows.append(
            {
                "column": dummy,
                "role": "demographics_controls",
                "sub_role": "one_hot_education",
                "selected_for_ml": True,
                "selected_for_logit": False,
                "reason": "Education is expanded to one-hot form for ML only.",
                "label_or_description": generated_label(dummy),
            }
        )

    return out, generated_rows, transform_notes


def build_feature_roles(
    df: pd.DataFrame,
    leakage_review: pd.DataFrame,
    labels_map: dict[str, str],
    generated_rows: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    review_lookup = leakage_review.set_index("column").to_dict("index")

    ml_missing_indicator_allow = {
        "poor_health_missing",
        "depression_high_missing",
        "co_reside_child_missing",
        "intergen_support_in_missing",
        "intergen_support_out_missing",
        "log_hhcperc_v1_missing",
        "medical_burden_missing",
        "medical_expense_missing",
        "total_cognition_missing",
    }

    for column in df.columns:
        review_row = review_lookup[column]
        role, sub_role = infer_base_role(column, str(review_row["decision"]))
        selected_for_ml = False
        selected_for_logit = False
        reason = str(review_row["reason"])
        label = lookup_label(column, labels_map)

        if role in {"health_constraints", "family_care", "economic_pressure", "demographics_controls", "behavior_controls"}:
            selected_for_ml = True
            selected_for_logit = True

        if column == "ins":
            role = "extension_only"
            sub_role = "ambiguous_institutional_proxy"
            selected_for_ml = False
            selected_for_logit = False
            reason = "Insurance coverage is deferred from v1 main models."

        if column == "rural":
            role = "demographics_controls"
            sub_role = "replaced_by_generated_urban"
            selected_for_ml = False
            selected_for_logit = False
            reason = "Replaced by generated urban for clearer interpretation."

        if column == "edu":
            role = "demographics_controls"
            sub_role = "ordinal_education"
            selected_for_ml = False
            selected_for_logit = True
            reason = "Kept as ordinal education in logit; replaced by dummies in ML."

        if column == "year":
            role = "id_year"
            sub_role = "time_tracking"
            selected_for_ml = False
            selected_for_logit = False
            reason = "Original year retained only for tracking; generated year_2020 enters models."

        if column == "wave":
            role = "id_year"
            sub_role = "survey_wave"
            selected_for_ml = False
            selected_for_logit = False
            reason = "Wave retained only for tracking."

        if column.endswith("_missing"):
            if column in ml_missing_indicator_allow and review_row["unique_count"] > 1 and review_row["missing_rate"] < 0.99:
                selected_for_ml = True
                selected_for_logit = False
                reason = "Missing indicator retained for ML to preserve upstream missingness signal."
            else:
                role = "excluded_high_missing_or_low_variance"
                sub_role = "missing_indicator_not_retained"
                selected_for_ml = False
                selected_for_logit = False
                reason = "Missing indicator excluded in v1 because it is too sparse or not needed."

        if column in {"INDV_weight", "HH_weight", "INDV_weight_missing", "HH_weight_missing"}:
            role = "extension_only"
            sub_role = "survey_weight"
            selected_for_ml = False
            selected_for_logit = False
            reason = "Survey weights are kept for extension or weighted estimation, not as predictors."

        if column in {"medical_expense", "intergen_support_in", "intergen_support_out", "total_cognition", "log_hhcperc_v1", "medical_burden", "totmet"}:
            selected_for_ml = False
            selected_for_logit = False
            reason = "Replaced by transformed version in final modeling tables."

        rows.append(
            {
                "column": column,
                "role": role,
                "sub_role": sub_role,
                "selected_for_ml": bool(selected_for_ml),
                "selected_for_logit": bool(selected_for_logit),
                "reason": reason,
                "label_or_description": label,
            }
        )

    rows.extend(generated_rows)
    roles = pd.DataFrame(rows)
    roles = roles.drop_duplicates(subset=["column"], keep="last").sort_values("column").reset_index(drop=True)
    return roles


def ensure_integer_binary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = as_binary(out[column])
    return out


def drop_high_corr_logit(df: pd.DataFrame, features: list[str]) -> tuple[list[str], pd.DataFrame]:
    corr_df = df[features].corr(numeric_only=True).abs()
    priority_lookup = {name: idx for idx, name in enumerate(LOGIT_PRIORITY)}
    keep = list(features)
    rows = []
    changed = True
    while changed:
        changed = False
        corr_df = df[keep].corr(numeric_only=True).abs()
        for i, left in enumerate(keep):
            for right in keep[i + 1 :]:
                if frozenset({left, right}) in CORR_EXEMPT_PAIRS:
                    rows.append(
                        {
                            "feature_a": left,
                            "feature_b": right,
                            "abs_corr": round(float(corr_df.loc[left, right]), 6),
                            "action": "kept_both_exempt",
                            "reason": "Predefined polynomial pair exemption.",
                        }
                    )
                    continue
                corr_value = float(corr_df.loc[left, right])
                if corr_value <= 0.9:
                    continue
                left_rank = priority_lookup.get(left, 999)
                right_rank = priority_lookup.get(right, 999)
                drop_feature = right if left_rank <= right_rank else left
                keep.remove(drop_feature)
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "abs_corr": round(corr_value, 6),
                        "action": f"dropped_{drop_feature}",
                        "reason": "Absolute correlation above 0.9; retained the more interpretable feature.",
                    }
                )
                changed = True
                break
            if changed:
                break
    if not rows:
        rows.append(
            {
                "feature_a": "",
                "feature_b": "",
                "abs_corr": np.nan,
                "action": "no_pair_above_threshold",
                "reason": "No non-exempt logit feature pair exceeded 0.9.",
            }
        )
    return keep, pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = search_file(INPUT_FILENAME)
    if input_path is None:
        raise FileNotFoundError(f"Could not find {INPUT_FILENAME}")

    df = pd.read_csv(input_path, low_memory=False)
    labels_map, labels_path = load_labels_map()

    target_col = detect_target(df.columns)
    if target_col is None:
        print("Could not identify target column. Available columns:")
        for column in df.columns:
            print(column)
        raise SystemExit(1)

    id_fields = detect_id_fields(df.columns)
    year_fields = detect_year_fields(df.columns)

    df[target_col] = as_binary(df[target_col])
    binary_candidates = [
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
    ] + [column for column in df.columns if column.endswith("_missing")]
    df = ensure_integer_binary(df, [column for column in binary_candidates if column in df.columns and is_binary(df[column])])

    leakage_review = build_leakage_review(df, target_col, id_fields, year_fields, labels_map)
    transformed_df, generated_rows, transform_notes = add_transformed_features(df)
    feature_roles = build_feature_roles(df, leakage_review, labels_map, generated_rows)

    id_year_fields = []
    for field in id_fields + year_fields:
        if field in transformed_df.columns and field not in id_year_fields:
            id_year_fields.append(field)

    ml_feature_order = [
        "age",
        "age_squared",
        "female",
        "married",
        "urban",
        "poor_health",
        "chronic_count",
        "adl_limit",
        "iadl_limit",
        "depression_high",
        "total_cognition_w",
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
        "care_elder_or_disabled",
        "family_care_index_v1",
        "log_hhcperc_v1_w",
        "log_medical_expense_w",
        "medical_burden_w",
        "economic_pressure_index_v1",
        "log_intergen_support_in_w",
        "log_intergen_support_out_w",
        "smokev",
        "drinkl",
        "exercise",
        "totmet_w",
        "year_2020",
    ]
    ml_feature_order.extend(sorted([column for column in transformed_df.columns if re.fullmatch(r"edu_\d+", column)]))

    ml_missing_indicators = [
        column
        for column in transformed_df.columns
        if column.endswith("_missing")
        and column in set(feature_roles.loc[feature_roles["selected_for_ml"], "column"].tolist())
    ]
    ml_feature_order.extend(sorted(ml_missing_indicators))
    ml_features = [column for column in ml_feature_order if column in transformed_df.columns]

    logit_feature_order = [
        "age",
        "age_squared",
        "female",
        "married",
        "urban",
        "edu",
        "year_2020",
        "poor_health",
        "chronic_count",
        "adl_limit",
        "iadl_limit",
        "depression_high",
        "total_cognition_w",
        "family_care_index_v1",
        "co_reside_child",
        "care_elder_or_disabled",
        "log_hhcperc_v1_w",
        "log_medical_expense_w",
        "economic_pressure_index_v1",
        "log_intergen_support_in_w",
        "log_intergen_support_out_w",
        "smokev",
        "drinkl",
        "exercise",
    ]
    logit_features = [column for column in logit_feature_order if column in transformed_df.columns]
    logit_features, corr_check = drop_high_corr_logit(transformed_df, logit_features)

    ml_columns = id_year_fields + [target_col] + ml_features
    logit_columns = id_year_fields + [target_col] + logit_features
    ml_final = transformed_df[ml_columns].copy()
    logit_final = transformed_df[logit_columns].copy()

    ml_final.columns = [sanitize_column_name(column) for column in ml_final.columns]
    logit_final.columns = [sanitize_column_name(column) for column in logit_final.columns]

    ml_missing = int(ml_final.isna().sum().sum())
    logit_missing = int(logit_final.isna().sum().sum())
    if ml_missing != 0 or logit_missing != 0:
        raise ValueError("Final output still contains missing values.")

    feature_roles["selected_for_ml"] = feature_roles["column"].isin(ml_features)
    feature_roles["selected_for_logit"] = feature_roles["column"].isin(logit_features)
    feature_roles.loc[feature_roles["role"] == "target", "selected_for_ml"] = False
    feature_roles.loc[feature_roles["role"] == "target", "selected_for_logit"] = False
    feature_roles.loc[feature_roles["role"] == "id_year", "selected_for_ml"] = False
    feature_roles.loc[feature_roles["role"] == "id_year", "selected_for_logit"] = False

    generated_selected_rows = []
    for row in generated_rows:
        generated_selected_rows.append(row)
    if generated_selected_rows:
        generated_df = pd.DataFrame(generated_selected_rows)
        feature_roles = feature_roles[~feature_roles["column"].isin(generated_df["column"])].copy()
        generated_df["selected_for_ml"] = generated_df["column"].isin(ml_features)
        generated_df["selected_for_logit"] = generated_df["column"].isin(logit_features)
        feature_roles = pd.concat([feature_roles, generated_df], ignore_index=True)
    feature_roles = feature_roles.sort_values(["role", "column"]).reset_index(drop=True)

    data_overview = pd.DataFrame(
        [
            {"item": "input_file", "value": str(input_path)},
            {"item": "labels_file", "value": str(labels_path) if labels_path else ""},
            {"item": "source_rows", "value": len(df)},
            {"item": "source_columns", "value": df.shape[1]},
            {"item": "target_column", "value": target_col},
            {"item": "id_fields", "value": ", ".join(id_fields)},
            {"item": "year_fields", "value": ", ".join(year_fields)},
        ]
    )
    target_distribution = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("target_value")
        .reset_index(name="count")
    )
    target_distribution["share"] = (target_distribution["count"] / len(df)).round(6)
    sample_filtering = pd.DataFrame(
        [
            {"step": "source_v3_train_ready", "rows": len(df), "columns": df.shape[1], "note": "Loaded source file."},
            {"step": "post_target_check", "rows": len(df), "columns": df.shape[1], "note": "Target identified; no row dropped."},
            {"step": "final_ml_export", "rows": len(ml_final), "columns": ml_final.shape[1], "note": "Final ML-ready table."},
            {"step": "final_logit_export", "rows": len(logit_final), "columns": logit_final.shape[1], "note": "Final logit-ready table."},
        ]
    )
    missing_summary = (
        df.isna().mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .merge(leakage_review[["column", "decision"]], on="column", how="left")
        .sort_values(["missing_rate", "column"], ascending=[False, True])
    )

    ml_selected_features = pd.DataFrame({"column": ml_features}).merge(
        feature_roles[["column", "role", "sub_role", "label_or_description"]],
        on="column",
        how="left",
    )
    logit_selected_features = pd.DataFrame({"column": logit_features}).merge(
        feature_roles[["column", "role", "sub_role", "label_or_description"]],
        on="column",
        how="left",
    )
    excluded_features = feature_roles.loc[
        ~feature_roles["selected_for_ml"] & ~feature_roles["selected_for_logit"],
        ["column", "role", "sub_role", "reason", "label_or_description"],
    ].copy()

    notes_rows = [
        {"item": "no_training", "note": "This script prepares final modeling data only; it does not train models."},
        {"item": "pension_rule", "note": "Pension-related variables are deferred to extension-only and do not enter v1 main models."},
        {"item": "ml_encoding", "note": "Education is one-hot encoded in ML data; original year remains for tracking while year_2020 enters models."},
        {"item": "logit_design", "note": "Logit data keeps a smaller, more interpretable set of features and checks pairwise correlation above 0.9."},
        {"item": "weights", "note": "Survey weights are not used as predictors in v1 final datasets."},
        {"item": "region_fields", "note": "No province or region field was found in v3 train ready, so region controls are absent from v1."},
    ]
    for column, note in sorted(transform_notes.items()):
        notes_rows.append(
            {
                "item": column,
                "note": f"{note['transformation']} from {note['source']}; lower={note['lower']}, upper={note['upper']}",
            }
        )
    notes = pd.DataFrame(notes_rows)

    leakage_path = OUTPUT_DIR / "leakage_review_v1.xlsx"
    roles_path = OUTPUT_DIR / "final_feature_roles_v1.xlsx"
    ml_path = OUTPUT_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final.csv"
    logit_path = OUTPUT_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final.csv"
    report_path = OUTPUT_DIR / "final_model_data_report_v1.xlsx"

    ml_final.to_csv(ml_path, index=False, encoding="utf-8-sig")
    logit_final.to_csv(logit_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(leakage_path, engine="openpyxl") as writer:
        leakage_review.to_excel(writer, sheet_name="leakage_review", index=False)

    with pd.ExcelWriter(roles_path, engine="openpyxl") as writer:
        feature_roles.to_excel(writer, sheet_name="feature_roles", index=False)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        data_overview.to_excel(writer, sheet_name="data_overview", index=False)
        target_distribution.to_excel(writer, sheet_name="target_distribution", index=False)
        sample_filtering.to_excel(writer, sheet_name="sample_filtering", index=False)
        missing_summary.to_excel(writer, sheet_name="missing_summary", index=False)
        leakage_review.to_excel(writer, sheet_name="leakage_review", index=False)
        feature_roles.to_excel(writer, sheet_name="feature_roles", index=False)
        ml_selected_features.to_excel(writer, sheet_name="ml_selected_features", index=False)
        logit_selected_features.to_excel(writer, sheet_name="logit_selected_features", index=False)
        excluded_features.to_excel(writer, sheet_name="excluded_features", index=False)
        corr_check.to_excel(writer, sheet_name="correlation_check_logit", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)

    leakage_count = int((leakage_review["decision"] == "excluded_leakage").sum())
    output_files = [leakage_path, roles_path, ml_path, logit_path, report_path]

    print(f"Input file path: {input_path}")
    print(f"Original v3 shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target column: {target_col}")
    print("Target distribution:")
    for row in target_distribution.itertuples(index=False):
        print(f"  {row.target_value}: {row.count} ({row.share:.4f})")
    print(f"Identified ID fields: {', '.join(id_fields) if id_fields else 'None'}")
    print(f"Identified year fields: {', '.join(year_fields) if year_fields else 'None'}")
    print(f"Suspected leakage variable count: {leakage_count}")
    print(f"Final ML shape: {ml_final.shape[0]} rows x {ml_final.shape[1]} columns")
    print(f"Final Logit shape: {logit_final.shape[0]} rows x {logit_final.shape[1]} columns")
    print(f"ML feature count: {len(ml_features)}")
    print(f"Logit feature count: {len(logit_features)}")
    print(f"Remaining missing values: ML={ml_missing}, Logit={logit_missing}")
    print("Output files:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
