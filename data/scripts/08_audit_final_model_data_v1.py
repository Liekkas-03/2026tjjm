from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "outputs" / "final_model_data_v1"
AUDIT_DIR = BASE_DIR / "final_leakage_audit_v1"

ML_FILE = BASE_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final.csv"
LOGIT_FILE = BASE_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final.csv"
ROLE_FILE = BASE_DIR / "final_feature_roles_v1.xlsx"

TARGET_COL = "labor_participation"
ID_CANDIDATES = {
    "id",
    "pid",
    "person_id",
    "respondent_id",
    "individual_id",
    "householdid",
    "household_id",
    "hhid",
    "communityid",
    "community_id",
    "wave",
}
ALLOWED_CONTROL_VARS = {"year"}
HIGH_RISK_KEYWORDS = [
    "labor",
    "work",
    "working",
    "job",
    "retire",
    "retired",
    "retirement",
    "employment",
    "employed",
    "self_employed",
    "farm_work",
    "nonfarm_work",
    "side_job",
    "wage",
    "salary",
    "occupation",
    "hours",
    "days_yearly",
    "suitability",
    "group",
    "target",
    "label",
    "pension",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_id_or_group_col(column: str) -> bool:
    return column.lower() in ID_CANDIDATES


def feature_columns(df: pd.DataFrame) -> list[str]:
    features = []
    for column in df.columns:
        if column == TARGET_COL:
            continue
        if is_id_or_group_col(column):
            continue
        features.append(column)
    return features


def top_value_share(series: pd.Series) -> float:
    value_counts = series.value_counts(dropna=False, normalize=True)
    if value_counts.empty:
        return np.nan
    return float(value_counts.iloc[0])


def numeric_min_max(series: pd.Series) -> tuple[float | str | None, float | str | None]:
    numeric = safe_numeric(series)
    if numeric.notna().any():
        return float(numeric.min()), float(numeric.max())
    if series.dropna().empty:
        return None, None
    return str(series.dropna().min()), str(series.dropna().max())


def build_columns_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        min_value, max_value = numeric_min_max(series)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_rate": round(float(series.isna().mean()), 6),
                "unique_count": int(series.nunique(dropna=False)),
                "min": min_value,
                "max": max_value,
                "is_target": column == TARGET_COL,
                "is_id_or_group": is_id_or_group_col(column),
                "will_enter_model": (column != TARGET_COL) and (not is_id_or_group_col(column)),
            }
        )
    return pd.DataFrame(rows)


def find_keyword(column: str) -> str:
    lower = column.lower()
    if lower == "y":
        return "y"
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in lower:
            return keyword
    return ""


def build_strict_keyword_check(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        matched = find_keyword(column)
        is_target = column == TARGET_COL
        is_id = is_id_or_group_col(column)
        enters_model = (column != TARGET_COL) and (not is_id)

        if is_target:
            flag = "target"
            action = "keep_as_target_only"
            reason = "Target variable is not a feature."
        elif is_id:
            flag = "id_or_group"
            action = "keep_for_grouping_only"
            reason = "ID or grouping variable does not enter features."
        elif column in ALLOWED_CONTROL_VARS:
            flag = "year_control"
            action = "keep_as_control"
            reason = "Year is allowed as a control variable."
        elif "suitability_group" in column.lower():
            flag = "high_risk_leakage"
            action = "remove"
            reason = "Suitability group is target-derived and must not enter model."
        elif "pension" in column.lower():
            flag = "high_risk_leakage"
            action = "extension_only_or_remove"
            reason = "Pension variables should not enter the main model."
        elif matched:
            flag = "high_risk_leakage"
            action = "remove"
            reason = "Column name matches a leakage keyword."
        else:
            flag = "clear"
            action = "keep"
            reason = "No strict keyword leakage found."

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "matched_keyword": matched,
                "is_target": is_target,
                "is_id_or_group": is_id,
                "will_enter_model": enters_model,
                "flag": flag,
                "action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def single_variable_accuracy_auc(feature: pd.Series, target: pd.Series) -> tuple[float | None, float | None]:
    x = safe_numeric(feature)
    y = safe_numeric(target)
    mask = x.notna() & y.notna()
    x = x.loc[mask]
    y = y.loc[mask]
    if len(x) == 0 or y.nunique() < 2 or x.nunique() < 2:
        return None, None

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    try:
        model.fit(x.to_frame(), y.astype(int))
        pred_prob = model.predict_proba(x.to_frame())[:, 1]
        pred_label = model.predict(x.to_frame())
        auc = roc_auc_score(y.astype(int), pred_prob)
        acc = accuracy_score(y.astype(int), pred_label)
        return float(auc), float(acc)
    except Exception:
        try:
            scores = x.astype(float)
            auc = roc_auc_score(y.astype(int), scores)
            auc = max(float(auc), float(1 - auc))
            threshold = float(scores.median())
            pred_label = (scores >= threshold).astype(int)
            acc = accuracy_score(y.astype(int), pred_label)
            return auc, float(acc)
        except Exception:
            return None, None


def build_proxy_check(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    target = safe_numeric(df[TARGET_COL]).astype(int)
    for column in feature_columns(df):
        series = df[column]
        numeric = safe_numeric(series)
        corr = None
        if numeric.notna().any() and numeric.nunique(dropna=True) > 1:
            corr = float(numeric.corr(target))
        auc, acc = single_variable_accuracy_auc(series, target)
        share = top_value_share(series)
        flags: list[str] = []
        if auc is not None and auc > 0.85:
            flags.append("suspicious_proxy")
        if corr is not None and abs(corr) > 0.75:
            flags.append("suspicious_proxy")
        if share > 0.99:
            flags.append("low_variance")
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "corr_with_target": round(corr, 6) if corr is not None else np.nan,
                "single_variable_auc": round(auc, 6) if auc is not None else np.nan,
                "single_variable_accuracy": round(acc, 6) if acc is not None else np.nan,
                "unique_count": int(series.nunique(dropna=False)),
                "top_value_share": round(float(share), 6) if pd.notna(share) else np.nan,
                "flag": "; ".join(flags) if flags else "clear",
            }
        )
    return pd.DataFrame(rows)


def build_role_consistency(
    roles: pd.DataFrame,
    ml_df: pd.DataFrame,
    logit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ml_features = set(feature_columns(ml_df))
    logit_features = set(feature_columns(logit_df))
    ml_selected = set(roles.loc[roles["selected_for_ml"] == True, "column"].astype(str))
    logit_selected = set(roles.loc[roles["selected_for_logit"] == True, "column"].astype(str))
    role_lookup = roles.set_index("column").to_dict("index")

    for column in sorted(ml_selected):
        rows.append(
            {
                "check_type": "role_selected_ml_in_data",
                "column": column,
                "status": "pass" if column in ml_features else "fail",
                "detail": "Selected_for_ml variable should exist in ML feature columns.",
            }
        )
    for column in sorted(ml_features):
        rows.append(
            {
                "check_type": "ml_feature_in_role_table",
                "column": column,
                "status": "pass" if column in role_lookup else "fail",
                "detail": "Every ML feature should have a role-table record.",
            }
        )
    for column in sorted(logit_selected):
        rows.append(
            {
                "check_type": "role_selected_logit_in_data",
                "column": column,
                "status": "pass" if column in logit_features else "fail",
                "detail": "Selected_for_logit variable should exist in Logit feature columns.",
            }
        )
    for column in sorted(logit_features):
        rows.append(
            {
                "check_type": "logit_feature_in_role_table",
                "column": column,
                "status": "pass" if column in role_lookup else "fail",
                "detail": "Every Logit feature should have a role-table record.",
            }
        )
    for column in sorted(ml_features | logit_features):
        role = role_lookup.get(column, {}).get("role", "")
        if role == "excluded_leakage":
            rows.append(
                {
                    "check_type": "excluded_leakage_in_final_features",
                    "column": column,
                    "status": "fail",
                    "detail": "Role table marks this as excluded_leakage but it is still in final features.",
                }
            )
        if role == "extension_only":
            rows.append(
                {
                    "check_type": "extension_only_in_final_features",
                    "column": column,
                    "status": "fail",
                    "detail": "Role table marks this as extension_only but it is still in final features.",
                }
            )

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby("check_type")["status"]
        .apply(lambda x: "pass" if (x == "pass").all() else "fail")
        .reset_index(name="overall_status")
    )
    return summary_df, detail_df


def remove_problem_columns(df: pd.DataFrame, columns_to_remove: set[str]) -> pd.DataFrame:
    keep_cols = [column for column in df.columns if column not in columns_to_remove]
    return df[keep_cols].copy()


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    ml_df = pd.read_csv(ML_FILE, low_memory=False)
    logit_df = pd.read_csv(LOGIT_FILE, low_memory=False)
    roles = pd.read_excel(ROLE_FILE)

    ml_columns_audit = build_columns_audit(ml_df)
    logit_columns_audit = build_columns_audit(logit_df)

    strict_ml = build_strict_keyword_check(ml_df, "ml")
    strict_logit = build_strict_keyword_check(logit_df, "logit")
    strict_all = pd.concat([strict_ml, strict_logit], ignore_index=True)

    proxy_ml = build_proxy_check(ml_df, "ml")
    proxy_logit = build_proxy_check(logit_df, "logit")
    proxy_all = pd.concat([proxy_ml, proxy_logit], ignore_index=True)

    role_summary, role_detail = build_role_consistency(roles, ml_df, logit_df)

    high_risk_feature_rows = strict_all.loc[
        (strict_all["will_enter_model"] == True) & (strict_all["flag"] == "high_risk_leakage")
    ].copy()
    high_risk_columns = set(high_risk_feature_rows["column"].astype(str))

    suspicious_proxy_rows = proxy_all.loc[proxy_all["flag"].str.contains("suspicious_proxy", na=False)].copy()
    suspicious_proxy_columns = set(suspicious_proxy_rows["column"].astype(str))

    extension_conflict_cols = set(
        role_detail.loc[role_detail["check_type"] == "extension_only_in_final_features", "column"].astype(str)
    )
    excluded_leakage_conflict_cols = set(
        role_detail.loc[role_detail["check_type"] == "excluded_leakage_in_final_features", "column"].astype(str)
    )

    pension_in_features = any("pension" in column.lower() for column in feature_columns(ml_df) + feature_columns(logit_df))
    suitability_in_features = any("suitability_group" in column.lower() for column in feature_columns(ml_df) + feature_columns(logit_df))
    labor_like_in_features = any(
        any(keyword in column.lower() for keyword in ["labor", "work", "retire", "job"])
        for column in feature_columns(ml_df) + feature_columns(logit_df)
    )

    removal_columns = high_risk_columns | extension_conflict_cols | excluded_leakage_conflict_cols
    audited_ml = remove_problem_columns(ml_df, removal_columns)
    audited_logit = remove_problem_columns(logit_df, removal_columns)

    ml_audited_path = BASE_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"
    logit_audited_path = BASE_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv"
    audited_ml.to_csv(ml_audited_path, index=False, encoding="utf-8-sig")
    audited_logit.to_csv(logit_audited_path, index=False, encoding="utf-8-sig")

    final_columns_path = AUDIT_DIR / "final_columns_audit_v1.xlsx"
    strict_keyword_path = AUDIT_DIR / "strict_keyword_leakage_check_v1.xlsx"
    proxy_path = AUDIT_DIR / "proxy_leakage_check_v1.xlsx"
    role_check_path = AUDIT_DIR / "feature_role_consistency_check_v1.xlsx"
    report_path = AUDIT_DIR / "final_leakage_audit_report_v1.txt"

    with pd.ExcelWriter(final_columns_path, engine="openpyxl") as writer:
        ml_columns_audit.to_excel(writer, sheet_name="ml_columns", index=False)
        logit_columns_audit.to_excel(writer, sheet_name="logit_columns", index=False)

    with pd.ExcelWriter(strict_keyword_path, engine="openpyxl") as writer:
        strict_ml.to_excel(writer, sheet_name="ml_strict_check", index=False)
        strict_logit.to_excel(writer, sheet_name="logit_strict_check", index=False)
        high_risk_feature_rows.to_excel(writer, sheet_name="high_risk_feature_rows", index=False)

    with pd.ExcelWriter(proxy_path, engine="openpyxl") as writer:
        proxy_ml.to_excel(writer, sheet_name="ml_proxy_check", index=False)
        proxy_logit.to_excel(writer, sheet_name="logit_proxy_check", index=False)
        suspicious_proxy_rows.to_excel(writer, sheet_name="suspicious_proxy_rows", index=False)

    with pd.ExcelWriter(role_check_path, engine="openpyxl") as writer:
        role_summary.to_excel(writer, sheet_name="summary", index=False)
        role_detail.to_excel(writer, sheet_name="detail", index=False)

    pass_conditions = [
        len(high_risk_columns) == 0,
        len(extension_conflict_cols) == 0,
        len(excluded_leakage_conflict_cols) == 0,
        not pension_in_features,
        not suitability_in_features,
        not labor_like_in_features,
    ]
    final_status = "PASS" if all(pass_conditions) else "FAIL"

    lines = [
        "Final Leakage Audit Report V1",
        f"ML data ready for training: {'Yes' if final_status == 'PASS' else 'No'}",
        f"Logit data ready for regression: {'Yes' if final_status == 'PASS' else 'No'}",
        f"High-risk leakage variables found: {'No' if len(high_risk_columns) == 0 else 'Yes'}",
        f"Suspicious proxy variables found: {'No' if len(suspicious_proxy_columns) == 0 else 'Yes'}",
        f"Pension-related variables entered main features: {'Yes' if pension_in_features else 'No'}",
        f"Suitability-group variables entered main features: {'Yes' if suitability_in_features else 'No'}",
        f"Labor/work/retire/job variables entered main features: {'Yes' if labor_like_in_features else 'No'}",
    ]

    if removal_columns:
        lines.append("Variables removed in audited outputs:")
        for column in sorted(removal_columns):
            lines.append(f"  - {column}")
    else:
        lines.append("No mandatory-removal variables were found; audited outputs mirror the original v4 files.")

    if suspicious_proxy_columns:
        lines.append("Suspicious proxy variables requiring attention:")
        for column in sorted(suspicious_proxy_columns):
            lines.append(f"  - {column}")

    lines.append(f"Final conclusion: {final_status}")
    if final_status == "PASS":
        lines.append("PASS，可以进入建模。")
    else:
        lines.append("FAIL，建议先按 audited 文件继续，并人工复核报告中列出的变量。")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    ml_feature_count = len(feature_columns(ml_df))
    logit_feature_count = len(feature_columns(logit_df))

    print(f"ML data shape: {ml_df.shape[0]} rows x {ml_df.shape[1]} columns")
    print(f"Logit data shape: {logit_df.shape[0]} rows x {logit_df.shape[1]} columns")
    print(f"ML feature count: {ml_feature_count}")
    print(f"Logit feature count: {logit_feature_count}")
    print(f"High-risk leakage variable count: {len(high_risk_columns)}")
    print(f"Suspicious proxy variable count: {len(suspicious_proxy_columns)}")
    print(f"Pension-related variables entered final features: {'Yes' if pension_in_features else 'No'}")
    print(f"Suitability-group variables entered final features: {'Yes' if suitability_in_features else 'No'}")
    print(f"Labor/work/retire/job variables entered final features: {'Yes' if labor_like_in_features else 'No'}")
    print(f"Final conclusion: {final_status}")
    print("Audited file paths:")
    print(f"  {ml_audited_path}")
    print(f"  {logit_audited_path}")


if __name__ == "__main__":
    main()
