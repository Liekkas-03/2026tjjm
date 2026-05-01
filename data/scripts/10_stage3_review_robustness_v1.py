from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from statsmodels.discrete.discrete_model import Logit, Probit


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "outputs" / "final_model_data_v1"
BASELINE_DIR = ROOT / "outputs" / "model_training_v1"
OUTPUT_DIR = ROOT / "outputs" / "model_stage3_review_v1"

LOGIT_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv"
ML_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"
LOGIT_RESULTS = BASELINE_DIR / "logit_results_v1.xlsx"
PROBIT_RESULTS = BASELINE_DIR / "probit_results_v1.xlsx"
ML_METRICS = BASELINE_DIR / "ml_metrics_v1.xlsx"
GROUP_CV_METRICS = BASELINE_DIR / "group_cv_metrics_v1.xlsx"
FEATURE_IMPORTANCE = BASELINE_DIR / "feature_importance_baseline_v1.xlsx"
ROBUSTNESS_METRICS = BASELINE_DIR / "robustness_metrics_v1.xlsx"
MODEL_DIAGNOSTICS = BASELINE_DIR / "model_diagnostics_v1.xlsx"
BASELINE_REPORT = BASELINE_DIR / "baseline_model_report_v1.txt"

TARGET = "labor_participation"
GROUP = "ID"
NON_FEATURE_COLS = {TARGET, "ID", "wave", "year"}
FOCUS_VARS = [
    "poor_health",
    "chronic_count",
    "adl_limit",
    "iadl_limit",
    "depression_high",
    "total_cognition_w",
    "family_care_index_v1",
    "co_reside_child",
    "care_elder_or_disabled",
    "hchild",
    "family_size",
    "economic_pressure_index_v1",
    "log_hhcperc_v1_w",
    "log_medical_expense_w",
    "medical_burden_w",
    "smokev",
    "drinkl",
    "exercise",
    "year_2020",
]
EXPECTED_DIRECTION = {
    "poor_health": "negative",
    "chronic_count": "negative_or_mixed",
    "adl_limit": "negative",
    "iadl_limit": "negative",
    "depression_high": "negative_or_mixed",
    "total_cognition_w": "positive_or_mixed",
    "family_care_index_v1": "negative_or_mixed",
    "co_reside_child": "negative_or_mixed",
    "care_elder_or_disabled": "negative",
    "hchild": "negative_or_mixed",
    "family_size": "negative_or_mixed",
    "economic_pressure_index_v1": "positive",
    "log_hhcperc_v1_w": "positive",
    "log_medical_expense_w": "negative",
    "medical_burden_w": "negative",
    "smokev": "mixed",
    "drinkl": "mixed",
    "exercise": "positive",
    "year_2020": "control_only",
}
ML_ROLE_MAP = {
    "totmet_w": "possible_proxy_risk",
    "year_2020": "year_control",
    "family_care_index_v1": "family_care_proxy",
    "economic_pressure_index_v1": "economic_pressure_index",
}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def metrics_dict(y_true: pd.Series, pred_prob: np.ndarray, pred_label: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred_prob)),
    }


def prepare_logit_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edu" in out.columns:
        values = sorted([int(v) for v in safe_numeric(out["edu"]).dropna().unique().tolist()])
        for value in values:
            col = f"edu_{value}"
            if col not in out.columns:
                out[col] = np.where(safe_numeric(out["edu"]) == value, 1, 0).astype(int)
    return out


def fit_logit(
    df: pd.DataFrame,
    target: str,
    variables: list[str],
    subset_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = df.copy()
    if subset_mask is not None:
        data = data.loc[subset_mask].copy()
    available = [var for var in variables if var in data.columns]
    missing = [var for var in variables if var not in data.columns]
    working = available.copy()
    removed: list[str] = []

    for var in available.copy():
        if safe_numeric(data[var]).nunique(dropna=True) <= 1:
            working.remove(var)
            removed.append(f"{var}: zero variance in this specification")

    while working:
        x = data[working].apply(safe_numeric)
        x = x.loc[:, ~x.columns.duplicated()].copy()
        if x.empty:
            break
        if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
            drop_col = x.columns[-1]
            working.remove(drop_col)
            removed.append(f"{drop_col}: singularity/rank deficiency")
            continue
        y = safe_numeric(data[target]).astype(int)
        x_const = sm.add_constant(x, has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = Logit(y, x_const).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data[GROUP]})
            rows = []
            for var in result.params.index:
                if var == "const":
                    continue
                rows.append(
                    {
                        "variable": var,
                        "coef": float(result.params[var]),
                        "std_err": float(result.bse[var]),
                        "p_value": float(result.pvalues[var]),
                        "odds_ratio": float(np.exp(result.params[var])),
                        "nobs": int(result.nobs),
                        "pseudo_r2": float(result.prsquared),
                        "AIC": float(result.aic),
                        "BIC": float(result.bic),
                    }
                )
            return pd.DataFrame(rows), {
                "success": True,
                "nobs": int(result.nobs),
                "pseudo_r2": float(result.prsquared),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "used_variables": working.copy(),
                "missing_variables": missing,
                "removed_variables": removed,
                "converged": bool(result.mle_retvals.get("converged", True)) if hasattr(result, "mle_retvals") else True,
            }
        except Exception as exc:
            drop_col = working[-1]
            working.remove(drop_col)
            removed.append(f"{drop_col}: dropped after {type(exc).__name__}")
    return pd.DataFrame(columns=["variable", "coef", "std_err", "p_value", "odds_ratio", "nobs", "pseudo_r2", "AIC", "BIC"]), {
        "success": False,
        "nobs": int(data.shape[0]),
        "pseudo_r2": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "used_variables": [],
        "missing_variables": missing,
        "removed_variables": removed,
        "converged": False,
    }


def train_tree_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    model.fit(X_train, y_train)
    pred_prob = model.predict_proba(X_test)[:, 1]
    pred_label = model.predict(X_test)
    return metrics_dict(y_test, pred_prob, pred_label)


def grouped_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return next(gss.split(X, y, groups))


def classify_direction(coef: float | None, expected: str) -> tuple[bool | None, str]:
    if coef is None or pd.isna(coef):
        return None, "not_estimated"
    sign = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
    if expected == "control_only":
        return True, "year control only"
    if expected == "negative":
        return coef < 0, sign
    if expected == "positive":
        return coef > 0, sign
    if expected == "negative_or_mixed":
        return True, sign
    if expected == "positive_or_mixed":
        return True, sign
    if expected == "mixed":
        return True, sign
    return None, sign


def interpretation_note(variable: str, coef: float | None) -> tuple[str, str]:
    if variable == "family_care_index_v1" and coef is not None and coef > 0:
        return (
            "Positive sign likely reflects family structure or intergenerational support proxy rather than care burden promoting labor.",
            "high",
        )
    if variable == "economic_pressure_index_v1" and coef is not None and coef > 0:
        return (
            "Positive sign is more consistent with economic pressure pushing continued labor, not retirement-delay suitability.",
            "medium",
        )
    if variable == "year_2020":
        return ("Year control only; do not interpret as substantive policy effect.", "high")
    if variable == "depression_high" and coef is not None and coef > 0:
        return ("Unexpected positive sign; recheck coding and selection effects before substantive interpretation.", "high")
    if variable == "total_cognition_w" and coef is not None and coef < 0:
        return ("Unexpected negative sign; possible coding, sample composition, or nonlinear selection issue.", "high")
    if variable == "totmet_w":
        return ("May partly proxy labor intensity itself; avoid relying on it as a substantive explanatory feature.", "high")
    if variable in {"poor_health", "adl_limit", "iadl_limit", "care_elder_or_disabled", "log_medical_expense_w"}:
        return ("Direction is broadly interpretable as a labor-participation constraint.", "low")
    return ("Interpret with ordinary caution; combine with baseline and robustness results.", "medium")


def summarize_descriptive(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        if var not in df.columns:
            continue
        series = safe_numeric(df[var])
        rows.append(
            {
                "variable": var,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "n": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logit_df = prepare_logit_df(pd.read_csv(LOGIT_DATA, low_memory=False))
    ml_df = pd.read_csv(ML_DATA, low_memory=False)
    logit_summary = pd.read_excel(LOGIT_RESULTS, sheet_name="model_summary")
    logit_m5 = pd.read_excel(LOGIT_RESULTS, sheet_name="Model_5")
    probit_summary = pd.read_excel(PROBIT_RESULTS, sheet_name="summary")
    probit_m4 = pd.read_excel(PROBIT_RESULTS, sheet_name="Probit_Model_4")
    baseline_metrics = pd.read_excel(ML_METRICS)
    baseline_cv = pd.read_excel(GROUP_CV_METRICS, sheet_name="summary")

    direction_rows = []
    probit_lookup = probit_m4.set_index("variable").to_dict("index") if not probit_m4.empty else {}
    logit_lookup = logit_m5.set_index("variable").to_dict("index") if not logit_m5.empty else {}
    for variable in FOCUS_VARS:
        logit_row = logit_lookup.get(variable, {})
        probit_row = probit_lookup.get(variable, {})
        logit_coef = logit_row.get("coef", np.nan)
        probit_coef = probit_row.get("coef", np.nan)
        expected = EXPECTED_DIRECTION.get(variable, "mixed")
        consistent, _sign = classify_direction(logit_coef if pd.notna(logit_coef) else None, expected)
        note, risk = interpretation_note(variable, logit_coef if pd.notna(logit_coef) else None)
        if pd.notna(logit_coef) and pd.notna(probit_coef):
            sign_consistent = np.sign(logit_coef) == np.sign(probit_coef)
            direction_consistent = bool(consistent) and bool(sign_consistent) if consistent is not None else bool(sign_consistent)
        else:
            direction_consistent = consistent if consistent is not None else False
        direction_rows.append(
            {
                "variable": variable,
                "expected_direction": expected,
                "logit_coef": logit_coef,
                "logit_p_value": logit_row.get("p_value", np.nan),
                "logit_odds_ratio": logit_row.get("odds_ratio", np.nan),
                "probit_coef": probit_coef,
                "probit_p_value": probit_row.get("p_value", np.nan),
                "direction_consistent": direction_consistent,
                "interpretation_note": note,
                "risk_level": risk,
            }
        )
    direction_df = pd.DataFrame(direction_rows)
    direction_path = OUTPUT_DIR / "direction_review_v1.xlsx"
    with pd.ExcelWriter(direction_path, engine="openpyxl") as writer:
        direction_df.to_excel(writer, sheet_name="direction_review", index=False)

    base_vars = [
        "age", "age_squared", "female", "married", "urban", "edu_1", "edu_2", "edu_3", "edu_4", "year_2020",
        "poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w",
        "family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "hchild", "family_size",
        "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "medical_burden_w",
        "smokev", "drinkl", "exercise",
    ]
    robustness_specs = {
        "R1_no_behavior": {
            "vars": [v for v in base_vars if v not in {"smokev", "drinkl", "exercise"}],
            "mask": None,
            "note": "Remove behavior controls.",
        },
        "R2_no_family_index": {
            "vars": [v for v in base_vars if v != "family_care_index_v1"],
            "mask": None,
            "note": "Use family component items without family_care_index_v1.",
        },
        "R3_no_econ_index": {
            "vars": [v for v in base_vars if v != "economic_pressure_index_v1"] + ["log_intergen_support_in_w", "log_intergen_support_out_w"],
            "mask": None,
            "note": "Replace economic_pressure_index_v1 with component variables when available.",
        },
        "R4_no_cognition": {
            "vars": [v for v in base_vars if v != "total_cognition_w"],
            "mask": None,
            "note": "Drop total_cognition_w.",
        },
        "R5_only_2018": {
            "vars": base_vars.copy(),
            "mask": safe_numeric(logit_df["year"]) == 2018,
            "note": "2018 sample only.",
        },
        "R6_only_2020": {
            "vars": base_vars.copy(),
            "mask": safe_numeric(logit_df["year"]) == 2020,
            "note": "2020 sample only.",
        },
        "R7_age_le_80": {
            "vars": base_vars.copy(),
            "mask": safe_numeric(logit_df["age"]) <= 80,
            "note": "Exclude age above 80.",
        },
        "R8_no_missing_flags": {
            "vars": [v for v in base_vars if not v.endswith("_missing")],
            "mask": None,
            "note": "Drop missing-indicator variables if present.",
        },
    }
    robustness_book: dict[str, pd.DataFrame] = {}
    robustness_summary_rows = []
    direction_summary_rows = []
    for name, spec in robustness_specs.items():
        result_df, meta = fit_logit(logit_df, TARGET, spec["vars"], spec["mask"])
        robustness_book[name] = result_df
        robustness_summary_rows.append(
            {
                "model": name,
                "success": meta["success"],
                "nobs": meta["nobs"],
                "pseudo_r2": meta["pseudo_r2"],
                "AIC": meta["AIC"],
                "BIC": meta["BIC"],
                "used_variables": ", ".join(meta["used_variables"]),
                "missing_variables": ", ".join(meta["missing_variables"]),
                "removed_variables": " | ".join(meta["removed_variables"]),
                "note": spec["note"],
            }
        )
        for variable in ["poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w", "family_care_index_v1", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w"]:
            sub = result_df.loc[result_df["variable"] == variable]
            if sub.empty:
                direction_summary_rows.append({"model": name, "variable": variable, "coef": np.nan, "p_value": np.nan, "odds_ratio": np.nan, "direction": "not_estimated"})
            else:
                coef = float(sub.iloc[0]["coef"])
                direction_summary_rows.append({"model": name, "variable": variable, "coef": coef, "p_value": float(sub.iloc[0]["p_value"]), "odds_ratio": float(sub.iloc[0]["odds_ratio"]), "direction": "positive" if coef > 0 else "negative"})

    robustness_path = OUTPUT_DIR / "logit_robustness_stage3_v1.xlsx"
    with pd.ExcelWriter(robustness_path, engine="openpyxl") as writer:
        pd.DataFrame(robustness_summary_rows).to_excel(writer, sheet_name="summary", index=False)
        for name, df_result in robustness_book.items():
            df_result.to_excel(writer, sheet_name=name[:31], index=False)
    robustness_direction_path = OUTPUT_DIR / "robustness_direction_summary_v1.xlsx"
    pd.DataFrame(direction_summary_rows).to_excel(robustness_direction_path, index=False)

    ml_features = [c for c in ml_df.columns if c not in NON_FEATURE_COLS]
    groups = ml_df[GROUP]
    y = ml_df[TARGET].astype(int)
    X = ml_df[ml_features].copy()
    train_idx, test_idx = grouped_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    def xgb_model() -> XGBClassifier:
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )

    def lgbm_model() -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        )

    robustness_ml_specs = {
        "M1_baseline_all_features": ml_features,
        "M2_drop_totmet_w": [c for c in ml_features if c != "totmet_w"],
        "M3_drop_behavior": [c for c in ml_features if c not in {"smokev", "drinkl", "exercise", "totmet_w"}],
        "M4_drop_missing_indicators": [c for c in ml_features if not c.endswith("_missing")],
        "M5_drop_year2020": [c for c in ml_features if c != "year_2020"],
        "M6_core_features_only": [c for c in [
            "age", "age_squared", "female", "married", "urban", "edu_1", "edu_2", "edu_3", "edu_4", "year_2020",
            "poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w",
            "family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "hchild", "family_size",
            "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "medical_burden_w"
        ] if c in ml_features],
    }
    ml_robustness_rows = []
    for name, cols in robustness_ml_specs.items():
        xgb_metrics = train_tree_model(xgb_model(), X_train[cols], y_train, X_test[cols], y_test)
        lgbm_metrics = train_tree_model(lgbm_model(), X_train[cols], y_train, X_test[cols], y_test)
        ml_robustness_rows.append({"scenario": name, "model": "XGBoost", **xgb_metrics, "feature_count": len(cols)})
        ml_robustness_rows.append({"scenario": name, "model": "LightGBM", **lgbm_metrics, "feature_count": len(cols)})
    ml_robustness_df = pd.DataFrame(ml_robustness_rows)
    ml_robustness_path = OUTPUT_DIR / "ml_robustness_stage3_v1.xlsx"
    ml_robustness_df.to_excel(ml_robustness_path, index=False)

    baseline_fi = pd.read_excel(FEATURE_IMPORTANCE, sheet_name="lightgbm")
    xgb_fi = pd.read_excel(FEATURE_IMPORTANCE, sheet_name="xgboost")
    lgbm_fi = pd.read_excel(FEATURE_IMPORTANCE, sheet_name="lightgbm")
    def role_tag(feature: str) -> str:
        if feature in ML_ROLE_MAP:
            return ML_ROLE_MAP[feature]
        if feature.endswith("_missing"):
            return "missing_indicator"
        if feature == "year_2020":
            return "year_control"
        if feature in {"age", "age_squared", "female", "married", "urban"} or feature.startswith("edu_"):
            return "demographic_control"
        if feature in {"smokev", "drinkl", "exercise", "totmet_w"}:
            return "behavior_control"
        if feature in {"poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w", "hibpe", "diabe", "hearte", "stroke", "arthre", "kidneye", "digeste"}:
            return "health_constraints"
        if feature in {"family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "hchild", "family_size"}:
            return "family_care"
        if feature in {"economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "medical_burden_w", "log_intergen_support_in_w", "log_intergen_support_out_w"}:
            return "economic_pressure"
        return "other"

    importance_review_rows = []
    for model_name, fi_df, value_col in [("XGBoost", xgb_fi.head(30), "importance"), ("LightGBM", lgbm_fi.head(30), "importance")]:
        for row in fi_df.itertuples(index=False):
            feature = row.feature
            importance_review_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": getattr(row, value_col),
                    "role_tag": role_tag(feature),
                    "note": interpretation_note(feature, None)[0] if feature in {"totmet_w", "family_care_index_v1", "economic_pressure_index_v1", "year_2020"} or feature.endswith("_missing") else "",
                }
            )
    importance_review_df = pd.DataFrame(importance_review_rows)
    importance_review_path = OUTPUT_DIR / "feature_importance_review_v1.xlsx"
    importance_review_df.to_excel(importance_review_path, index=False)

    descriptive_vars = ["labor_participation", "age", "female", "married", "urban", "poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w", "family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "smokev", "drinkl", "exercise"]
    table1 = summarize_descriptive(logit_df, descriptive_vars)
    table1_path = OUTPUT_DIR / "paper_table1_descriptive_stats_v1.xlsx"
    table1.to_excel(table1_path, index=False)

    group_rows = []
    temp = logit_df.copy()
    temp["age_group"] = pd.cut(safe_numeric(temp["age"]), bins=[44, 49, 54, 59, 64, 69, 120], labels=["45-49", "50-54", "55-59", "60-64", "65-69", "70+"])
    temp["family_care_high"] = np.where(safe_numeric(temp["family_care_index_v1"]) >= safe_numeric(temp["family_care_index_v1"]).median(), 1, 0)
    temp["economic_pressure_high"] = np.where(safe_numeric(temp["economic_pressure_index_v1"]) >= safe_numeric(temp["economic_pressure_index_v1"]).median(), 1, 0)
    for col in ["year", "age_group", "female", "urban", "poor_health", "family_care_high", "economic_pressure_high"]:
        tab = temp.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
        tab["group_var"] = col
        tab = tab.rename(columns={col: "group_value", "mean": "labor_rate", "count": "n"})
        group_rows.append(tab[["group_var", "group_value", "labor_rate", "n"]])
    table2 = pd.concat(group_rows, ignore_index=True)
    table2_path = OUTPUT_DIR / "paper_table2_labor_rate_by_group_v1.xlsx"
    table2.to_excel(table2_path, index=False)

    paper3_rows = []
    for model_name in ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]:
        dfm = pd.read_excel(LOGIT_RESULTS, sheet_name=model_name)
        for row in dfm.itertuples(index=False):
            paper3_rows.append(
                {
                    "variable": row.variable,
                    "model": model_name,
                    "coef": row.coef,
                    "std_err": row.std_err,
                    "p_value": row.p_value,
                    "odds_ratio": row.odds_ratio,
                    "stars": "***" if row.p_value < 0.01 else "**" if row.p_value < 0.05 else "*" if row.p_value < 0.1 else "",
                }
            )
    paper3 = pd.DataFrame(paper3_rows)
    table3_path = OUTPUT_DIR / "paper_table3_logit_main_results_v1.xlsx"
    with pd.ExcelWriter(table3_path, engine="openpyxl") as writer:
        paper3.to_excel(writer, sheet_name="long_format", index=False)
        paper3.pivot(index="variable", columns="model", values="odds_ratio").to_excel(writer, sheet_name="odds_ratio_pivot")

    paper4_path = OUTPUT_DIR / "paper_table4_robustness_v1.xlsx"
    with pd.ExcelWriter(paper4_path, engine="openpyxl") as writer:
        pd.DataFrame(robustness_summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(direction_summary_rows).to_excel(writer, sheet_name="core_direction", index=False)

    table5_rows = baseline_metrics.copy()
    table5_rows["scenario"] = "baseline_test"
    robust_table5 = ml_robustness_df.rename(columns={"scenario": "scenario"})
    paper5 = pd.concat([table5_rows[["scenario", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]], robust_table5[["scenario", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]]], ignore_index=True)
    table5_path = OUTPUT_DIR / "paper_table5_ml_performance_v1.xlsx"
    paper5.to_excel(table5_path, index=False)

    baseline_auc = float(baseline_metrics.loc[baseline_metrics["model"] == "LightGBM", "roc_auc"].iloc[0])
    drop_totmet_auc = float(ml_robustness_df.loc[(ml_robustness_df["scenario"] == "M2_drop_totmet_w") & (ml_robustness_df["model"] == "LightGBM"), "roc_auc"].iloc[0])
    drop_behavior_auc = float(ml_robustness_df.loc[(ml_robustness_df["scenario"] == "M3_drop_behavior") & (ml_robustness_df["model"] == "LightGBM"), "roc_auc"].iloc[0])
    totmet_delta = drop_totmet_auc - baseline_auc
    behavior_delta = drop_behavior_auc - baseline_auc
    totmet_risk = abs(totmet_delta) > 0.03

    logit_probit_consistent = bool((direction_df["direction_consistent"].fillna(False)).mean() >= 0.6)
    stable_core = pd.DataFrame(direction_summary_rows)
    core_vars_to_check = ["poor_health", "chronic_count", "adl_limit", "iadl_limit", "family_care_index_v1", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w"]
    stable_support_count = 0
    for var in core_vars_to_check:
        sub = stable_core.loc[(stable_core["variable"] == var) & (stable_core["direction"] != "not_estimated")]
        if sub.empty:
            continue
        dominant = sub["direction"].mode().iloc[0]
        share = (sub["direction"] == dominant).mean()
        if share >= 0.7:
            stable_support_count += 1
    robustness_supports = stable_support_count >= 5

    recommended_shap_model = "LightGBM" if baseline_auc >= float(baseline_metrics.loc[baseline_metrics["model"] == "XGBoost", "roc_auc"].iloc[0]) else "XGBoost"
    if totmet_risk:
        recommended_feature_set = "去掉 totmet_w 的版本"
    elif abs(behavior_delta) < 0.02:
        recommended_feature_set = "去掉行为变量的版本"
    else:
        recommended_feature_set = "全特征版本"

    report_lines = [
        "Stage 3 Review Report V1",
        f"Baseline results suitable for paper entry: {'Yes' if baseline_auc >= 0.70 else 'No'}",
        f"Logit and Probit directions broadly consistent: {'Yes' if logit_probit_consistent else 'No'}",
        f"Stable variables across robustness checks: {stable_support_count} of {len(core_vars_to_check)} core variables show direction stability.",
        "High-risk interpretation variables:",
        "  family_care_index_v1: continue only as family care and intergenerational support pressure proxy, not pure care burden.",
        "  total_cognition_w and depression_high: recheck coding and discuss selection/composition risk.",
        "  year_2020: keep only as year control, not a substantive policy result.",
        f"family_care_index_v1 usage recommendation: {'Can continue with careful proxy wording.' if 'family_care_index_v1' in direction_df['variable'].values else 'Unavailable.'}",
        "Recommended wording: use '家庭照料与代际支持压力' instead of directly claiming pure 家庭照料责任.",
        f"totmet_w SHAP recommendation: {'Do not place totmet_w in the main SHAP specification without sensitivity comparison.' if totmet_risk else 'Can be tested, but still label it as possible proxy risk.'}",
        "year_2020 recommendation: control only.",
        f"ML stability assessment: {'Stable' if baseline_auc >= 0.70 and abs(drop_behavior_auc - baseline_auc) < 0.03 else 'Needs caution'}",
        f"Proceed to formal SHAP: {'Yes' if baseline_auc >= 0.70 else 'No'}",
        f"Recommended SHAP model: {recommended_shap_model}",
        f"Recommended SHAP feature set: {recommended_feature_set}",
    ]
    if totmet_risk:
        report_lines.append("totmet_w likely carries strong predictive power with interpretation risk because removing it changes AUC by more than 0.03.")
    report_lines.append(f"LightGBM AUC change after dropping totmet_w: {totmet_delta:.4f}")
    report_lines.append(f"LightGBM AUC change after dropping behavior variables: {behavior_delta:.4f}")
    report_lines.append(f"Robustness regressions support main conclusion: {'Yes' if robustness_supports else 'Partially'}")
    report_path = OUTPUT_DIR / "stage3_review_report_v1.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Baseline AUC reasonable: {'Yes' if 0.70 <= baseline_auc <= 0.88 else 'Borderline'}")
    print(f"Logit/Probit broadly consistent: {'Yes' if logit_probit_consistent else 'No'}")
    family_row = direction_df.loc[direction_df["variable"] == "family_care_index_v1"].iloc[0]
    econ_row = direction_df.loc[direction_df["variable"] == "economic_pressure_index_v1"].iloc[0]
    print(f"family_care_index_v1 direction and advice: coef={family_row['logit_coef']:.4f}; {family_row['interpretation_note']}")
    print(f"economic_pressure_index_v1 direction and advice: coef={econ_row['logit_coef']:.4f}; {econ_row['interpretation_note']}")
    print(f"LightGBM AUC change after dropping totmet_w: {totmet_delta:.4f}")
    print(f"LightGBM AUC change after dropping behavior variables: {behavior_delta:.4f}")
    print(f"Robustness regressions support main conclusions: {'Yes' if robustness_supports else 'Partially'}")
    print(f"Recommended SHAP model: {recommended_shap_model}")
    print("Recommended main Logit model for the paper: Model 5 as baseline main specification, with R1-R4 as interpretation-focused robustness checks")
    print(f"Ready to start SHAP next: {'Yes' if baseline_auc >= 0.70 else 'No'}")


if __name__ == "__main__":
    main()
