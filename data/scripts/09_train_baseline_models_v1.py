from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "outputs" / "final_model_data_v1"
OUTPUT_DIR = ROOT / "outputs" / "model_training_v1"

AUDIT_REPORT = FINAL_DIR / "final_leakage_audit_v1" / "final_leakage_audit_report_v1.txt"
LOGIT_FILE = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv"
ML_FILE = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"

TARGET_COL = "labor_participation"
GROUP_COL = "ID"
NON_FEATURE_COLS = {GROUP_COL, "wave", "year", TARGET_COL}
LEAKAGE_KEYWORDS = ["labor", "work", "retire", "job", "employment", "employed", "pension", "suitability", "group"]

LOGIT_MODELS = {
    "Model_1": ["age", "age_squared", "female", "married", "urban", "edu_1", "edu_2", "edu_3", "edu_4", "year_2020"],
    "Model_2": ["poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w"],
    "Model_3": ["family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "hchild", "family_size"],
    "Model_4": ["economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "medical_burden_w"],
    "Model_5": ["smokev", "drinkl", "exercise"],
}
ROBUSTNESS_CORE_VARS = [
    "age",
    "age_squared",
    "female",
    "married",
    "urban",
    "edu_1",
    "edu_2",
    "edu_3",
    "edu_4",
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
    "hchild",
    "family_size",
    "economic_pressure_index_v1",
    "log_hhcperc_v1_w",
    "log_medical_expense_w",
    "medical_burden_w",
    "smokev",
    "drinkl",
    "exercise",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_audit_status() -> tuple[str, str]:
    text = AUDIT_REPORT.read_text(encoding="utf-8", errors="replace")
    final_line = ""
    for line in text.splitlines():
        if "Final conclusion:" in line:
            final_line = line.strip()
    status = "FAIL"
    if "PASS" in final_line:
        status = "PASS"
    return status, text


def leakage_match(column: str) -> str:
    lower = column.lower()
    for keyword in LEAKAGE_KEYWORDS:
        if keyword in lower:
            return keyword
    return ""


def prepare_logit_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edu" in out.columns:
        edu_values = sorted([int(v) for v in safe_numeric(out["edu"]).dropna().unique().tolist()])
        for value in edu_values:
            col = f"edu_{value}"
            if col not in out.columns:
                out[col] = np.where(safe_numeric(out["edu"]) == value, 1, 0).astype(int)
    return out


def build_metrics(y_true: pd.Series, pred_prob: np.ndarray, pred_label: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred_prob)),
    }


def fit_discrete_model(
    df: pd.DataFrame,
    target: str,
    variables: list[str],
    groups: pd.Series | None,
    model_type: str,
) -> tuple[pd.DataFrame, dict[str, object], list[str], list[str]]:
    available = [var for var in variables if var in df.columns]
    missing = [var for var in variables if var not in df.columns]
    working = available.copy()
    removed: list[str] = []
    fit_used_cluster = False

    for var in available.copy():
        if safe_numeric(df[var]).nunique(dropna=True) <= 1:
            working.remove(var)
            removed.append(f"{var}: dropped for zero variance")

    while working:
        x = df[working].apply(safe_numeric)
        x = x.loc[:, ~x.columns.duplicated()].copy()
        if x.empty:
            break
        rank = np.linalg.matrix_rank(x.to_numpy())
        if rank < x.shape[1]:
            to_drop = x.columns[-1]
            working.remove(to_drop)
            removed.append(f"{to_drop}: dropped for singularity/rank deficiency")
            continue
        x_const = sm.add_constant(x, has_constant="add")
        y = safe_numeric(df[target]).astype(int)
        try:
            model = Logit(y, x_const) if model_type == "logit" else Probit(y, x_const)
            try:
                if groups is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
                    fit_used_cluster = True
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = model.fit(disp=0)
            except Exception:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(disp=0)
                fit_used_cluster = False
            rows = []
            for var in result.params.index:
                if var == "const":
                    continue
                rows.append(
                    {
                        "variable": var,
                        "coef": float(result.params[var]),
                        "std_err": float(result.bse[var]),
                        "z": float(result.tvalues[var]),
                        "p_value": float(result.pvalues[var]),
                        "odds_ratio": float(np.exp(result.params[var])) if model_type == "logit" else np.nan,
                        "nobs": int(result.nobs),
                        "pseudo_r2": float(getattr(result, "prsquared", np.nan)),
                        "AIC": float(result.aic),
                        "BIC": float(result.bic),
                    }
                )
            stats = {
                "success": True,
                "used_variables": working.copy(),
                "missing_variables": missing,
                "removed_variables": removed,
                "nobs": int(result.nobs),
                "pseudo_r2": float(getattr(result, "prsquared", np.nan)),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "converged": bool(result.mle_retvals.get("converged", True)) if hasattr(result, "mle_retvals") else True,
                "used_cluster_se": fit_used_cluster,
            }
            return pd.DataFrame(rows), stats, working.copy(), missing
        except Exception as exc:
            to_drop = working[-1]
            working.remove(to_drop)
            removed.append(f"{to_drop}: dropped after {type(exc).__name__}")

    return (
        pd.DataFrame(columns=["variable", "coef", "std_err", "z", "p_value", "odds_ratio", "nobs", "pseudo_r2", "AIC", "BIC"]),
        {
            "success": False,
            "used_variables": [],
            "missing_variables": missing,
            "removed_variables": removed,
            "nobs": 0,
            "pseudo_r2": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "converged": False,
            "used_cluster_se": False,
        },
        [],
        missing,
    )


def compute_vif(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    if not variables:
        return pd.DataFrame(columns=["variable", "VIF"])
    x = df[variables].apply(safe_numeric)
    x = x.loc[:, ~x.columns.duplicated()]
    rows = []
    for i, column in enumerate(x.columns):
        vif = variance_inflation_factor(x.values, i)
        rows.append({"variable": column, "VIF": float(vif)})
    return pd.DataFrame(rows)


def group_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def train_lr_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    pred_prob = pipeline.predict_proba(X_test)[:, 1]
    pred_label = pipeline.predict(X_test)
    metrics = build_metrics(y_test, pred_prob, pred_label)
    coefs = pipeline.named_steps["model"].coef_[0]
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return metrics, confusion_matrix(y_test, pred_label), importance


def train_tree_baseline(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    model.fit(X_train, y_train)
    pred_prob = model.predict_proba(X_test)[:, 1]
    pred_label = model.predict(X_test)
    metrics = build_metrics(y_test, pred_prob, pred_label)
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return metrics, confusion_matrix(y_test, pred_label), importance


def run_group_cv(model_factory, X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred_label = model.predict(X.iloc[test_idx])
        metrics = build_metrics(y.iloc[test_idx], pred_prob, pred_label)
        metrics["model"] = model_name
        metrics["fold"] = fold
        rows.append(metrics)
    return pd.DataFrame(rows)


def auc_flag(value: float) -> str:
    if value > 0.90:
        return "possible_leakage"
    if value >= 0.70:
        return "reasonable"
    if value < 0.65:
        return "weak_prediction"
    return "borderline"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_status, audit_text = read_audit_status()
    if audit_status != "PASS":
        print("Audit report conclusion: FAIL")
        print(audit_text)
        raise SystemExit(1)

    logit_df = prepare_logit_df(pd.read_csv(LOGIT_FILE, low_memory=False))
    ml_df = pd.read_csv(ML_FILE, low_memory=False)

    if GROUP_COL not in ml_df.columns:
        raise ValueError("ID column is required for GroupShuffleSplit and GroupKFold.")

    ml_features_all = [col for col in ml_df.columns if col not in NON_FEATURE_COLS]
    removed_leakage_features = []
    cleaned_ml_features = []
    for col in ml_features_all:
        keyword = leakage_match(col)
        if keyword:
            removed_leakage_features.append({"column": col, "keyword": keyword, "reason": "Removed by pre-training leakage keyword screen."})
        else:
            cleaned_ml_features.append(col)

    X_ml = ml_df[cleaned_ml_features].copy()
    y_ml = ml_df[TARGET_COL].astype(int)
    groups = ml_df[GROUP_COL]
    train_idx, test_idx = group_split(X_ml, y_ml, groups)
    X_train = X_ml.iloc[train_idx].copy()
    X_test = X_ml.iloc[test_idx].copy()
    y_train = y_ml.iloc[train_idx].copy()
    y_test = y_ml.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    logit_results = {}
    logit_summary_rows = []
    used_logit_variables = {}
    cumulative = []
    for model_name, variables in LOGIT_MODELS.items():
        cumulative.extend(variables)
        cumulative_unique = []
        for var in cumulative:
            if var not in cumulative_unique:
                cumulative_unique.append(var)
        result_df, stats, used_vars, missing_vars = fit_discrete_model(
            logit_df,
            TARGET_COL,
            cumulative_unique,
            logit_df[GROUP_COL] if GROUP_COL in logit_df.columns else None,
            "logit",
        )
        logit_results[model_name] = result_df
        used_logit_variables[model_name] = used_vars
        logit_summary_rows.append(
            {
                "model": model_name,
                "success": stats["success"],
                "nobs": stats["nobs"],
                "pseudo_r2": stats["pseudo_r2"],
                "AIC": stats["AIC"],
                "BIC": stats["BIC"],
                "converged": stats["converged"],
                "used_cluster_se": stats["used_cluster_se"],
                "used_variables": ", ".join(used_vars),
                "missing_variables": ", ".join(missing_vars),
                "removed_variables": " | ".join(stats["removed_variables"]),
            }
        )

    probit_vars = used_logit_variables.get("Model_4", []).copy()
    probit_df, probit_stats, probit_used, _ = fit_discrete_model(
        logit_df,
        TARGET_COL,
        probit_vars,
        logit_df[GROUP_COL] if GROUP_COL in logit_df.columns else None,
        "probit",
    )

    vif_df = compute_vif(logit_df, used_logit_variables.get("Model_5", []))

    lr_metrics, lr_conf, lr_importance = train_lr_baseline(X_train, y_train, X_test, y_test)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    xgb_metrics, xgb_conf, xgb_importance = train_tree_baseline(xgb_model, X_train, y_train, X_test, y_test)

    lgbm_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    lgbm_metrics, lgbm_conf, lgbm_importance = train_tree_baseline(lgbm_model, X_train, y_train, X_test, y_test)

    ml_metrics_df = pd.DataFrame(
        [
            {"model": "LogisticRegression", **lr_metrics},
            {"model": "XGBoost", **xgb_metrics},
            {"model": "LightGBM", **lgbm_metrics},
        ]
    )

    confusion_sheets = {
        "LogisticRegression": pd.DataFrame(lr_conf, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]),
        "XGBoost": pd.DataFrame(xgb_conf, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]),
        "LightGBM": pd.DataFrame(lgbm_conf, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]),
    }

    cv_xgb = run_group_cv(
        lambda: XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        ),
        X_ml,
        y_ml,
        groups,
        "XGBoost",
    )
    cv_lgbm = run_group_cv(
        lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        ),
        X_ml,
        y_ml,
        groups,
        "LightGBM",
    )
    cv_metrics = pd.concat([cv_xgb, cv_lgbm], ignore_index=True)
    cv_summary = cv_metrics.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().reset_index()

    importance_path = OUTPUT_DIR / "feature_importance_baseline_v1.xlsx"
    with pd.ExcelWriter(importance_path, engine="openpyxl") as writer:
        lr_importance.to_excel(writer, sheet_name="logistic_regression", index=False)
        xgb_importance.to_excel(writer, sheet_name="xgboost", index=False)
        lgbm_importance.to_excel(writer, sheet_name="lightgbm", index=False)

    robustness_rows = []
    robustness_notes = []

    no_year_features = [c for c in cleaned_ml_features if c != "year_2020"]
    for model_name, model in [
        ("XGBoost_no_year2020", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=4
        )),
        ("LightGBM_no_year2020", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbosity=-1
        )),
    ]:
        model.fit(X_train[no_year_features], y_train)
        pred_prob = model.predict_proba(X_test[no_year_features])[:, 1]
        pred_label = model.predict(X_test[no_year_features])
        robustness_rows.append({"scenario": model_name, **build_metrics(y_test, pred_prob, pred_label)})

    no_missing_features = [c for c in cleaned_ml_features if not c.endswith("_missing")]
    for model_name, model in [
        ("XGBoost_no_missing_indicators", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=4
        )),
        ("LightGBM_no_missing_indicators", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbosity=-1
        )),
    ]:
        model.fit(X_train[no_missing_features], y_train)
        pred_prob = model.predict_proba(X_test[no_missing_features])[:, 1]
        pred_label = model.predict(X_test[no_missing_features])
        robustness_rows.append({"scenario": model_name, **build_metrics(y_test, pred_prob, pred_label)})

    core_features = [c for c in ROBUSTNESS_CORE_VARS if c in cleaned_ml_features]
    skipped_core = [c for c in ROBUSTNESS_CORE_VARS if c not in cleaned_ml_features]
    robustness_notes.append({"item": "core_features_skipped", "detail": ", ".join(skipped_core)})
    core_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    core_model.fit(X_train[core_features], y_train)
    core_prob = core_model.predict_proba(X_test[core_features])[:, 1]
    core_pred = core_model.predict(X_test[core_features])
    robustness_rows.append({"scenario": "LightGBM_core_features_only", **build_metrics(y_test, core_prob, core_pred)})
    robustness_df = pd.DataFrame(robustness_rows)

    auc_flags_df = ml_metrics_df.copy()
    auc_flags_df["auc_flag"] = auc_flags_df["roc_auc"].apply(auc_flag)

    top30_if_leakage = pd.DataFrame(columns=["model", "feature", "importance"])
    if (auc_flags_df["auc_flag"] == "possible_leakage").any():
        sheets = []
        for model_name, imp_df, col_name in [
            ("LogisticRegression", lr_importance.head(30).rename(columns={"abs_coefficient": "importance"}), "importance"),
            ("XGBoost", xgb_importance.head(30), "importance"),
            ("LightGBM", lgbm_importance.head(30), "importance"),
        ]:
            temp = imp_df.copy()
            if col_name not in temp.columns and "abs_coefficient" in temp.columns:
                temp = temp.rename(columns={"abs_coefficient": "importance"})
            temp["model"] = model_name
            top30_if_leakage = pd.concat([top30_if_leakage, temp[["model", "feature", "importance"]]], ignore_index=True)

    best_model_name = ml_metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_importance = {
        "LogisticRegression": lr_importance.rename(columns={"abs_coefficient": "importance"}),
        "XGBoost": xgb_importance,
        "LightGBM": lgbm_importance,
    }[best_model_name]
    top20_features = best_importance.head(20)

    diagnostics_overview = pd.DataFrame(
        [
            {"item": "audit_report_status", "value": audit_status},
            {"item": "logit_rows", "value": logit_df.shape[0]},
            {"item": "logit_columns", "value": logit_df.shape[1]},
            {"item": "ml_rows", "value": ml_df.shape[0]},
            {"item": "ml_columns", "value": ml_df.shape[1]},
            {"item": "train_rows", "value": len(train_idx)},
            {"item": "test_rows", "value": len(test_idx)},
            {"item": "train_ids", "value": groups_train.nunique()},
            {"item": "test_ids", "value": groups_test.nunique()},
            {"item": "group_splitter", "value": "GroupShuffleSplit(random_state=42, test_size=0.2)"},
        ]
    )
    target_distribution = (
        y_ml.value_counts()
        .rename_axis("target")
        .reset_index(name="count")
    )
    target_distribution["share"] = target_distribution["count"] / len(y_ml)
    removed_features_df = pd.DataFrame(removed_leakage_features if removed_leakage_features else [{"column": "", "keyword": "", "reason": "No leakage-keyword feature removed before ML training."}])
    leakage_check_df = pd.DataFrame(
        [
            {"check": "ml_removed_leakage_keyword_features", "value": len(removed_leakage_features)},
            {"check": "possible_leakage_auc_models", "value": int((auc_flags_df["auc_flag"] == "possible_leakage").sum())},
            {"check": "reasonable_auc_models", "value": int((auc_flags_df["auc_flag"] == "reasonable").sum())},
            {"check": "weak_prediction_models", "value": int((auc_flags_df["auc_flag"] == "weak_prediction").sum())},
        ]
    )

    logit_feature_list_rows = []
    for model_name, vars_used in used_logit_variables.items():
        for var in vars_used:
            logit_feature_list_rows.append({"model": model_name, "variable": var})
    logit_feature_list_df = pd.DataFrame(logit_feature_list_rows)

    diagnostics_path = OUTPUT_DIR / "model_diagnostics_v1.xlsx"
    with pd.ExcelWriter(diagnostics_path, engine="openpyxl") as writer:
        diagnostics_overview.to_excel(writer, sheet_name="data_overview", index=False)
        target_distribution.to_excel(writer, sheet_name="target_distribution", index=False)
        pd.DataFrame(
            [{"feature": c} for c in cleaned_ml_features]
        ).to_excel(writer, sheet_name="ml_features", index=False)
        logit_feature_list_df.to_excel(writer, sheet_name="logit_features", index=False)
        removed_features_df.to_excel(writer, sheet_name="removed_features", index=False)
        leakage_check_df.to_excel(writer, sheet_name="leakage_check", index=False)
        auc_flags_df.to_excel(writer, sheet_name="auc_flags", index=False)
        pd.DataFrame(logit_summary_rows).to_excel(writer, sheet_name="logit_model_status", index=False)
        top30_if_leakage.to_excel(writer, sheet_name="top30_if_leakage", index=False)

    logit_results_path = OUTPUT_DIR / "logit_results_v1.xlsx"
    with pd.ExcelWriter(logit_results_path, engine="openpyxl") as writer:
        pd.DataFrame(logit_summary_rows).to_excel(writer, sheet_name="model_summary", index=False)
        for model_name, result_df in logit_results.items():
            result_df.to_excel(writer, sheet_name=model_name[:31], index=False)

    probit_results_path = OUTPUT_DIR / "probit_results_v1.xlsx"
    with pd.ExcelWriter(probit_results_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "model": "Probit_Model_4",
                    "success": probit_stats["success"],
                    "nobs": probit_stats["nobs"],
                    "pseudo_r2": probit_stats["pseudo_r2"],
                    "AIC": probit_stats["AIC"],
                    "BIC": probit_stats["BIC"],
                    "converged": probit_stats["converged"],
                    "used_cluster_se": probit_stats["used_cluster_se"],
                    "used_variables": ", ".join(probit_used),
                    "missing_variables": ", ".join([v for v in LOGIT_MODELS["Model_1"] + LOGIT_MODELS["Model_2"] + LOGIT_MODELS["Model_3"] + LOGIT_MODELS["Model_4"] if v not in probit_used]),
                    "removed_variables": " | ".join(probit_stats["removed_variables"]),
                }
            ]
        ).to_excel(writer, sheet_name="summary", index=False)
        probit_df.to_excel(writer, sheet_name="Probit_Model_4", index=False)

    vif_path = OUTPUT_DIR / "logit_vif_v1.xlsx"
    with pd.ExcelWriter(vif_path, engine="openpyxl") as writer:
        vif_df.to_excel(writer, sheet_name="Model_5_VIF", index=False)

    ml_metrics_path = OUTPUT_DIR / "ml_metrics_v1.xlsx"
    with pd.ExcelWriter(ml_metrics_path, engine="openpyxl") as writer:
        ml_metrics_df.to_excel(writer, sheet_name="test_metrics", index=False)

    confusion_path = OUTPUT_DIR / "confusion_matrices_v1.xlsx"
    with pd.ExcelWriter(confusion_path, engine="openpyxl") as writer:
        for sheet_name, cm_df in confusion_sheets.items():
            cm_df.to_excel(writer, sheet_name=sheet_name[:31])

    cv_path = OUTPUT_DIR / "group_cv_metrics_v1.xlsx"
    with pd.ExcelWriter(cv_path, engine="openpyxl") as writer:
        cv_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        cv_summary.to_excel(writer, sheet_name="summary", index=False)

    robustness_path = OUTPUT_DIR / "robustness_metrics_v1.xlsx"
    with pd.ExcelWriter(robustness_path, engine="openpyxl") as writer:
        robustness_df.to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(robustness_notes).to_excel(writer, sheet_name="notes", index=False)

    report_path = OUTPUT_DIR / "baseline_model_report_v1.txt"
    logit_m5 = logit_results.get("Model_5", pd.DataFrame())
    significant_m5 = logit_m5.loc[logit_m5["p_value"] < 0.05, ["variable", "coef", "odds_ratio", "p_value"]] if not logit_m5.empty else pd.DataFrame()
    consistent_vars = []
    if not probit_df.empty and not logit_results.get("Model_4", pd.DataFrame()).empty:
        merged = logit_results["Model_4"][["variable", "coef", "p_value"]].merge(
            probit_df[["variable", "coef", "p_value"]],
            on="variable",
            suffixes=("_logit", "_probit"),
        )
        for row in merged.itertuples(index=False):
            if np.sign(row.coef_logit) == np.sign(row.coef_probit):
                consistent_vars.append(row.variable)

    lines = [
        "Baseline Model Report V1",
        f"Data files: {LOGIT_FILE} | {ML_FILE}",
        f"Audit report PASS: {audit_status}",
        f"Final sample size: logit={logit_df.shape[0]}, ml={ml_df.shape[0]}",
        "Target distribution:",
    ]
    for row in target_distribution.itertuples(index=False):
        lines.append(f"  {row.target}: {row.count} ({row.share:.4f})")
    lines.append("Split method: GroupShuffleSplit with group=ID, test_size=0.2, random_state=42")
    lines.append("Logit Model 5 main results:")
    if significant_m5.empty:
        lines.append("  No variable reached p<0.05 in Model 5, or Model 5 did not fit successfully.")
    else:
        for row in significant_m5.itertuples(index=False):
            lines.append(f"  {row.variable}: coef={row.coef:.4f}, OR={row.odds_ratio:.4f}, p={row.p_value:.4g}")
    lines.append(
        f"Probit consistency with Logit Model 4: {'Broadly consistent' if consistent_vars else 'Unable to confirm consistency or no overlapping stable signs.'}"
    )
    lines.append(
        f"XGBoost test metrics: AUC={xgb_metrics['roc_auc']:.4f}, F1={xgb_metrics['f1']:.4f}, Accuracy={xgb_metrics['accuracy']:.4f}"
    )
    lines.append(
        f"LightGBM test metrics: AUC={lgbm_metrics['roc_auc']:.4f}, F1={lgbm_metrics['f1']:.4f}, Accuracy={lgbm_metrics['accuracy']:.4f}"
    )
    lines.append(f"Best-performing model by test AUC: {best_model_name}")
    lines.append("Top 20 important features:")
    for row in top20_features.itertuples(index=False):
        importance_value = row.importance if hasattr(row, "importance") else row.abs_coefficient
        lines.append(f"  {row.feature}: {importance_value:.6f}")
    suspicious_flag = (auc_flags_df["auc_flag"] == "possible_leakage").any() or len(removed_leakage_features) > 0
    lines.append(f"Possible leakage detected: {'Yes' if suspicious_flag else 'No'}")
    lines.append(f"Recommendation on SHAP: {'Proceed to SHAP analysis.' if best_model_name in ['XGBoost', 'LightGBM'] and not suspicious_flag else 'Delay SHAP until leakage or model-stability concerns are reviewed.'}")
    lines.append("Interpretation risks for the paper:")
    lines.append("  year_2020 should be treated as a control, not a substantive explanatory variable.")
    lines.append("  Family and economic pressure variables partly capture proxy burden rather than direct causal mechanisms.")
    lines.append("  Baseline ML importance does not identify causality; it only summarizes predictive contribution.")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    output_files = [
        logit_results_path,
        probit_results_path,
        vif_path,
        ml_metrics_path,
        confusion_path,
        cv_path,
        importance_path,
        robustness_path,
        diagnostics_path,
        report_path,
    ]

    print(f"Audit report conclusion: {audit_status}")
    print(f"Logit data shape: {logit_df.shape[0]} rows x {logit_df.shape[1]} columns")
    print(f"ML data shape: {ml_df.shape[0]} rows x {ml_df.shape[1]} columns")
    print("Target distribution:")
    for row in target_distribution.itertuples(index=False):
        print(f"  {row.target}: {row.count} ({row.share:.4f})")
    print(f"Train/Test sample size: {len(train_idx)} / {len(test_idx)}")
    print(f"Train/Test ID count: {groups_train.nunique()} / {groups_test.nunique()}")
    print(f"Logit Model 5 success: {next((row['success'] for row in logit_summary_rows if row['model']=='Model_5'), False)}")
    print(f"Probit Model 4 success: {probit_stats['success']}")
    print(f"LogisticRegression test AUC / F1: {lr_metrics['roc_auc']:.4f} / {lr_metrics['f1']:.4f}")
    print(f"XGBoost test AUC / F1: {xgb_metrics['roc_auc']:.4f} / {xgb_metrics['f1']:.4f}")
    print(f"LightGBM test AUC / F1: {lgbm_metrics['roc_auc']:.4f} / {lgbm_metrics['f1']:.4f}")
    cv_auc_mean = cv_summary["roc_auc"].mean() if not cv_summary.empty else np.nan
    print(f"GroupKFold mean AUC: {cv_auc_mean:.4f}")
    print(f"Possible leakage detected: {'Yes' if suspicious_flag else 'No'}")
    print("Output files:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
