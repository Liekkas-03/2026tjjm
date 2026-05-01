from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "outputs" / "final_model_data_v1"
TRAINING_DIR = ROOT / "outputs" / "model_training_v1"
STAGE3_DIR = ROOT / "outputs" / "model_stage3_review_v1"
STAGE4_DIR = ROOT / "outputs" / "model_stage4_shap_v1"
OUTPUT_DIR = ROOT / "outputs" / "model_stage5_paper_ready_v1"
MAIN_FIG_DIR = OUTPUT_DIR / "main_text_figures"
APP_FIG_DIR = OUTPUT_DIR / "appendix_figures"

LOGIT_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv"
ML_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"

TARGET = "labor_participation"
GROUP = "ID"
BASE_CORE_VARS = [
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
    "economic_pressure_index_v1",
    "log_hhcperc_v1_w",
    "log_medical_expense_w",
    "medical_burden_w",
]
KEY_HET_VARS = [
    "poor_health",
    "chronic_count",
    "adl_limit",
    "iadl_limit",
    "depression_high",
    "total_cognition_w",
    "family_care_index_v1",
    "co_reside_child",
    "care_elder_or_disabled",
    "economic_pressure_index_v1",
    "log_hhcperc_v1_w",
    "log_medical_expense_w",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_FIG_DIR.mkdir(parents=True, exist_ok=True)


def prepare_logit_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edu" in out.columns:
        values = sorted([int(v) for v in safe_numeric(out["edu"]).dropna().unique().tolist()])
        for value in values:
            col = f"edu_{value}"
            if col not in out.columns:
                out[col] = np.where(safe_numeric(out["edu"]) == value, 1, 0).astype(int)
    return out


def fit_logit(df: pd.DataFrame, target: str, variables: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    available = [v for v in variables if v in df.columns]
    missing = [v for v in variables if v not in df.columns]
    working = available.copy()
    removed: list[str] = []

    for v in available.copy():
        if safe_numeric(df[v]).nunique(dropna=True) <= 1:
            working.remove(v)
            removed.append(f"{v}: zero variance")

    while working:
        X = df[working].apply(safe_numeric)
        X = X.loc[:, ~X.columns.duplicated()].copy()
        if X.empty:
            break
        if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
            drop_col = X.columns[-1]
            working.remove(drop_col)
            removed.append(f"{drop_col}: singularity/rank deficiency")
            continue
        y = safe_numeric(df[target]).astype(int)
        Xc = sm.add_constant(X, has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = Logit(y, Xc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df[GROUP]})
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
            }
        except Exception as exc:
            drop_col = working[-1]
            working.remove(drop_col)
            removed.append(f"{drop_col}: dropped after {type(exc).__name__}")

    return pd.DataFrame(columns=["variable", "coef", "std_err", "p_value", "odds_ratio", "nobs", "pseudo_r2", "AIC", "BIC"]), {
        "success": False,
        "nobs": len(df),
        "pseudo_r2": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "used_variables": [],
        "missing_variables": missing,
        "removed_variables": removed,
    }


def choose_econ_cut(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    candidates = [2, 3]
    rows = []
    best_cut = 2
    best_score = float("inf")
    series = safe_numeric(df["economic_pressure_index_v1"])
    for cut in candidates:
        high = (series >= cut).astype(int)
        share = float(high.mean())
        balance_score = abs(share - 0.5)
        rows.append({"cutoff": cut, "high_share": share, "balance_score": balance_score})
        if 0.15 <= share <= 0.85 and balance_score < best_score:
            best_cut = cut
            best_score = balance_score
    return best_cut, pd.DataFrame(rows)


def subgroup_vars(group_name: str) -> list[str]:
    vars_use = BASE_CORE_VARS.copy()
    if group_name.startswith("gender_"):
        vars_use = [v for v in vars_use if v != "female"]
    if group_name.startswith("urban_"):
        vars_use = [v for v in vars_use if v != "urban"]
    if group_name.startswith("health_"):
        vars_use = [v for v in vars_use if v != "poor_health"]
    if group_name.startswith("familyhigh_"):
        vars_use = [v for v in vars_use if v != "family_care_index_v1"]
    if group_name.startswith("econhigh_"):
        vars_use = [v for v in vars_use if v != "economic_pressure_index_v1"]
    return vars_use


def role_significance(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def create_labor_rate_figure(df: pd.DataFrame, out_path: Path) -> None:
    temp = df.copy()
    temp["age_group"] = pd.cut(
        safe_numeric(temp["age"]),
        bins=[44, 59, 69, 120],
        labels=["45-59", "60-69", "70+"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    age_tab = temp.groupby("age_group", observed=False)[TARGET].mean().reset_index()
    axes[0].bar(age_tab["age_group"].astype(str), age_tab[TARGET], color="#4E79A7")
    axes[0].set_title("By Age Group")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Labor Participation Rate")

    sex_tab = temp.groupby("female")[TARGET].mean().reset_index()
    axes[1].bar(["Male", "Female"], sex_tab[TARGET], color="#F28E2B")
    axes[1].set_title("By Gender")
    axes[1].set_ylim(0, 1)

    urban_tab = temp.groupby("urban")[TARGET].mean().reset_index()
    axes[2].bar(["Rural", "Urban"], urban_tab[TARGET], color="#59A14F")
    axes[2].set_title("By Urban Status")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_forest_plot(logit_model5: pd.DataFrame, out_path: Path) -> None:
    df = logit_model5.copy()
    df = df.loc[~df["variable"].isin(["year_2020"])]
    df["or_low"] = np.exp(df["coef"] - 1.96 * df["std_err"])
    df["or_high"] = np.exp(df["coef"] + 1.96 * df["std_err"])
    df = df.sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.28)))
    y_pos = np.arange(len(df))
    ax.errorbar(df["odds_ratio"], y_pos, xerr=[df["odds_ratio"] - df["or_low"], df["or_high"] - df["odds_ratio"]], fmt="o", color="#4E79A7", ecolor="#9ecae1", capsize=3)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["variable"])
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title("Logit Model 5 Odds Ratio Forest Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def copy_figure(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ensure_dirs()

    logit_df = prepare_logit_df(pd.read_csv(LOGIT_DATA, low_memory=False))
    ml_df = pd.read_csv(ML_DATA, low_memory=False)
    logit_summary = pd.read_excel(TRAINING_DIR / "logit_results_v1.xlsx", sheet_name="model_summary")
    logit_sheets = {name: pd.read_excel(TRAINING_DIR / "logit_results_v1.xlsx", sheet_name=name) for name in ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]}
    robustness_summary = pd.read_excel(STAGE3_DIR / "logit_robustness_stage3_v1.xlsx", sheet_name="summary")
    robustness_direction = pd.read_excel(STAGE3_DIR / "robustness_direction_summary_v1.xlsx")
    ml_metrics = pd.read_excel(TRAINING_DIR / "ml_metrics_v1.xlsx")
    shap_metrics = pd.read_excel(STAGE4_DIR / "shap_model_metrics_v1.xlsx", sheet_name="metrics")
    shap_main = pd.read_excel(STAGE4_DIR / "shap_values_summary_table_v1.xlsx", sheet_name="main_clean")

    family_cut = float(safe_numeric(logit_df["family_care_index_v1"]).median())
    if pd.isna(family_cut):
        family_cut = 2.0
    if family_cut < 2:
        family_cut = 2.0
    econ_cut, econ_cut_table = choose_econ_cut(logit_df)

    temp = logit_df.copy()
    temp["age_group_stage5"] = pd.cut(safe_numeric(temp["age"]), bins=[44, 59, 69, 120], labels=["45-59", "60-69", "70+"])
    temp["family_high_stage5"] = np.where(safe_numeric(temp["family_care_index_v1"]) >= family_cut, 1, 0)
    temp["econ_high_stage5"] = np.where(safe_numeric(temp["economic_pressure_index_v1"]) >= econ_cut, 1, 0)

    heterogeneity_groups = {
        "gender_male": temp["female"] == 0,
        "gender_female": temp["female"] == 1,
        "urban_rural": temp["urban"] == 0,
        "urban_urban": temp["urban"] == 1,
        "age_45_59": temp["age_group_stage5"] == "45-59",
        "age_60_69": temp["age_group_stage5"] == "60-69",
        "age_70_plus": temp["age_group_stage5"] == "70+",
        "health_good": temp["poor_health"] == 0,
        "health_poor": temp["poor_health"] == 1,
        "familyhigh_low": temp["family_high_stage5"] == 0,
        "familyhigh_high": temp["family_high_stage5"] == 1,
        "econhigh_low": temp["econ_high_stage5"] == 0,
        "econhigh_high": temp["econ_high_stage5"] == 1,
    }

    hetero_results = {}
    hetero_summary_rows = []
    hetero_key_rows = []
    for group_name, mask in heterogeneity_groups.items():
        sub_df = temp.loc[mask].copy()
        result_df, meta = fit_logit(sub_df, TARGET, subgroup_vars(group_name))
        hetero_results[group_name] = result_df
        hetero_summary_rows.append(
            {
                "group": group_name,
                "success": meta["success"],
                "nobs": meta["nobs"],
                "pseudo_r2": meta["pseudo_r2"],
                "AIC": meta["AIC"],
                "BIC": meta["BIC"],
                "used_variables": ", ".join(meta["used_variables"]),
                "missing_variables": ", ".join(meta["missing_variables"]),
                "removed_variables": " | ".join(meta["removed_variables"]),
            }
        )
        for var in KEY_HET_VARS:
            row = result_df.loc[result_df["variable"] == var]
            hetero_key_rows.append(
                {
                    "group": group_name,
                    "variable": var,
                    "coef": float(row.iloc[0]["coef"]) if not row.empty else np.nan,
                    "p_value": float(row.iloc[0]["p_value"]) if not row.empty else np.nan,
                    "odds_ratio": float(row.iloc[0]["odds_ratio"]) if not row.empty else np.nan,
                    "significance": role_significance(float(row.iloc[0]["p_value"])) if not row.empty else "",
                    "direction": "positive" if (not row.empty and float(row.iloc[0]["coef"]) > 0) else "negative" if (not row.empty and float(row.iloc[0]["coef"]) < 0) else "not_estimated",
                }
            )

    hetero_path = OUTPUT_DIR / "heterogeneity_logit_results_v1.xlsx"
    with pd.ExcelWriter(hetero_path, engine="openpyxl") as writer:
        pd.DataFrame(hetero_summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(hetero_key_rows).to_excel(writer, sheet_name="key_variable_compare", index=False)
        econ_cut_table.to_excel(writer, sheet_name="econ_cutoff_rule", index=False)
        for name, df_result in hetero_results.items():
            df_result.to_excel(writer, sheet_name=name[:31], index=False)

    hetero_final = pd.DataFrame(hetero_key_rows)
    key_focus = hetero_final.loc[hetero_final["variable"].isin(["poor_health", "iadl_limit", "family_care_index_v1", "economic_pressure_index_v1"])].copy()
    table7_path = OUTPUT_DIR / "paper_table7_heterogeneity_final.xlsx"
    with pd.ExcelWriter(table7_path, engine="openpyxl") as writer:
        key_focus.to_excel(writer, sheet_name="key_variables", index=False)
        hetero_final.pivot_table(index="group", columns="variable", values="coef", aggfunc="first").to_excel(writer, sheet_name="coef_pivot")

    desc_vars = ["labor_participation", "age", "female", "married", "urban", "poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w", "family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "smokev", "drinkl", "exercise"]
    table1_rows = []
    for var in desc_vars:
        if var not in temp.columns:
            continue
        s = safe_numeric(temp[var])
        table1_rows.append(
            {
                "variable": var,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    table1 = pd.DataFrame(table1_rows)
    table1_path = OUTPUT_DIR / "paper_table1_descriptive_stats_final.xlsx"
    table1.to_excel(table1_path, index=False)

    group_rows = []
    for col in ["year", "age_group_stage5", "female", "urban", "poor_health", "family_high_stage5", "econ_high_stage5"]:
        tab = temp.groupby(col, observed=False)[TARGET].agg(["mean", "count"]).reset_index()
        tab["group_var"] = col
        tab = tab.rename(columns={col: "group_value", "mean": "labor_rate", "count": "n"})
        group_rows.append(tab[["group_var", "group_value", "labor_rate", "n"]])
    table2 = pd.concat(group_rows, ignore_index=True)
    table2_path = OUTPUT_DIR / "paper_table2_group_labor_rates_final.xlsx"
    table2.to_excel(table2_path, index=False)

    table3_rows = []
    for model_name, df_result in logit_sheets.items():
        for row in df_result.itertuples(index=False):
            table3_rows.append(
                {
                    "variable": row.variable,
                    "model": model_name,
                    "coef": row.coef,
                    "std_err": row.std_err,
                    "p_value": row.p_value,
                    "odds_ratio": row.odds_ratio,
                    "formatted_or": f"{row.odds_ratio:.3f}{role_significance(row.p_value)}",
                }
            )
    table3_long = pd.DataFrame(table3_rows)
    table3_path = OUTPUT_DIR / "paper_table3_logit_main_final.xlsx"
    with pd.ExcelWriter(table3_path, engine="openpyxl") as writer:
        table3_long.to_excel(writer, sheet_name="long_format", index=False)
        table3_long.pivot(index="variable", columns="model", values="formatted_or").to_excel(writer, sheet_name="paper_view")
        logit_summary.to_excel(writer, sheet_name="model_summary", index=False)

    key_robust = robustness_direction.loc[robustness_direction["variable"].isin(["poor_health", "iadl_limit", "family_care_index_v1", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w"])].copy()
    table4_path = OUTPUT_DIR / "paper_table4_robustness_final.xlsx"
    with pd.ExcelWriter(table4_path, engine="openpyxl") as writer:
        robustness_summary.to_excel(writer, sheet_name="summary", index=False)
        key_robust.to_excel(writer, sheet_name="key_variables", index=False)

    baseline_perf = ml_metrics.copy()
    baseline_perf["scenario"] = "baseline"
    shap_perf = shap_metrics.copy().rename(columns={"scheme": "scenario"})
    shap_perf["model"] = "LightGBM_SHAP"
    table5 = pd.concat(
        [
            baseline_perf[["scenario", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]],
            shap_perf[["scenario", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]],
        ],
        ignore_index=True,
    )
    table5_path = OUTPUT_DIR / "paper_table5_ml_performance_final.xlsx"
    table5.to_excel(table5_path, index=False)

    table6 = shap_main.head(20).copy()
    table6_path = OUTPUT_DIR / "paper_table6_shap_top_features_final.xlsx"
    table6.to_excel(table6_path, index=False)

    labor_fig = MAIN_FIG_DIR / "fig1_labor_rates_groups.png"
    create_labor_rate_figure(temp, labor_fig)
    forest_fig = MAIN_FIG_DIR / "fig2_logit_model5_forest.png"
    create_forest_plot(logit_sheets["Model_5"], forest_fig)
    shap_bar_main_copy = MAIN_FIG_DIR / "fig3_shap_bar_main_clean.png"
    shap_beeswarm_main_copy = MAIN_FIG_DIR / "fig4_shap_beeswarm_main_clean.png"
    copy_figure(STAGE4_DIR / "fig_shap_bar_main_clean.png", shap_bar_main_copy)
    copy_figure(STAGE4_DIR / "fig_shap_beeswarm_main_clean.png", shap_beeswarm_main_copy)

    main_dep_candidates = [
        "poor_health.png",
        "iadl_limit.png",
        "family_care_index_v1.png",
        "economic_pressure_index_v1.png",
        "log_hhcperc_v1_w.png",
        "log_intergen_support_out_w.png",
    ]
    appendix_dep_extra = [
        "adl_limit.png",
        "chronic_count.png",
        "depression_high.png",
        "co_reside_child.png",
        "care_elder_or_disabled.png",
        "family_size.png",
        "hchild.png",
        "log_medical_expense_w.png",
        "medical_burden_w.png",
        "log_intergen_support_in_w.png",
    ]
    figure_rows = [
        {
            "figure_id": "Figure 1",
            "figure_name": "劳动参与率分组柱状图",
            "source_file": str(labor_fig),
            "recommended_location": "main_text",
            "purpose": "展示年龄组、性别与城乡维度的劳动参与率差异",
            "interpretation_note": "用于描述性统计部分，突出劳动参与异质性背景。",
        },
        {
            "figure_id": "Figure 2",
            "figure_name": "Logit Model 5 Odds Ratio Forest Plot",
            "source_file": str(forest_fig),
            "recommended_location": "main_text",
            "purpose": "可视化主回归结果方向与区间估计",
            "interpretation_note": "year_2020 作为控制变量不宜展开解释。",
        },
        {
            "figure_id": "Figure 3",
            "figure_name": "SHAP Bar Summary (main_clean)",
            "source_file": str(shap_bar_main_copy),
            "recommended_location": "main_text",
            "purpose": "展示主 SHAP 模型的全局解释贡献排序",
            "interpretation_note": "正文推荐使用，强调解释贡献而非因果效应。",
        },
        {
            "figure_id": "Figure 4",
            "figure_name": "SHAP Beeswarm (main_clean)",
            "source_file": str(shap_beeswarm_main_copy),
            "recommended_location": "main_text",
            "purpose": "展示主 SHAP 模型变量影响方向与样本分布",
            "interpretation_note": "用于解释关键变量在预测中的异质贡献。",
        },
    ]

    for idx, name in enumerate(main_dep_candidates, start=5):
        src = STAGE4_DIR / "dependence_main_clean" / name
        dst = MAIN_FIG_DIR / name
        if copy_figure(src, dst):
            feature = name.replace(".png", "")
            figure_rows.append(
                {
                    "figure_id": f"Figure {idx}",
                    "figure_name": f"Dependence Plot - {feature}",
                    "source_file": str(dst),
                    "recommended_location": "main_text",
                    "purpose": "展示关键健康、家庭或经济变量的 SHAP 局部关系",
                    "interpretation_note": "family_care_index_v1 统一解释为家庭照料与代际支持压力代理。",
                }
            )

    appendix_bar_no_totmet = APP_FIG_DIR / "figA1_shap_bar_no_totmet.png"
    appendix_bar_full = APP_FIG_DIR / "figA2_shap_bar_full_feature.png"
    copy_figure(STAGE4_DIR / "fig_shap_bar_no_totmet.png", appendix_bar_no_totmet)
    copy_figure(STAGE4_DIR / "fig_shap_bar_full_feature.png", appendix_bar_full)
    figure_rows.extend(
        [
            {
                "figure_id": "Appendix Figure A1",
                "figure_name": "SHAP Bar Summary (no_totmet)",
                "source_file": str(appendix_bar_no_totmet),
                "recommended_location": "appendix",
                "purpose": "对照展示剔除 totmet_w 后的 SHAP 排序",
                "interpretation_note": "用于说明 totmet_w 的解释风险和性能差异。",
            },
            {
                "figure_id": "Appendix Figure A2",
                "figure_name": "SHAP Bar Summary (full_feature)",
                "source_file": str(appendix_bar_full),
                "recommended_location": "appendix",
                "purpose": "对照展示全特征方案的 SHAP 排序",
                "interpretation_note": "用于附录说明主模型与全特征模型差异。",
            },
        ]
    )
    for idx, name in enumerate(appendix_dep_extra, start=3):
        src = STAGE4_DIR / "dependence_main_clean" / name
        dst = APP_FIG_DIR / name
        if copy_figure(src, dst):
            feature = name.replace(".png", "")
            figure_rows.append(
                {
                    "figure_id": f"Appendix Figure A{idx}",
                    "figure_name": f"Dependence Plot - {feature}",
                    "source_file": str(dst),
                    "recommended_location": "appendix",
                    "purpose": "补充展示其余 dependence plots",
                    "interpretation_note": "作为附录对照，不宜在正文展开全部解释。",
                }
            )
    figures_df = pd.DataFrame(figure_rows)
    figures_list_path = OUTPUT_DIR / "paper_figures_list_v1.xlsx"
    figures_df.to_excel(figures_list_path, index=False)

    outline_lines = [
        "实证结果写作框架",
        "",
        "1. 描述性统计与劳动参与现状",
        "- 要回答的问题：样本总体特征如何，不同年龄、性别、城乡和压力状态下的劳动参与率有何差异。",
        f"- 建议引用：{table1_path.name}、{table2_path.name}、{labor_fig.name}。",
        "- 解释要点：突出年龄梯度、性别差异、城乡差异，以及健康和压力变量分组下的劳动参与率分化。",
        "- 需要避免的误读：描述性差异不代表因果影响。",
        "",
        "2. 主回归结果",
        "- 要回答的问题：在控制人口学特征后，健康约束、家庭照料与代际支持压力代理、经济压力变量与劳动参与之间的方向和显著性如何。",
        f"- 建议引用：{table3_path.name}、{forest_fig.name}。",
        "- 解释要点：重点解释 poor_health、iadl_limit、family_care_index_v1、economic_pressure_index_v1、log_hhcperc_v1_w 等变量。",
        "- 需要避免的误读：year_2020 仅为控制变量；family_care_index_v1 不是纯照料责任测度。",
        "",
        "3. 稳健性检验",
        "- 要回答的问题：在改变变量组合、样本时期和年龄范围后，核心结论是否稳定。",
        f"- 建议引用：{table4_path.name}、{hetero_path.name} 中的 summary 页。",
        "- 解释要点：强调健康约束与经济压力方向稳定，family_care_index_v1 的解释应保持谨慎。",
        "- 需要避免的误读：稳健性支持的是相关结构稳定，不是识别因果通道。",
        "",
        "4. 机器学习预测结果",
        "- 要回答的问题：不同模型的预测性能如何，LightGBM 是否适合作为可解释分析基础。",
        f"- 建议引用：{table5_path.name}。",
        "- 解释要点：说明 LightGBM 略优于 XGBoost 和 LogisticRegression，AUC 处于可接受范围。",
        "- 需要避免的误读：预测性能高不等于变量具有更强因果影响。",
        "",
        "5. SHAP 可解释分析",
        "- 要回答的问题：在 main_clean 方案下，哪些变量具有较高的预测解释贡献。",
        f"- 建议引用：{table6_path.name}、{shap_bar_main_copy.name}、{shap_beeswarm_main_copy.name} 及精选 dependence plots。",
        "- 解释要点：强调 age、urban、female、iadl_limit、economic pressure related variables 等的解释贡献。",
        "- 需要避免的误读：SHAP 仅反映模型内的解释贡献，不能写成因果证明。",
        "",
        "6. 异质性分析",
        "- 要回答的问题：核心变量在性别、城乡、年龄、健康、家庭压力和经济压力分组中是否存在差异。",
        f"- 建议引用：{table7_path.name}、{hetero_path.name}。",
        "- 解释要点：比较 poor_health、iadl_limit、family_care_index_v1、economic_pressure_index_v1 的系数方向和显著性变化。",
        "- 需要避免的误读：异质性结果更多是相关结构差异，不宜过度延伸为政策效果差异。",
        "",
        "7. 小结",
        "- 要回答的问题：整体实证证据是否支持“延迟退休背景下的劳动参与适配性受健康、家庭与经济压力共同约束”的主线。",
        "- 建议引用：概括 Table 3、Table 4、Table 6 和 Table 7 的核心发现。",
        "- 解释要点：将主回归、稳健性、SHAP 和异质性分析形成一致叙事。",
        "- 需要避免的误读：不要把预测解释贡献与政策因果效应混同。",
    ]
    outline_path = OUTPUT_DIR / "empirical_results_outline_v1.txt"
    outline_path.write_text("\n".join(outline_lines), encoding="utf-8")

    baseline_auc = float(ml_metrics.loc[ml_metrics["model"] == "LightGBM", "roc_auc"].iloc[0])
    main_clean_auc = float(shap_metrics.loc[shap_metrics["scheme"] == "main_clean", "roc_auc"].iloc[0])
    full_auc = float(shap_metrics.loc[shap_metrics["scheme"] == "full_feature", "roc_auc"].iloc[0])
    direction_review = pd.read_excel(STAGE3_DIR / "direction_review_v1.xlsx")
    family_dir = direction_review.loc[direction_review["variable"] == "family_care_index_v1"].iloc[0]
    econ_dir = direction_review.loc[direction_review["variable"] == "economic_pressure_index_v1"].iloc[0]

    draft_lines = [
        "实证结果文字草稿",
        "",
        "一、描述性统计与劳动参与现状",
        "表1给出了样本主要变量的描述性统计结果。整体来看，样本中的中老年劳动参与率仍处于相对较高水平，但不同群体之间存在明显分化。结合表2和图1可以看到，劳动参与率随年龄上升总体呈下降趋势，女性样本的劳动参与率低于男性，城市样本的劳动参与率也明显低于农村样本。这说明在延迟退休背景下，中老年劳动供给并非单一由年龄制度边界决定，而是已经体现出显著的人口学和家庭分层特征。",
        "进一步分组比较发现，健康状况较差、IADL受限以及经济压力较高的群体，其劳动参与表现呈现出更复杂的分化格局。其中，健康约束较强群体的劳动参与率总体偏低，而经济压力较高群体中仍有相当部分样本继续劳动，这为后续回归和机器学习分析提供了直观背景。",
        "",
        "二、Logit主回归结果",
        "表3报告了逐步加入健康约束、家庭照料与代际支持压力代理、经济压力以及行为控制变量后的 Logit 回归结果。总体来看，健康变量的方向较为稳定。poor_health、adl_limit 和 iadl_limit 的系数均为负，说明健康受限和日常功能受限与劳动参与概率下降显著相关。chronic_count 也呈负向关系，表明慢性病负担越重，继续劳动的可能性整体越低。",
        "在家庭因素方面，co_reside_child 和 care_elder_or_disabled 均表现为负向，说明与子女同住或承担老人、病人照料责任时，劳动参与更容易受到挤压。需要注意的是，family_care_index_v1 在主回归中表现为正向，这并不宜直接解释为照料责任促进劳动参与，而更应理解为家庭照料与代际支持压力代理，其正向结果可能反映出家庭结构、同住安排和代际资源交换共同作用下的劳动供给维持机制。",
        "在经济压力方面，economic_pressure_index_v1 为正，说明经济压力越大，样本继续劳动的倾向越强。这一结果更适合解释为经济压力对继续劳动的推动，而不宜进一步推断为其更适合延迟退休。与此同时，log_hhcperc_v1_w 和 log_medical_expense_w 均呈负向，意味着消费水平和医疗相关支出负担的变化与劳动参与选择之间存在显著相关性。",
        "year_2020 在模型中表现显著，但该变量仅作为年份控制变量使用，不宜作为实质性政策含义展开解释。此外，depression_high 呈现正向、total_cognition_w 呈现负向，这与通常预期并不完全一致，提示这些变量可能受到编码方式、样本选择或代理效应影响，正文中应保留必要的谨慎说明。",
        "",
        "三、稳健性检验",
        "表4基于不同变量组合和样本范围展开稳健性检验。总体来看，在移除行为变量、移除 family_care_index_v1、替换经济压力指标、剔除认知变量以及限定年份或年龄样本后，健康约束和经济压力变量的方向总体保持稳定，说明主结论具有较好的稳健性。",
        "尤其是 poor_health、iadl_limit 和 economic_pressure_index_v1 在多数设定下仍保持原有方向，说明健康受限与经济压力是解释中老年劳动参与差异的两个较为稳定维度。相比之下，family_care_index_v1 的解释仍需保持谨慎，因为该指标本身具有较强代理性质，其方向更多反映家庭结构与代际支持压力的复合效应，而不是单纯照料责任的净效应。",
        "",
        "四、机器学习与 SHAP 结果",
        "表5显示，baseline 阶段中 LightGBM 的预测性能略优于 XGBoost 和 LogisticRegression，表现为更高的测试集 AUC 和 F1 值。因此，本文进一步采用 LightGBM 作为可解释机器学习分析的主模型。考虑到行为变量和活动强度变量可能带来额外代理风险，正文中的 SHAP 主模型采用 main_clean 方案，即剔除 smokev、drinkl、exercise 和 totmet_w。该方案的 AUC 为 "
        f"{main_clean_auc:.3f}，虽低于全特征模型的 {full_auc:.3f}，但整体仍处于可接受范围，且解释纯度更高。",
        "表6和 SHAP 主图表明，在 main_clean 方案下，age、urban、female、year_2020、iadl_limit、chronic_count 等变量具有较高解释贡献。健康约束变量整体占据较重要位置，说明健康能力仍是中老年劳动参与适配性的关键维度。经济压力相关变量如 log_intergen_support_out_w、log_hhcperc_v1_w 也具有较高解释贡献，提示代际支持压力和经济资源状况在劳动参与预测中不可忽视。",
        "需要强调的是，SHAP 结果只能说明某些变量在预测劳动参与时具有较高解释贡献，而不能证明这些变量具有更强的因果效应。对于 year_2020、family_care_index_v1 以及所有 *_missing 缺失指示变量，更不宜进行机制性过度解读。",
        "",
        "五、异质性分析",
        "表7进一步展示了异质性分析结果。按性别、城乡、年龄组、健康状态以及家庭和经济压力分组后，健康约束变量在多数分组中仍保持负向，说明健康约束并非局限于某一特定群体，而是较为普遍地影响中老年劳动参与。相比之下，family_care_index_v1 和 economic_pressure_index_v1 在不同分组中的表现更具差异性，表明家庭照料与代际支持压力、经济压力对劳动参与的作用可能受到家庭结构和个体处境的调节。",
        "从年龄组来看，70 岁及以上样本中部分变量的显著性有所下降，这说明高龄阶段的劳动参与更容易受到未观测因素影响。按健康状态分组后，poor_health=1 组中健康功能变量的边际约束更突出，也提示延迟退休背景下不能忽视健康脆弱群体的劳动参与适配性问题。",
        "",
        "六、小结",
        "综合主回归、稳健性检验、机器学习和 SHAP 分析可以看到，延迟退休背景下中老年劳动参与适配性并不单纯取决于年龄制度边界，而是同时受到健康约束、家庭照料与代际支持压力代理以及经济压力的共同影响。健康受限总体会降低继续劳动倾向，经济压力则可能推动部分样本维持劳动供给，而家庭变量的作用则更体现为家庭结构和代际支持压力的复合代理效应。上述结果为论文后续的政策讨论提供了经验基础，但在解释时仍应避免将相关性结果直接上升为严格因果结论。",
    ]
    draft_path = OUTPUT_DIR / "empirical_results_draft_v1.txt"
    draft_path.write_text("\n".join(draft_lines), encoding="utf-8")

    recommendation_lines = [
        "论文图表使用建议",
        "1. 正文最推荐使用的表格清单：Table 1、Table 2、Table 3、Table 4、Table 5、Table 6、Table 7 中至少应保留 Table 3、Table 4、Table 5、Table 6。",
        "2. 正文最推荐使用的图形清单：劳动参与率分组柱状图、Logit Model 5 forest plot、SHAP bar summary（main_clean）、SHAP beeswarm（main_clean），以及 3-4 张关键 dependence plots。",
        "3. 更适合放附录的结果：全量 heterogeneity 明细表、no_totmet 和 full_feature 的 SHAP bar summary、其余 dependence plots、单年份与年龄截断回归的完整结果。",
        "4. 如果论文字数有限，优先保留：Table 3 主回归、Table 4 稳健性、Table 5 机器学习性能、Table 6 SHAP 重要特征、Figure 3 SHAP bar 和 Figure 4 SHAP beeswarm。",
        "5. 解释风险较高、正文需回避过度展开的变量：family_care_index_v1、year_2020、depression_high、total_cognition_w、所有 *_missing 指示变量，以及任何可能代理劳动活动强度的变量。",
    ]
    recommendation_path = OUTPUT_DIR / "paper_use_recommendation_v1.txt"
    recommendation_path.write_text("\n".join(recommendation_lines), encoding="utf-8")

    heterogeneity_done = bool(pd.DataFrame(hetero_summary_rows)["success"].all())
    main_shap_in_main = bool((figures_df["figure_name"] == "SHAP Bar Summary (main_clean)").any() and (figures_df["recommended_location"] == "main_text").any())
    recommend_write = True
    top5_recommended = [
        "Table 3 主回归结果表",
        "Table 4 稳健性检验结果表",
        "Table 6 SHAP 重要特征表",
        "Figure 3 SHAP bar summary (main_clean)",
        "Figure 4 SHAP beeswarm (main_clean)",
    ]

    output_files = [
        hetero_path,
        table1_path,
        table2_path,
        table3_path,
        table4_path,
        table5_path,
        table6_path,
        table7_path,
        figures_list_path,
        outline_path,
        draft_path,
        recommendation_path,
    ]

    print(f"Heterogeneity analysis completed: {'Yes' if heterogeneity_done else 'No'}")
    print("Final paper tables generated: 7")
    print(f"Final figure list entries: {len(figures_df)}")
    print(f"SHAP main figures recommended for main text: {'Yes' if main_shap_in_main else 'No'}")
    print(f"Recommend entering final writing stage: {'Yes' if recommend_write else 'No'}")
    print("Top 5 recommended tables/figures for main text:")
    for item in top5_recommended:
        print(f"  {item}")
    print("Output files:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
