from __future__ import annotations

import math
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
STAGE5_DIR = ROOT / "outputs" / "model_stage5_paper_ready_v1"
OUTPUT_DIR = ROOT / "outputs" / "model_stage6_retirement_relevance_v1"

LOGIT_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv"
ML_DATA = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"

TARGET = "labor_participation"
GROUP = "ID"
RANDOM_STATE = 42


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_logit_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edu" in out.columns:
        edu_num = safe_numeric(out["edu"])
        values = sorted([int(v) for v in edu_num.dropna().unique().tolist()])
        for value in values:
            col = f"edu_{value}"
            if col not in out.columns:
                out[col] = (edu_num == value).astype(int)
    return out


def build_age_policy_vars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age = safe_numeric(out["age"])
    female = safe_numeric(out["female"]).fillna(0)

    male_relevant = (female == 0) & age.between(55, 64, inclusive="both")
    female_relevant = (female == 1) & age.between(45, 59, inclusive="both")
    out["retirement_relevant_age"] = (male_relevant | female_relevant).astype(int)

    male_core = (female == 0) & age.between(58, 63, inclusive="both")
    female_core = (female == 1) & age.between(50, 58, inclusive="both")
    out["near_retirement_core"] = (male_core | female_core).astype(int)

    out["older_post_retirement"] = (age >= 65).astype(int)

    out["age_policy_group"] = "A"
    out.loc[out["retirement_relevant_age"] == 1, "age_policy_group"] = "B"
    out.loc[out["near_retirement_core"] == 1, "age_policy_group"] = "C"
    out.loc[out["older_post_retirement"] == 1, "age_policy_group"] = "D"
    return out


def choose_family_cut(df: pd.DataFrame) -> tuple[float, float]:
    family = safe_numeric(df["family_care_index_v1"])
    median = float(family.median())
    cut = max(2.0, median)
    return cut, median


def add_retirement_suitability(df: pd.DataFrame, family_cut: float, econ_cut: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    labor = safe_numeric(out[TARGET]).fillna(0).astype(int)
    relevant = safe_numeric(out["retirement_relevant_age"]).fillna(0).astype(int)
    poor_health = safe_numeric(out.get("poor_health", 0)).fillna(0).astype(int)
    adl = safe_numeric(out.get("adl_limit", 0)).fillna(0).astype(int)
    iadl = safe_numeric(out.get("iadl_limit", 0)).fillna(0).astype(int)
    econ = safe_numeric(out.get("economic_pressure_index_v1", np.nan))
    family = safe_numeric(out.get("family_care_index_v1", np.nan))

    family_high = (family >= family_cut).fillna(False)
    econ_high = (econ >= econ_cut).fillna(False)
    health_constrained = ((poor_health == 1) | (adl == 1) | (iadl == 1))
    constrained_flag = health_constrained | family_high

    out["retirement_suitability_v1"] = "unknown"

    support_priority = health_constrained & econ_high & ((labor == 0))
    constrained_working = (relevant == 1) & (labor == 1) & constrained_flag
    support_priority = support_priority | ((relevant == 1) & constrained_working & econ_high)

    suitable_flexible = (
        (relevant == 1)
        & (labor == 1)
        & (poor_health == 0)
        & (adl == 0)
        & (iadl == 0)
        & (~econ_high)
        & (~family_high)
    )
    pressure_driven = (relevant == 1) & (labor == 1) & econ_high
    constrained_exit = (relevant == 1) & (labor == 0) & constrained_flag
    potential_supply = (
        (relevant == 1)
        & (labor == 0)
        & (poor_health == 0)
        & (adl == 0)
        & (iadl == 0)
        & (~family_high)
        & (~econ_high)
    )

    # Priority order: support > constrained_exit > constrained_working > pressure > suitable > potential
    out.loc[support_priority, "retirement_suitability_v1"] = "support_priority"
    out.loc[constrained_exit & (out["retirement_suitability_v1"] == "unknown"), "retirement_suitability_v1"] = "constrained_exit"
    out.loc[constrained_working & (out["retirement_suitability_v1"] == "unknown"), "retirement_suitability_v1"] = "constrained_working"
    out.loc[pressure_driven & (out["retirement_suitability_v1"] == "unknown"), "retirement_suitability_v1"] = "pressure_driven_work"
    out.loc[suitable_flexible & (out["retirement_suitability_v1"] == "unknown"), "retirement_suitability_v1"] = "suitable_for_flexible_delay"
    out.loc[potential_supply & (out["retirement_suitability_v1"] == "unknown"), "retirement_suitability_v1"] = "potential_labor_supply"

    out["family_pressure_high"] = family_high.astype(int)
    out["economic_pressure_high"] = econ_high.astype(int)
    out["support_priority_flag"] = support_priority.astype(int)
    return out


def fit_logit(df: pd.DataFrame, variables: list[str], group_col: str = GROUP) -> tuple[pd.DataFrame, dict[str, object]]:
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

        y = safe_numeric(df[TARGET]).fillna(0).astype(int)
        Xc = sm.add_constant(X, has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if group_col in df.columns:
                    result = Logit(y, Xc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df[group_col]})
                else:
                    result = Logit(y, Xc).fit(disp=0)

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

    empty = pd.DataFrame(columns=["variable", "coef", "std_err", "z", "p_value", "odds_ratio", "nobs", "pseudo_r2", "AIC", "BIC"])
    return empty, {
        "success": False,
        "nobs": len(df),
        "pseudo_r2": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "used_variables": [],
        "missing_variables": missing,
        "removed_variables": removed,
    }


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age = safe_numeric(out["age"])
    out["broad_age_group"] = pd.cut(age, bins=[44, 54, 59, 64, 120], labels=["45-54", "55-59", "60-64", "65+"], include_lowest=True)
    out["retirement_age_group"] = pd.cut(age, bins=[44, 54, 59, 64], labels=["45-54", "55-59", "60-64"], include_lowest=True)
    return out


def group_labor_table(df: pd.DataFrame, group_cols: list[str], name: str) -> pd.DataFrame:
    tab = (
        df.groupby(group_cols, dropna=False, observed=False)[TARGET]
        .agg(sample_size="count", labor_participation_rate="mean")
        .reset_index()
    )
    tab.insert(0, "table_name", name)
    return tab


def create_bar_plot(df: pd.DataFrame, out_path: Path) -> None:
    temp = df.copy()
    temp["ret_group_label"] = np.where(temp["retirement_relevant_age"] == 1, "准退休年龄段", "其他年龄段")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    age_tab = temp.groupby("retirement_age_group", observed=False)[TARGET].mean().reset_index()
    axes[0].bar(age_tab["retirement_age_group"].astype(str), age_tab[TARGET], color="#4E79A7")
    axes[0].set_title("Labor Rate by Retirement Age Group")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Labor Participation Rate")

    sex_tab = temp.groupby("female")[TARGET].mean().reset_index()
    axes[1].bar(["Male", "Female"], sex_tab[TARGET], color="#F28E2B")
    axes[1].set_title("Labor Rate by Gender")
    axes[1].set_ylim(0, 1)

    urban_tab = temp.groupby("urban")[TARGET].mean().reset_index()
    axes[2].bar(["Rural", "Urban"], urban_tab[TARGET], color="#59A14F")
    axes[2].set_title("Labor Rate by Urban Status")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_distribution_plot(df: pd.DataFrame, out_path: Path) -> None:
    dist = df["retirement_suitability_v1"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(dist.index.astype(str), dist.values, color="#4E79A7")
    ax.set_title("Retirement Suitability Distribution")
    ax.set_ylabel("Sample Size")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_health_plot(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for var in ["poor_health", "adl_limit", "iadl_limit"]:
        if var in df.columns:
            tab = df.groupby(var, observed=False)[TARGET].mean().reset_index()
            for _, row in tab.iterrows():
                rows.append({"variable": var, "group": int(row[var]), "labor_rate": float(row[TARGET])})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    base = np.arange(len(plot_df["variable"].unique()))
    for idx, group in enumerate(sorted(plot_df["group"].unique())):
        subset = plot_df[plot_df["group"] == group]
        ax.bar(base + idx * width, subset["labor_rate"], width=width, label=f"value={group}")
    ax.set_xticks(base + width / 2)
    ax.set_xticklabels(plot_df["variable"].unique())
    ax.set_ylim(0, 1)
    ax.set_title("Health Constraints and Labor Participation")
    ax.set_ylabel("Labor Participation Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_econ_plot(df: pd.DataFrame, out_path: Path) -> None:
    tab = df.groupby("economic_pressure_high", observed=False)[TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Low", "High"], tab[TARGET], color=["#59A14F", "#E15759"])
    ax.set_ylim(0, 1)
    ax.set_title("Economic Pressure and Labor Participation")
    ax.set_ylabel("Labor Participation Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    logit_df = prepare_logit_df(pd.read_csv(LOGIT_DATA, low_memory=False))
    ml_df = pd.read_csv(ML_DATA, low_memory=False)

    logit_df = build_age_policy_vars(add_age_group(logit_df))
    ml_df = build_age_policy_vars(add_age_group(ml_df))

    family_cut, family_median = choose_family_cut(ml_df)
    ml_df = add_retirement_suitability(ml_df, family_cut=family_cut, econ_cut=3.0)
    logit_df = add_retirement_suitability(logit_df, family_cut=family_cut, econ_cut=3.0)

    # 1. retirement relevance groups
    age_group_labels = {
        "A": "A_non_relevant",
        "B": "B_relevant",
        "C": "C_core_relevant",
        "D": "D_age65plus",
    }
    age_group_dist = (
        ml_df.groupby("age_policy_group", observed=False)[TARGET]
        .agg(sample_size="count", labor_participation_rate="mean")
        .reset_index()
    )
    age_group_dist["group_label"] = age_group_dist["age_policy_group"].map(age_group_labels)

    relevant_df = ml_df.loc[ml_df["retirement_relevant_age"] == 1].copy()
    relevant_df["family_pressure_group"] = np.where(relevant_df["family_pressure_high"] == 1, "high", "low")
    relevant_df["economic_pressure_group"] = np.where(relevant_df["economic_pressure_high"] == 1, "high", "low")

    suitability_dist_all = (
        ml_df.groupby("retirement_suitability_v1", observed=False)[TARGET]
        .agg(sample_size="count", labor_participation_rate="mean")
        .reset_index()
        .sort_values("sample_size", ascending=False)
    )
    suitability_dist_relevant = (
        relevant_df.groupby("retirement_suitability_v1", observed=False)[TARGET]
        .agg(sample_size="count", labor_participation_rate="mean")
        .reset_index()
        .sort_values("sample_size", ascending=False)
    )

    suitability_path = OUTPUT_DIR / "retirement_suitability_distribution_v1.xlsx"
    with pd.ExcelWriter(suitability_path) as writer:
        age_group_dist.to_excel(writer, sheet_name="age_policy_groups", index=False)
        suitability_dist_relevant.to_excel(writer, sheet_name="suitability_relevant_only", index=False)
        suitability_dist_all.to_excel(writer, sheet_name="suitability_full_sample", index=False)
        pd.DataFrame(
            {
                "metric": [
                    "family_pressure_cut",
                    "family_pressure_median",
                    "economic_pressure_high_cut",
                    "female_relevant_rule",
                    "male_relevant_rule",
                    "female_core_rule",
                    "male_core_rule",
                ],
                "value": [
                    family_cut,
                    family_median,
                    3,
                    "45-59",
                    "55-64",
                    "50-58",
                    "58-63",
                ],
            }
        ).to_excel(writer, sheet_name="rules", index=False)

    descriptive_path = OUTPUT_DIR / "near_retirement_descriptive_tables_v1.xlsx"
    with pd.ExcelWriter(descriptive_path) as writer:
        group_labor_table(relevant_df, ["female"], "by_gender").to_excel(writer, sheet_name="by_gender", index=False)
        group_labor_table(relevant_df, ["urban"], "by_urban").to_excel(writer, sheet_name="by_urban", index=False)
        group_labor_table(relevant_df, ["retirement_age_group"], "by_age_group").to_excel(writer, sheet_name="by_age_group", index=False)
        group_labor_table(relevant_df, ["poor_health"], "by_poor_health").to_excel(writer, sheet_name="by_health", index=False)
        group_labor_table(relevant_df, ["family_pressure_group"], "by_family_pressure").to_excel(writer, sheet_name="by_family_pressure", index=False)
        group_labor_table(relevant_df, ["economic_pressure_group"], "by_economic_pressure").to_excel(writer, sheet_name="by_economic_pressure", index=False)
        suitability_dist_relevant.to_excel(writer, sheet_name="suitability_distribution", index=False)

    # 3. near-retirement-core logit
    core_df = logit_df.loc[logit_df["near_retirement_core"] == 1].copy()
    near_retirement_vars = [
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
    ]
    near_logit, near_meta = fit_logit(core_df, near_retirement_vars)
    near_path = OUTPUT_DIR / "near_retirement_logit_results_v1.xlsx"
    with pd.ExcelWriter(near_path) as writer:
        near_logit.to_excel(writer, sheet_name="coefficients", index=False)
        pd.DataFrame(
            {
                "metric": ["success", "nobs", "pseudo_r2", "AIC", "BIC"],
                "value": [
                    near_meta["success"],
                    near_meta["nobs"],
                    near_meta["pseudo_r2"],
                    near_meta["AIC"],
                    near_meta["BIC"],
                ],
            }
        ).to_excel(writer, sheet_name="model_summary", index=False)
        pd.DataFrame({"missing_variables": near_meta["missing_variables"]}).to_excel(writer, sheet_name="missing_variables", index=False)
        pd.DataFrame({"removed_variables": near_meta["removed_variables"]}).to_excel(writer, sheet_name="removed_variables", index=False)

    # 4. interaction models
    base_controls = [
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
    interaction_targets = [
        "poor_health",
        "adl_limit",
        "iadl_limit",
        "family_care_index_v1",
        "economic_pressure_index_v1",
    ]
    interaction_rows = []
    interaction_meta_rows = []
    for var in interaction_targets:
        work_df = logit_df.copy()
        inter_col = f"{var}_x_retirement_relevant_age"
        work_df[inter_col] = safe_numeric(work_df.get(var, np.nan)) * safe_numeric(work_df["retirement_relevant_age"])
        vars_use = base_controls.copy()
        if inter_col not in vars_use:
            vars_use.append("retirement_relevant_age")
            vars_use.append(inter_col)
        result_df, meta = fit_logit(work_df, vars_use)
        result_df.insert(0, "interaction_model", f"{var} x retirement_relevant_age")
        interaction_rows.append(result_df)
        interaction_meta_rows.append(
            {
                "interaction_model": f"{var} x retirement_relevant_age",
                "success": meta["success"],
                "nobs": meta["nobs"],
                "pseudo_r2": meta["pseudo_r2"],
                "AIC": meta["AIC"],
                "BIC": meta["BIC"],
                "missing_variables": "; ".join(meta["missing_variables"]),
                "removed_variables": "; ".join(meta["removed_variables"]),
            }
        )

    interaction_path = OUTPUT_DIR / "retirement_interaction_logit_v1.xlsx"
    with pd.ExcelWriter(interaction_path) as writer:
        pd.concat(interaction_rows, ignore_index=True).to_excel(writer, sheet_name="coefficients", index=False)
        pd.DataFrame(interaction_meta_rows).to_excel(writer, sheet_name="model_summary", index=False)

    # 5. figures
    create_bar_plot(relevant_df, OUTPUT_DIR / "fig_near_retirement_labor_rates.png")
    create_distribution_plot(relevant_df, OUTPUT_DIR / "fig_retirement_suitability_distribution.png")
    create_health_plot(relevant_df, OUTPUT_DIR / "fig_health_constraints_near_retirement.png")
    create_econ_plot(relevant_df, OUTPUT_DIR / "fig_economic_pressure_near_retirement.png")

    # 6. writeups
    support_dist = suitability_dist_relevant.set_index("retirement_suitability_v1")["sample_size"].to_dict()
    pressure_share = support_dist.get("pressure_driven_work", 0) / max(int(relevant_df.shape[0]), 1)
    constrained_share = support_dist.get("constrained_exit", 0) / max(int(relevant_df.shape[0]), 1)
    potential_share = support_dist.get("potential_labor_supply", 0) / max(int(relevant_df.shape[0]), 1)
    support_priority_share = support_dist.get("support_priority", 0) / max(int(relevant_df.shape[0]), 1)

    writeup_text = f"""延迟退休相关性分析说明

1. 为什么本文不是延迟退休政策效果评估
本文并未利用政策实施前后差异、试点差异或法定退休年龄调整冲击识别政策因果效应，因此不能将结果解释为“延迟退休政策导致了劳动参与变化”。本文的定位是：在延迟退休背景下，识别哪些中老年群体具备继续劳动的适配性，哪些群体受到健康、家庭责任和经济压力约束。

2. 为什么仍然可以放在延迟退休背景下
延迟退休改革的现实含义，不只是提高法定退休年龄，更是要求判断：在接近退休年龄的人群中，哪些人有能力继续劳动，哪些人即使仍在劳动也属于压力驱动或带约束劳动，哪些人需要弹性退出和保障支持。本文的准退休年龄段分析正是围绕这一问题展开。

3. 准退休年龄段分析说明了什么
本轮将男性 55-64 岁、女性 45-59 岁定义为宽口径准退休年龄段，并将男性 58-63 岁、女性 50-58 岁定义为核心准退休年龄段。这样做不是为了精确模拟法定退休制度，而是为了识别在制度讨论最敏感的人群中，劳动参与与健康、家庭、经济条件之间的关系。

4. 健康约束如何对应“能不能延迟”
如果 poor_health、adl_limit、iadl_limit 等变量在准退休年龄段内仍表现出显著负向关系，就说明延迟退休能否转化为有效劳动力供给，首先取决于中老年人是否具备继续工作的健康能力。这一层对应的是“能不能延迟”。

5. 家庭责任如何对应“有没有条件延迟”
family_care_index_v1、co_reside_child、care_elder_or_disabled 等变量反映了家庭照料与代际支持压力。当这些变量抑制劳动参与时，说明即使个体具备一定健康能力，也可能因为照护责任和家庭结构约束而缺乏继续劳动的条件。这一层对应的是“有没有条件延迟”。

6. 经济压力如何对应“主动延迟还是被动劳动”
economic_pressure_index_v1 及相关收入、医疗支出变量，如果与继续劳动呈正向关系，不能简单解释为“适合延迟退休”，更应理解为部分个体由于养老金不足、医疗负担或家庭支出压力而被动维持劳动。换言之，继续劳动并不必然代表适配性高，也可能是保障不足。

7. 适配性分层如何支撑“弹性退休”和“分类支持”
本轮构造的 retirement_suitability_v1 将准退休年龄段样本区分为适配弹性延迟型、压力驱动继续劳动型、带约束继续劳动型、健康/家庭约束退出型、潜在劳动供给型和保障支持重点型。这样的分层说明：延迟退休不宜采取单一推进方式，而应根据健康能力、家庭责任和经济压力实行分类支持。

8. 政策建议应如何从分层结果推出
如果一个群体属于 suitable_for_flexible_delay，则更适合通过弹性岗位、再就业服务和技能更新来释放劳动供给；如果属于 pressure_driven_work，则重点不是鼓励其继续劳动，而是缓解其经济脆弱性；如果属于 constrained_exit 或 support_priority，则需要更多健康管理、照护服务和基本保障；如果属于 constrained_working，则应加强劳动保护和工时弹性。这说明“延迟退休背景下的适配性识别”可以为分类推进、弹性退出和重点保障提供经验依据。

补充说明
女性准退休年龄段采用 45-59 岁宽口径，是因为 CHARLS 中难以稳定区分女职工 50 岁退休与女干部 55 岁退休。year_2020 仍只作为年份控制变量，不解释为政策效果。
"""
    write_text(OUTPUT_DIR / "retirement_relevance_writeup_v1.txt", writeup_text)

    implication_text = """延迟退休相关政策建议

1. 对适配继续劳动型
应优先提供灵活岗位、再就业服务和技能培训，帮助其在健康状况允许的情况下平稳延长劳动参与时间，提升劳动力供给转化效率。

2. 对经济压力驱动型
应完善养老金、低收入补贴和医疗保障，避免个体因收入不足、医疗负担或家庭支出压力而被迫劳动，降低“被动延迟”的风险。

3. 对健康/家庭约束退出型
应允许弹性退出，并加强健康管理、康复服务、家庭照护和社区托育托老支持，避免把延迟退休理解为对所有群体的统一要求。

4. 对带约束继续劳动型
应加强劳动保护、工时弹性、岗位适配和职业健康支持，避免其在健康受限或照护压力较高的状态下被迫持续劳动。

5. 对保障支持重点型
应提供长期护理、社区照护、基本生活保障和医疗救助，把政策重点放在底线保障和风险缓释上，而不是单纯提高劳动参与率。
"""
    write_text(OUTPUT_DIR / "retirement_policy_implications_v1.txt", implication_text)

    output_files = [
        suitability_path,
        descriptive_path,
        near_path,
        interaction_path,
        OUTPUT_DIR / "fig_near_retirement_labor_rates.png",
        OUTPUT_DIR / "fig_retirement_suitability_distribution.png",
        OUTPUT_DIR / "fig_health_constraints_near_retirement.png",
        OUTPUT_DIR / "fig_economic_pressure_near_retirement.png",
        OUTPUT_DIR / "retirement_relevance_writeup_v1.txt",
        OUTPUT_DIR / "retirement_policy_implications_v1.txt",
    ]

    print(f"准退休年龄段样本量: {int(relevant_df.shape[0])}")
    print(f"核心准退休年龄段样本量: {int(core_df.shape[0])}")
    print(f"准退休年龄段劳动参与率: {float(relevant_df[TARGET].mean()):.4f}")
    print("retirement_suitability_v1 分布:")
    for row in suitability_dist_relevant.itertuples(index=False):
        print(f"  {row.retirement_suitability_v1}: {int(row.sample_size)} ({float(row.sample_size / max(len(relevant_df), 1)):.4f})")
    print(f"准退休年龄段 Logit 是否跑通: {near_meta['success']}")
    print(f"交互项模型是否跑通: {all(item['success'] for item in interaction_meta_rows)}")
    print("哪些结果最能直接支撑“延迟退休背景”:")
    print("  1. 准退休年龄段与核心准退休年龄段的劳动参与率差异")
    print("  2. retirement_suitability_v1 的适配性分层分布")
    print("  3. 核心准退休年龄段 Logit 中健康、家庭、经济变量方向")
    print("  4. 全样本交互项模型对 retirement_relevant_age 的强化效应")
    print("是否建议将论文最终主线改成“延迟退休适配性识别”: 是")
    print("输出文件路径列表:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
