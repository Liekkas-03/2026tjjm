from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
STAGE5_DIR = ROOT / "outputs" / "model_stage5_paper_ready_v1"
STAGE4_DIR = ROOT / "outputs" / "model_stage4_shap_v1"
STAGE6_DIR = ROOT / "outputs" / "model_stage6_retirement_relevance_v1"
TRAINING_DIR = ROOT / "outputs" / "model_training_v1"
STAGE3_DIR = ROOT / "outputs" / "model_stage3_review_v1"
OUTPUT_DIR = ROOT / "outputs" / "paper_figures_final_v1"
APPENDIX_DIR = OUTPUT_DIR / "appendix"


GROUP_LABELS = {
    "age_group_stage5": ("年龄组", {"45-59": "45-59岁", "60-69": "60-69岁", "70+": "70岁及以上"}),
    "female": ("性别", {"0": "男性", "1": "女性", 0: "男性", 1: "女性"}),
    "urban": ("城乡", {"0": "农村", "1": "城镇", 0: "农村", 1: "城镇"}),
    "poor_health": ("健康状态", {"0": "健康状况较好", "1": "自评健康较差", 0: "健康状况较好", 1: "自评健康较差"}),
    "family_high_stage5": ("家庭压力", {"0": "家庭压力较低", "1": "家庭压力较高", 0: "家庭压力较低", 1: "家庭压力较高"}),
    "econ_high_stage5": ("经济压力", {"0": "经济压力较低", "1": "经济压力较高", 0: "经济压力较低", 1: "经济压力较高"}),
    "year": ("年份", {"2018": "2018年", "2020": "2020年", 2018: "2018年", 2020: "2020年"}),
}

FIG2_VARIABLES = [
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
    "female",
    "urban",
    "year_2020",
]

FIG2_LABELS = {
    "poor_health": "自评健康较差",
    "chronic_count": "慢性病数量",
    "adl_limit": "ADL受限",
    "iadl_limit": "IADL受限",
    "depression_high": "抑郁风险",
    "total_cognition_w": "认知能力",
    "family_care_index_v1": "家庭压力指数",
    "co_reside_child": "与子女同住",
    "care_elder_or_disabled": "照护老人或失能成员",
    "economic_pressure_index_v1": "经济压力指数",
    "log_hhcperc_v1_w": "家庭人均收入",
    "log_medical_expense_w": "医疗支出",
    "female": "女性",
    "urban": "城镇",
    "year_2020": "2020年",
}

SUITABILITY_LABELS = {
    "suitable_for_flexible_delay": "适配继续劳动型",
    "pressure_driven_work": "压力驱动继续劳动型",
    "constrained_working": "带约束继续劳动型",
    "constrained_exit": "健康/家庭约束退出型",
    "potential_labor_supply": "潜在劳动供给型",
    "support_priority": "保障支持重点型",
    "unknown": "无法判断",
}

SUITABILITY_ORDER = [
    "suitable_for_flexible_delay",
    "pressure_driven_work",
    "constrained_working",
    "constrained_exit",
    "potential_labor_supply",
    "support_priority",
    "unknown",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


def percent_text(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def write_excel(path: Path, data: pd.DataFrame | dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if isinstance(data, dict):
            for sheet_name, df in data.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
        else:
            data.to_excel(writer, index=False, sheet_name="Sheet1")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            as_float = float(text)
            if as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            pass
    return text


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_excel_sheets(path: Path) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(path)
    return {sheet: excel.parse(sheet) for sheet in excel.sheet_names}


def build_fig1_data(source_path: Path) -> pd.DataFrame:
    sheets = read_excel_sheets(source_path)
    rows: list[pd.DataFrame] = []
    wanted = ["age_group_stage5", "female", "urban", "poor_health", "family_high_stage5", "econ_high_stage5"]

    for sheet_name, df in sheets.items():
        temp = df.copy()
        temp.columns = [str(col).strip() for col in temp.columns]
        lower_map = {str(col).strip().lower(): col for col in temp.columns}
        if {"group_var", "group_value", "labor_rate"}.issubset({c.lower() for c in temp.columns}):
            group_var_col = lower_map["group_var"]
            group_value_col = lower_map["group_value"]
            labor_rate_col = lower_map["labor_rate"]
            n_col = lower_map.get("n")
            sub = temp[temp[group_var_col].astype(str).isin(wanted)].copy()
            if not sub.empty:
                out = pd.DataFrame(
                    {
                        "source_sheet": sheet_name,
                        "group_var": sub[group_var_col].astype(str),
                        "group_value_raw": sub[group_value_col],
                        "labor_rate": pd.to_numeric(sub[labor_rate_col], errors="coerce"),
                        "sample_size": pd.to_numeric(sub[n_col], errors="coerce") if n_col else np.nan,
                    }
                )
                rows.append(out)
            continue

        for group_var in wanted:
            matched_cols = [col for col in temp.columns if str(col).strip().lower() == group_var.lower()]
            rate_cols = [col for col in temp.columns if "rate" in str(col).strip().lower()]
            n_cols = [col for col in temp.columns if str(col).strip().lower() in {"n", "sample_size"}]
            if matched_cols and rate_cols:
                group_col = matched_cols[0]
                rate_col = rate_cols[0]
                n_col = n_cols[0] if n_cols else None
                sub = temp[[group_col, rate_col] + ([n_col] if n_col else [])].copy()
                out = pd.DataFrame(
                    {
                        "source_sheet": sheet_name,
                        "group_var": group_var,
                        "group_value_raw": sub[group_col],
                        "labor_rate": pd.to_numeric(sub[rate_col], errors="coerce"),
                        "sample_size": pd.to_numeric(sub[n_col], errors="coerce") if n_col else np.nan,
                    }
                )
                rows.append(out)

    if not rows:
        raise ValueError(f"未能从 {source_path} 识别图1所需分组数据。")

    result = pd.concat(rows, ignore_index=True).dropna(subset=["labor_rate"])
    result["group_value"] = result["group_value_raw"].map(normalize_text)
    result["panel_title"] = result["group_var"].map(lambda x: GROUP_LABELS.get(x, (x, {}))[0])
    result["group_label"] = result.apply(
        lambda row: GROUP_LABELS.get(row["group_var"], ("", {}))[1].get(row["group_value"], row["group_value"]),
        axis=1,
    )

    ordered_rows: list[pd.DataFrame] = []
    for key in wanted:
        sub = result[result["group_var"] == key].copy()
        label_map = GROUP_LABELS[key][1]
        order_lookup = {normalize_text(k): idx for idx, k in enumerate(label_map.keys())}
        sub["sort_order"] = sub["group_value"].map(lambda x: order_lookup.get(x, 999))
        sub = sub.sort_values(["sort_order", "group_label"]).drop(columns="sort_order")
        ordered_rows.append(sub)
    return pd.concat(ordered_rows, ignore_index=True)


def generate_fig1(paths_out: list[Path], missing_inputs: list[str]) -> Path | None:
    source_path = STAGE5_DIR / "paper_table2_group_labor_rates_final.xlsx"
    if not source_path.exists():
        missing_inputs.append(str(source_path))
        return None

    data = build_fig1_data(source_path)
    fig_path = OUTPUT_DIR / "fig1_group_labor_rates.png"
    data_path = OUTPUT_DIR / "fig1_group_labor_rates_data.xlsx"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    colors = ["#35618f", "#cc7b39", "#4f8a5b", "#b25050", "#7c64a8", "#7d8f3d"]
    for idx, (group_var, sub) in enumerate(data.groupby("group_var", sort=False)):
        ax = axes[idx]
        bars = ax.bar(sub["group_label"], sub["labor_rate"], color=colors[idx], width=0.65)
        ax.set_title(GROUP_LABELS[group_var][0], fontsize=12)
        ax.set_ylim(0, min(1.0, max(0.8, sub["labor_rate"].max() * 1.18)))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=20)
        if idx % 3 == 0:
            ax.set_ylabel("劳动参与率")
        for bar, value in zip(bars, sub["labor_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, percent_text(value), ha="center", va="bottom", fontsize=9)
    fig.suptitle("不同群体中老年劳动参与率比较", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_excel(data_path, data)
    paths_out.extend([fig_path, data_path])
    return fig_path


def load_model5_logit() -> tuple[pd.DataFrame | None, str | None]:
    stage5_path = STAGE5_DIR / "paper_table3_logit_main_final.xlsx"
    if stage5_path.exists():
        sheets = read_excel_sheets(stage5_path)
        if "long_format" in sheets:
            df = sheets["long_format"].copy()
            df = df[df["model"].astype(str) == "Model_5"].copy()
            if not df.empty:
                return df, str(stage5_path)

    training_path = TRAINING_DIR / "logit_results_v1.xlsx"
    if training_path.exists():
        excel = pd.ExcelFile(training_path)
        if "Model_5" in excel.sheet_names:
            return excel.parse("Model_5"), str(training_path)
    return None, None


def generate_fig2(paths_out: list[Path], missing_inputs: list[str]) -> Path | None:
    df, source = load_model5_logit()
    if df is None or source is None:
        missing_inputs.append(str(STAGE5_DIR / "paper_table3_logit_main_final.xlsx"))
        missing_inputs.append(str(TRAINING_DIR / "logit_results_v1.xlsx"))
        return None

    temp = df.copy()
    temp.columns = [str(c).strip() for c in temp.columns]
    temp = temp[temp["variable"].astype(str).isin(FIG2_VARIABLES)].copy()
    if temp.empty:
        raise ValueError("图2所需变量在 Model 5 结果中未找到。")

    temp["order"] = temp["variable"].map({var: idx for idx, var in enumerate(FIG2_VARIABLES)})
    temp["variable_cn"] = temp["variable"].map(FIG2_LABELS)
    temp["odds_ratio"] = pd.to_numeric(temp["odds_ratio"], errors="coerce")
    temp["std_err"] = pd.to_numeric(temp.get("std_err"), errors="coerce")
    temp["coef"] = pd.to_numeric(temp.get("coef"), errors="coerce")
    has_ci = temp["coef"].notna().all() and temp["std_err"].notna().all()
    if has_ci:
        temp["ci_low"] = np.exp(temp["coef"] - 1.96 * temp["std_err"])
        temp["ci_high"] = np.exp(temp["coef"] + 1.96 * temp["std_err"])
    else:
        temp["ci_low"] = np.nan
        temp["ci_high"] = np.nan

    temp = temp.sort_values("order", ascending=False).reset_index(drop=True)
    temp["source_file"] = source
    temp["has_95ci"] = has_ci

    fig_path = OUTPUT_DIR / "fig2_logit_or_forest.png"
    data_path = OUTPUT_DIR / "fig2_logit_or_forest_data.xlsx"

    fig, ax = plt.subplots(figsize=(10, 7.5))
    y_pos = np.arange(len(temp))
    if has_ci:
        xerr = np.vstack([temp["odds_ratio"] - temp["ci_low"], temp["ci_high"] - temp["odds_ratio"]])
        ax.errorbar(temp["odds_ratio"], y_pos, xerr=xerr, fmt="o", color="#2f6f91", ecolor="#93b7c9", capsize=3, elinewidth=1.3)
        ax.set_xlabel("OR（95%置信区间）")
    else:
        ax.scatter(temp["odds_ratio"], y_pos, color="#2f6f91", s=35)
        ax.set_xlabel("OR")
    ax.axvline(1.0, color="#7f7f7f", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(temp["variable_cn"])
    ax.set_title("Logit主模型关键变量OR森林图")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    xmin = float(np.nanmin(temp["ci_low"] if has_ci else temp["odds_ratio"]))
    xmax = float(np.nanmax(temp["ci_high"] if has_ci else temp["odds_ratio"]))
    left = min(0.2, xmin * 0.9) if xmin > 0 else 0
    right = xmax * 1.12
    ax.set_xlim(left, right)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_excel(data_path, temp)
    paths_out.extend([fig_path, data_path])
    return fig_path


def collect_ml_performance() -> tuple[pd.DataFrame | None, list[str]]:
    searched: list[str] = []
    candidates = [
        STAGE5_DIR / "paper_table5_ml_performance_final.xlsx",
        TRAINING_DIR / "ml_metrics_v1.xlsx",
        STAGE4_DIR / "shap_model_metrics_v1.xlsx",
    ]
    baseline_df: pd.DataFrame | None = None
    shap_df: pd.DataFrame | None = None

    for path in candidates:
        searched.append(str(path))
        if not path.exists():
            continue
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet)
            cols = {str(c).strip().lower(): c for c in df.columns}
            if "model" in cols and {"accuracy", "f1"}.issubset(cols):
                temp = df.rename(columns={cols[k]: k for k in cols})
                if "roc_auc" in temp.columns:
                    temp = temp.copy()
                    if "scenario" in temp.columns:
                        baseline = temp[temp["scenario"].astype(str).str.lower() == "baseline"].copy()
                        if not baseline.empty:
                            baseline_df = baseline
                    else:
                        baseline_df = temp
            if "scheme" in cols and {"accuracy", "f1", "roc_auc"}.issubset(cols):
                shap_df = df.rename(columns={cols[k]: k for k in cols}).copy()

    final_rows: list[dict[str, object]] = []
    for model_name in ["LogisticRegression", "XGBoost", "LightGBM"]:
        row = None
        if baseline_df is not None and "model" in baseline_df.columns:
            matched = baseline_df[baseline_df["model"].astype(str) == model_name]
            if not matched.empty:
                row = matched.iloc[0]
        if row is None and model_name == "LightGBM" and shap_df is not None:
            matched = shap_df[shap_df["scheme"].astype(str).str.lower() == "full_feature"]
            if not matched.empty:
                row = matched.iloc[0]
        if row is not None:
            final_rows.append(
                {
                    "model": model_name,
                    "AUC": pd.to_numeric(row.get("roc_auc"), errors="coerce"),
                    "F1": pd.to_numeric(row.get("f1"), errors="coerce"),
                    "Accuracy": pd.to_numeric(row.get("accuracy"), errors="coerce"),
                }
            )
    if not final_rows:
        return None, searched
    return pd.DataFrame(final_rows), searched


def generate_fig3(paths_out: list[Path], missing_inputs: list[str]) -> Path | None:
    data, searched = collect_ml_performance()
    if data is None:
        missing_inputs.extend(searched)
        return None

    fig_path = OUTPUT_DIR / "fig3_ml_performance_comparison.png"
    data_path = OUTPUT_DIR / "fig3_ml_performance_comparison_data.xlsx"

    metrics = ["AUC", "F1", "Accuracy"]
    if len(metrics) * len(data) > 9:
        metrics = ["AUC", "F1"]
    long_df = data.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data))
    width = 0.22 if len(metrics) == 3 else 0.32
    palette = {"AUC": "#35618f", "F1": "#cc7b39", "Accuracy": "#4f8a5b"}
    for idx, metric in enumerate(metrics):
        subset = data[["model", metric]].copy()
        offsets = x + (idx - (len(metrics) - 1) / 2) * width
        bars = ax.bar(offsets, subset[metric], width=width, label=metric, color=palette[metric])
        for bar, value in zip(bars, subset[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(data["model"])
    ax.set_ylim(0, min(1.0, max(0.9, long_df["value"].max() * 1.08)))
    ax.set_ylabel("指标值")
    ax.set_title("机器学习模型性能比较")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_excel(data_path, {"wide": data, "long": long_df})
    paths_out.extend([fig_path, data_path])
    return fig_path


def copy_file_if_exists(src: Path, dst: Path, missing_inputs: list[str], paths_out: list[Path]) -> bool:
    if not src.exists():
        missing_inputs.append(str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    paths_out.append(dst)
    return True


def load_retirement_suitability_source() -> tuple[pd.DataFrame | None, str | None]:
    primary = STAGE6_DIR / "retirement_suitability_distribution_v1.xlsx"
    if primary.exists():
        excel = pd.ExcelFile(primary)
        preferred = [s for s in ["suitability_relevant_only", "suitability_full_sample"] if s in excel.sheet_names]
        for sheet in preferred:
            df = excel.parse(sheet)
            if "retirement_suitability_v1" in df.columns:
                return df, f"{primary}::{sheet}"

    for path in STAGE5_DIR.glob("*.xlsx"):
        if "suitability" not in path.stem.lower() and "retirement_suitability" not in path.stem.lower():
            continue
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            df = excel.parse(sheet)
            if "retirement_suitability_v1" in df.columns:
                return df, f"{path}::{sheet}"
    return None, None


def generate_fig6(paths_out: list[Path], missing_inputs: list[str]) -> Path | None:
    df, source = load_retirement_suitability_source()
    if df is None or source is None:
        missing_inputs.append(str(STAGE6_DIR / "retirement_suitability_distribution_v1.xlsx"))
        return None

    temp = df.copy()
    if "sample_size" not in temp.columns:
        raise ValueError("图6数据缺少 sample_size 列。")
    total = pd.to_numeric(temp["sample_size"], errors="coerce").sum()
    temp = temp.copy()
    temp["sample_size"] = pd.to_numeric(temp["sample_size"], errors="coerce")
    temp["share"] = temp["sample_size"] / total
    temp["category_code"] = temp["retirement_suitability_v1"].astype(str)
    temp["category_cn"] = temp["category_code"].map(SUITABILITY_LABELS).fillna(temp["category_code"])
    temp["order"] = temp["category_code"].map({code: idx for idx, code in enumerate(SUITABILITY_ORDER)}).fillna(999)
    temp = temp.sort_values("order", ascending=False).reset_index(drop=True)
    temp["source_file"] = source

    fig_path = OUTPUT_DIR / "fig6_retirement_suitability_distribution.png"
    data_path = OUTPUT_DIR / "fig6_retirement_suitability_distribution_data.xlsx"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(temp["category_cn"], temp["share"], color="#587b5a")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("占准退休年龄段比例")
    ax.set_ylabel("适配性类别")
    ax.set_title("准退休年龄段适配性分层分布")
    ax.set_xlim(0, min(1.0, temp["share"].max() * 1.18))
    for bar, value in zip(bars, temp["share"]):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, percent_text(value), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_excel(data_path, temp)
    paths_out.extend([fig_path, data_path])
    return fig_path


def build_fig7_data(source_path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(source_path)
    mapping = {
        "by_gender": ("性别", "female", {"0": "男性", "1": "女性", 0: "男性", 1: "女性"}),
        "by_urban": ("城乡", "urban", {"0": "农村", "1": "城镇", 0: "农村", 1: "城镇"}),
        "by_health": ("健康状态", "poor_health", {"0": "健康状况较好", "1": "自评健康较差", 0: "健康状况较好", 1: "自评健康较差"}),
        "by_family_pressure": ("家庭压力高低", "family_pressure_group", {"low": "家庭压力较低", "high": "家庭压力较高"}),
        "by_economic_pressure": ("经济压力高低", "economic_pressure_group", {"low": "经济压力较低", "high": "经济压力较高"}),
    }
    rows: list[pd.DataFrame] = []
    for sheet, (panel_title, group_col, label_map) in mapping.items():
        if sheet not in excel.sheet_names:
            continue
        df = excel.parse(sheet).copy()
        if group_col not in df.columns or "labor_participation_rate" not in df.columns:
            continue
        out = pd.DataFrame(
            {
                "panel_title": panel_title,
                "group_value_raw": df[group_col],
                "group_label": df[group_col].map(lambda x: label_map.get(normalize_text(x), label_map.get(x, normalize_text(x)))),
                "labor_rate": pd.to_numeric(df["labor_participation_rate"], errors="coerce"),
                "sample_size": pd.to_numeric(df.get("sample_size"), errors="coerce"),
                "source_sheet": sheet,
            }
        )
        rows.append(out)
    if not rows:
        raise ValueError(f"未能从 {source_path} 识别图7所需数据。")
    return pd.concat(rows, ignore_index=True)


def generate_fig7(paths_out: list[Path], missing_inputs: list[str]) -> Path | None:
    source_path = STAGE6_DIR / "near_retirement_descriptive_tables_v1.xlsx"
    if not source_path.exists():
        missing_inputs.append(str(source_path))
        return None

    data = build_fig7_data(source_path)
    fig_path = OUTPUT_DIR / "fig7_near_retirement_group_labor_rates.png"
    data_path = OUTPUT_DIR / "fig7_near_retirement_group_labor_rates_data.xlsx"

    panels = data["panel_title"].drop_duplicates().tolist()
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes_array = np.atleast_1d(axes).flatten()
    colors = ["#35618f", "#cc7b39", "#4f8a5b", "#b25050", "#7c64a8"]
    for idx, panel in enumerate(panels):
        ax = axes_array[idx]
        sub = data[data["panel_title"] == panel].copy()
        bars = ax.bar(sub["group_label"], sub["labor_rate"], color=colors[idx], width=0.65)
        ax.set_title(panel)
        ax.set_ylim(0, min(1.0, max(0.8, sub["labor_rate"].max() * 1.18)))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(axis="x", rotation=18)
        if idx % ncols == 0:
            ax.set_ylabel("劳动参与率")
        for bar, value in zip(bars, sub["labor_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, percent_text(value), ha="center", va="bottom", fontsize=9)
    for idx in range(len(panels), len(axes_array)):
        axes_array[idx].axis("off")
    fig.suptitle("准退休年龄段分组劳动参与率比较", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_excel(data_path, data)
    paths_out.extend([fig_path, data_path])
    return fig_path


def generate_caption_md(paths_out: list[Path]) -> Path:
    path = OUTPUT_DIR / "paper_figure_caption_and_placement_v1.md"
    content = """# 论文图表使用说明

| 文件名 | 推荐图号 | 推荐放置章节 | 图题 | 正文解释重点 | 推荐位置 |
|---|---|---|---|---|---|
| fig1_group_labor_rates.png | 图1 | 第四节 | 不同群体中老年劳动参与率比较 | 对比年龄、性别、城乡、健康、家庭压力和经济压力分组下的劳动参与率差异，作为描述性事实基础。 | 正文 |
| fig2_logit_or_forest.png | 图2 | 第五节 | Logit主模型关键变量OR森林图 | 展示健康、人力资本、家庭照护、经济压力和人口学变量对劳动参与的方向与强度。 | 正文 |
| fig3_ml_performance_comparison.png | 图3 | 第六节或附录 | 机器学习模型性能比较 | 对比 LogisticRegression、XGBoost、LightGBM 的 AUC、F1 与 Accuracy，说明机器学习识别能力。 | 可选正文/附录 |
| fig4_shap_bar_main_clean.png | 图4 | 第六节 | SHAP主模型特征重要性条形图 | 突出主模型中关键变量的重要性排序，承接机器学习模型解释。 | 正文 |
| fig5_shap_beeswarm_main_clean.png | 图5 | 第六节 | SHAP主模型蜂群图 | 展示关键变量取值变化对劳动参与预测方向与异质性的影响。 | 正文 |
| fig6_retirement_suitability_distribution.png | 图6 | 第七节 | 准退休年龄段适配性分层分布 | 展示准退休年龄段样本在不同适配性类别中的占比，支撑分层支持政策建议。 | 正文 |
| fig7_near_retirement_group_labor_rates.png | 图7 | 第七节或附录 | 准退休年龄段分组劳动参与率比较 | 展示准退休年龄段内部在性别、城乡、健康与压力维度上的劳动参与差异。 | 可选正文/附录 |
| appendix_fig_shap_bar_no_totmet.png | 附图A1 | 附录 | 去除 TOTMET 方案 SHAP 条形图 | 用于和主模型 SHAP 结果做稳健性对照。 | 附录 |
| appendix_fig_shap_bar_full_feature.png | 附图A2 | 附录 | 全特征方案 SHAP 条形图 | 展示完整特征集下的重要性排序结果。 | 附录 |
| appendix/poor_health.png 等 dependence plots | 附图A3-A8 | 附录 | SHAP dependence plots | 细化关键变量的边际解释模式，辅助说明非线性和阈值特征。 | 附录 |

## 正文推荐保留图

1. 图1 `fig1_group_labor_rates.png`
2. 图2 `fig2_logit_or_forest.png`
3. 图4 `fig4_shap_bar_main_clean.png`
4. 图5 `fig5_shap_beeswarm_main_clean.png`
5. 图6 `fig6_retirement_suitability_distribution.png`
"""
    path.write_text(content, encoding="utf-8")
    paths_out.append(path)
    return path


def main() -> None:
    ensure_dirs()
    configure_matplotlib()

    output_paths: list[Path] = []
    missing_inputs: list[str] = []
    generated_main_figures = 0
    copied_shap_figures = 0
    generated_appendix_figures = 0

    for generator in [generate_fig1, generate_fig2, generate_fig3, generate_fig6, generate_fig7]:
        result = generator(output_paths, missing_inputs)
        if result is not None:
            generated_main_figures += 1

    shap_main_pairs = [
        (STAGE4_DIR / "fig_shap_bar_main_clean.png", OUTPUT_DIR / "fig4_shap_bar_main_clean.png"),
        (STAGE4_DIR / "fig_shap_beeswarm_main_clean.png", OUTPUT_DIR / "fig5_shap_beeswarm_main_clean.png"),
    ]
    for src, dst in shap_main_pairs:
        if copy_file_if_exists(src, dst, missing_inputs, output_paths):
            copied_shap_figures += 1

    appendix_pairs = [
        (STAGE4_DIR / "fig_shap_bar_no_totmet.png", OUTPUT_DIR / "appendix_fig_shap_bar_no_totmet.png"),
        (STAGE4_DIR / "fig_shap_bar_full_feature.png", OUTPUT_DIR / "appendix_fig_shap_bar_full_feature.png"),
    ]
    for src, dst in appendix_pairs:
        if copy_file_if_exists(src, dst, missing_inputs, output_paths):
            generated_appendix_figures += 1

    dependence_dir = STAGE4_DIR / "dependence_main_clean"
    dependence_vars = [
        "poor_health",
        "iadl_limit",
        "family_care_index_v1",
        "economic_pressure_index_v1",
        "log_hhcperc_v1_w",
        "log_intergen_support_out_w",
    ]
    if dependence_dir.exists():
        for var in dependence_vars:
            src = dependence_dir / f"{var}.png"
            dst = APPENDIX_DIR / f"{var}.png"
            if copy_file_if_exists(src, dst, missing_inputs, output_paths):
                generated_appendix_figures += 1
    else:
        missing_inputs.append(str(dependence_dir))

    generate_caption_md(output_paths)

    print(f"成功生成的正文图数量：{generated_main_figures}")
    print(f"成功复制的SHAP图数量：{copied_shap_figures}")
    print(f"成功生成的附录图数量：{generated_appendix_figures}")
    if missing_inputs:
        print("缺失的输入文件列表：")
        for item in sorted(dict.fromkeys(missing_inputs)):
            print(f"- {item}")
    else:
        print("缺失的输入文件列表：无")
    print("推荐正文保留的5张图：")
    for item in [
        "图1 fig1_group_labor_rates.png",
        "图2 fig2_logit_or_forest.png",
        "图4 fig4_shap_bar_main_clean.png",
        "图5 fig5_shap_beeswarm_main_clean.png",
        "图6 fig6_retirement_suitability_distribution.png",
    ]:
        print(f"- {item}")
    print("所有输出文件路径：")
    for path in sorted(dict.fromkeys(output_paths)):
        print(f"- {path}")


if __name__ == "__main__":
    main()
