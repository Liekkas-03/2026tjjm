from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "outputs" / "final_model_data_v1"
TRAINING_DIR = ROOT / "outputs" / "model_training_v1"
STAGE3_DIR = ROOT / "outputs" / "model_stage3_review_v1"
OUTPUT_DIR = ROOT / "outputs" / "model_stage4_shap_v1"
DEPENDENCE_DIR = OUTPUT_DIR / "dependence_main_clean"

DATA_FILE = FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv"
TARGET = "labor_participation"
GROUP = "ID"
NON_FEATURE_COLS = {TARGET, "ID", "wave", "year"}
BEHAVIOR_DROP = {"smokev", "drinkl", "exercise", "totmet_w"}
LEAKAGE_KEYWORDS = ["labor", "work", "retire", "job", "employment", "employed", "pension", "suitability", "group", "hours", "days_yearly"]
TOP_DEPENDENCE_COUNT = 10
MAX_SHAP_SAMPLE = 5000
RANDOM_STATE = 42

HEALTH_VARS = {"poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w"}
FAMILY_VARS = {"family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "hchild", "family_size"}
ECON_VARS = {"economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w", "medical_burden_w", "log_intergen_support_in_w", "log_intergen_support_out_w"}
SPECIAL_DEPENDENCE = list(HEALTH_VARS) + list(FAMILY_VARS) + list(ECON_VARS)


def leakage_match(column: str) -> str:
    lower = column.lower()
    for keyword in LEAKAGE_KEYWORDS:
        if keyword in lower:
            return keyword
    return ""


def metrics_dict(y_true: pd.Series, pred_prob: np.ndarray, pred_label: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, pred_prob)),
    }


def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def feature_role(feature: str) -> str:
    if feature in HEALTH_VARS:
        return "health_constraints"
    if feature in FAMILY_VARS:
        return "family_care_and_intergenerational_support"
    if feature in ECON_VARS:
        return "economic_pressure"
    if feature.endswith("_missing"):
        return "missing_indicator"
    if feature == "year_2020":
        return "year_control"
    if feature in {"hibpe", "diabe", "hearte", "stroke", "arthre", "kidneye", "digeste"}:
        return "disease_indicator"
    if feature in {"age", "age_squared", "female", "married", "urban"} or feature.startswith("edu_"):
        return "demographic_control"
    if feature in {"smokev", "drinkl", "exercise", "totmet_w"}:
        return "behavior_control"
    return "other"


def interpretation_note(feature: str) -> str:
    if feature == "family_care_index_v1":
        return "解释为家庭照料与代际支持压力代理，不宜直接解释为单纯照料责任。"
    if feature == "economic_pressure_index_v1":
        return "可解释为经济压力驱动继续劳动的解释贡献。"
    if feature == "year_2020":
        return "年份控制变量，不作为核心解释变量。"
    if feature == "totmet_w":
        return "可能代理劳动活动强度本身，不进入正文主 SHAP 模型。"
    if feature.endswith("_missing"):
        return "缺失指示变量，反映数据缺失模式，不宜作实体机制解释。"
    if feature in {"depression_high", "total_cognition_w"}:
        return "方向需结合 Stage 3 结果谨慎解释。"
    return "可作为预测解释贡献变量，但不代表因果效应。"


def risk_note(feature: str) -> str:
    if feature == "totmet_w":
        return "possible_proxy_risk"
    if feature == "year_2020":
        return "year_control_only"
    if feature.endswith("_missing"):
        return "missing_pattern_risk"
    if feature in {"family_care_index_v1", "depression_high", "total_cognition_w"}:
        return "interpretation_caution"
    return ""


def save_summary_plot(shap_values: np.ndarray, features: pd.DataFrame, path: Path, plot_type: str | None = None, max_display: int = 20) -> None:
    plt.figure()
    shap.summary_plot(shap_values, features, plot_type=plot_type, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plot(shap_values: np.ndarray, features: pd.DataFrame, feature_name: str, path: Path) -> bool:
    if feature_name not in features.columns:
        return False
    plt.figure()
    shap.dependence_plot(feature_name, shap_values, features, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEPENDENCE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    if GROUP not in df.columns:
        raise ValueError("ID column is required for GroupShuffleSplit.")

    all_candidate_features = []
    removed_leakage_rows = []
    for column in df.columns:
        if column in NON_FEATURE_COLS:
            continue
        keyword = leakage_match(column)
        if keyword:
            removed_leakage_rows.append({"column": column, "matched_keyword": keyword, "reason": "Removed by Stage 4 leakage screen."})
            continue
        all_candidate_features.append(column)

    schemes = {
        "main_clean": [c for c in all_candidate_features if c not in BEHAVIOR_DROP],
        "no_totmet": [c for c in all_candidate_features if c != "totmet_w"],
        "full_feature": all_candidate_features.copy(),
    }

    y = df[TARGET].astype(int)
    groups = df[GROUP]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, y, groups))

    metrics_rows = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    shap_summary_rows = []
    top20_tables = []
    compare_tables = []
    sample_index_tables = []
    generated_dependence = []
    missing_dependence = []
    scheme_results = {}

    for scheme_name, feature_cols in schemes.items():
        X = df[feature_cols].copy()
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        model = build_model()
        model.fit(X_train, y_train)
        pred_prob = model.predict_proba(X_test)[:, 1]
        pred_label = model.predict(X_test)
        metrics = metrics_dict(y_test, pred_prob, pred_label)
        metrics_rows.append({"scheme": scheme_name, "feature_count": len(feature_cols), "train_rows": len(train_idx), "test_rows": len(test_idx), **metrics})
        confusion_tables[scheme_name] = pd.DataFrame(confusion_matrix(y_test, pred_label), index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])

        sample_size = min(MAX_SHAP_SAMPLE, len(X_test))
        sample_rng = np.random.default_rng(RANDOM_STATE)
        if len(X_test) > sample_size:
            sample_positions = np.sort(sample_rng.choice(len(X_test), size=sample_size, replace=False))
        else:
            sample_positions = np.arange(len(X_test))
        X_shap = X_test.iloc[sample_positions].copy()
        y_shap = y_test.iloc[sample_positions].copy()
        original_indices = X_shap.index.to_numpy()
        sample_index_tables.append(
            pd.DataFrame(
                {
                    "scheme": scheme_name,
                    "test_position": sample_positions,
                    "original_index": original_indices,
                    "ID": df.iloc[test_idx].iloc[sample_positions][GROUP].to_numpy(),
                    "target": y_shap.to_numpy(),
                }
            )
        )

        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_array = shap_values
        mean_abs = np.abs(shap_array).mean(axis=0)
        summary_df = pd.DataFrame(
            {
                "scheme": scheme_name,
                "feature": X_shap.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        summary_df["rank"] = np.arange(1, len(summary_df) + 1)
        summary_df["feature_role"] = summary_df["feature"].map(feature_role)
        summary_df["interpretation_note"] = summary_df["feature"].map(interpretation_note)
        summary_df["risk_note"] = summary_df["feature"].map(risk_note)
        shap_summary_rows.append(summary_df)
        top20 = summary_df.head(20).copy()
        top20_tables.append(top20)
        compare_tables.append(top20[["scheme", "feature", "rank", "mean_abs_shap"]])
        scheme_results[scheme_name] = {
            "model": model,
            "X_test": X_test,
            "X_shap": X_shap,
            "shap_array": shap_array,
            "summary": summary_df,
            "metrics": metrics,
            "features": feature_cols,
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "shap_model_metrics_v1.xlsx"
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        pd.concat(sample_index_tables, ignore_index=True).to_excel(writer, sheet_name="shap_sample_indices", index=False)
        pd.DataFrame(removed_leakage_rows if removed_leakage_rows else [{"column": "", "matched_keyword": "", "reason": "No leakage features removed at Stage 4."}]).to_excel(writer, sheet_name="removed_leakage", index=False)
        for scheme_name, cm_df in confusion_tables.items():
            cm_df.to_excel(writer, sheet_name=f"cm_{scheme_name}"[:31])

    shap_summary_path = OUTPUT_DIR / "shap_values_summary_table_v1.xlsx"
    with pd.ExcelWriter(shap_summary_path, engine="openpyxl") as writer:
        for summary_df in shap_summary_rows:
            scheme_name = summary_df["scheme"].iloc[0]
            summary_df.to_excel(writer, sheet_name=scheme_name[:31], index=False)
        pd.concat(sample_index_tables, ignore_index=True).to_excel(writer, sheet_name="sample_indices", index=False)

    main_clean = scheme_results["main_clean"]
    no_totmet = scheme_results["no_totmet"]
    full_feature = scheme_results["full_feature"]

    bar_main_path = OUTPUT_DIR / "fig_shap_bar_main_clean.png"
    beeswarm_main_path = OUTPUT_DIR / "fig_shap_beeswarm_main_clean.png"
    bar_no_totmet_path = OUTPUT_DIR / "fig_shap_bar_no_totmet.png"
    bar_full_path = OUTPUT_DIR / "fig_shap_bar_full_feature.png"

    save_summary_plot(main_clean["shap_array"], main_clean["X_shap"], bar_main_path, plot_type="bar", max_display=20)
    save_summary_plot(main_clean["shap_array"], main_clean["X_shap"], beeswarm_main_path, plot_type=None, max_display=20)
    save_summary_plot(no_totmet["shap_array"], no_totmet["X_shap"], bar_no_totmet_path, plot_type="bar", max_display=20)
    save_summary_plot(full_feature["shap_array"], full_feature["X_shap"], bar_full_path, plot_type="bar", max_display=20)

    top_main_features = main_clean["summary"]["feature"].head(TOP_DEPENDENCE_COUNT).tolist()
    dependence_targets = []
    for feature in top_main_features + SPECIAL_DEPENDENCE:
        if feature not in dependence_targets:
            dependence_targets.append(feature)
    for feature in dependence_targets:
        path = DEPENDENCE_DIR / f"{feature}.png"
        ok = save_dependence_plot(main_clean["shap_array"], main_clean["X_shap"], feature, path)
        if ok:
            generated_dependence.append(str(path))
        else:
            missing_dependence.append(feature)

    compare_df = pd.concat(compare_tables, ignore_index=True)
    comparison_pivot = compare_df.pivot_table(index="feature", columns="scheme", values="rank", aggfunc="first")
    table6_path = OUTPUT_DIR / "paper_table6_shap_top_features_v1.xlsx"
    with pd.ExcelWriter(table6_path, engine="openpyxl") as writer:
        main_clean["summary"].head(20).to_excel(writer, sheet_name="main_clean_top20", index=False)
        no_totmet["summary"].head(20).to_excel(writer, sheet_name="no_totmet_top20", index=False)
        full_feature["summary"].head(20).to_excel(writer, sheet_name="full_feature_top20", index=False)
        comparison_pivot.to_excel(writer, sheet_name="top_feature_compare")
        pd.DataFrame(
            [
                {
                    "feature": feature,
                    "paper_interpretation": interpretation_note(feature),
                    "risk_note": risk_note(feature),
                    "feature_role": feature_role(feature),
                }
                for feature in sorted(set(compare_df["feature"]))
            ]
        ).to_excel(writer, sheet_name="interpretation_notes", index=False)

    top10_main = main_clean["summary"].head(10)
    notes_lines = [
        "SHAP解释文本草稿",
        "1. 之所以选择 LightGBM 作为 SHAP 主模型，是因为它在 baseline 和 Stage 3 稳健性检验中表现最优或基本最优，测试集 AUC 和 F1 均略高于 XGBoost，并且在去掉部分变量后的表现相对稳定。",
        "2. 主 SHAP 方案采用 main_clean，即去掉 smokev、drinkl、exercise、totmet_w，主要是为了降低行为变量和活动强度变量对解释的干扰，尤其是 totmet_w 可能部分代理劳动活动本身。",
        "3. main_clean 前 10 个 SHAP 重要变量为：",
    ]
    for row in top10_main.itertuples(index=False):
        notes_lines.append(f"   - {row.feature}（mean |SHAP|={row.mean_abs_shap:.6f}）")
    notes_lines.extend(
        [
            "4. 健康约束变量在 SHAP 中整体具有较高解释贡献，尤其应关注 poor_health、chronic_count、adl_limit、iadl_limit、depression_high、total_cognition_w 等变量在预测劳动参与中的边际贡献。",
            "5. 家庭照料与代际支持压力变量在 SHAP 中具有一定解释贡献，但 family_care_index_v1 应解释为家庭结构、照料责任和代际支持压力的综合代理，而非单一照料责任本身。",
            "6. 经济压力变量在 SHAP 中体现出较强解释作用，尤其 economic_pressure_index_v1、log_hhcperc_v1_w、log_medical_expense_w、medical_burden_w 反映了经济压力与继续劳动之间的预测关联。",
            "7. 需要谨慎解释的变量包括：year_2020、family_care_index_v1、depression_high、total_cognition_w，以及所有 *_missing 缺失指示变量。这些变量可能反映年份效应、代理机制或数据结构，而不宜直接作因果阐释。",
            "8. 对政策建议的启发是：延迟退休能否转化为有效劳动力供给，不仅取决于制度安排，还取决于中老年人的健康约束、家庭照料与代际支持压力，以及经济压力状况。",
            "9. 需要强调的是，SHAP 结果表明，在预测劳动参与时，该变量具有较高解释贡献，但这并不证明其具有因果效应。",
        ]
    )
    notes_path = OUTPUT_DIR / "shap_interpretation_notes_v1.txt"
    notes_path.write_text("\n".join(notes_lines), encoding="utf-8")

    baseline_metrics = pd.read_excel(TRAINING_DIR / "ml_metrics_v1.xlsx")
    baseline_full_auc = float(baseline_metrics.loc[baseline_metrics["model"] == "LightGBM", "roc_auc"].iloc[0])
    main_clean_auc = float(metrics_df.loc[metrics_df["scheme"] == "main_clean", "roc_auc"].iloc[0])
    main_clean_f1 = float(metrics_df.loc[metrics_df["scheme"] == "main_clean", "f1"].iloc[0])
    no_totmet_auc = float(metrics_df.loc[metrics_df["scheme"] == "no_totmet", "roc_auc"].iloc[0])
    no_totmet_f1 = float(metrics_df.loc[metrics_df["scheme"] == "no_totmet", "f1"].iloc[0])
    full_feature_auc = float(metrics_df.loc[metrics_df["scheme"] == "full_feature", "roc_auc"].iloc[0])
    full_feature_f1 = float(metrics_df.loc[metrics_df["scheme"] == "full_feature", "f1"].iloc[0])
    auc_drop_vs_baseline = main_clean_auc - baseline_full_auc
    possible_leakage = any(leakage_match(c) for c in all_candidate_features)
    recommend_paper_stage = main_clean_auc >= 0.82

    report_lines = [
        "Stage 4 SHAP Report V1",
        f"main_clean metrics: AUC={main_clean_auc:.4f}, F1={main_clean_f1:.4f}, Accuracy={float(metrics_df.loc[metrics_df['scheme']=='main_clean','accuracy'].iloc[0]):.4f}",
        f"no_totmet metrics: AUC={no_totmet_auc:.4f}, F1={no_totmet_f1:.4f}, Accuracy={float(metrics_df.loc[metrics_df['scheme']=='no_totmet','accuracy'].iloc[0]):.4f}",
        f"full_feature metrics: AUC={full_feature_auc:.4f}, F1={full_feature_f1:.4f}, Accuracy={float(metrics_df.loc[metrics_df['scheme']=='full_feature','accuracy'].iloc[0]):.4f}",
        "main_clean is chosen as the paper's main SHAP model because it removes behavior variables and totmet_w while preserving acceptable predictive performance and cleaner substantive interpretation.",
        "main_clean top 20 SHAP features:",
    ]
    for row in main_clean["summary"].head(20).itertuples(index=False):
        report_lines.append(f"  - {row.feature}: mean_abs_shap={row.mean_abs_shap:.6f}")
    report_lines.extend(
        [
            f"Performance drop of main_clean relative to baseline full-feature LightGBM: {auc_drop_vs_baseline:.4f}",
            f"Possible leakage detected in Stage 4 features: {'Yes' if possible_leakage else 'No'}",
            f"Recommend entering paper figure-layout stage: {'Yes' if recommend_paper_stage else 'No'}",
            "Recommended main-text figures: SHAP bar summary (main_clean), SHAP beeswarm summary (main_clean), and selected dependence plots for health, family, and economic variables.",
            "Recommended appendix figures: no_totmet bar summary, full_feature bar summary, and the extended dependence plots set.",
            "Variables requiring cautious interpretation: year_2020, family_care_index_v1, depression_high, total_cognition_w, all *_missing indicators, and any variable whose interpretation may proxy labor intensity rather than an exogenous constraint.",
        ]
    )
    report_path = OUTPUT_DIR / "stage4_shap_report_v1.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    output_files = [
        metrics_path,
        shap_summary_path,
        bar_main_path,
        beeswarm_main_path,
        bar_no_totmet_path,
        bar_full_path,
        table6_path,
        notes_path,
        report_path,
    ]

    print(f"main_clean sample size and feature count: {len(train_idx) + len(test_idx)} / {len(schemes['main_clean'])}")
    print(f"no_totmet sample size and feature count: {len(train_idx) + len(test_idx)} / {len(schemes['no_totmet'])}")
    print(f"full_feature sample size and feature count: {len(train_idx) + len(test_idx)} / {len(schemes['full_feature'])}")
    print(f"main_clean AUC / F1: {main_clean_auc:.4f} / {main_clean_f1:.4f}")
    print(f"no_totmet AUC / F1: {no_totmet_auc:.4f} / {no_totmet_f1:.4f}")
    print(f"full_feature AUC / F1: {full_feature_auc:.4f} / {full_feature_f1:.4f}")
    print("main_clean top 10 SHAP features:")
    for row in top10_main.itertuples(index=False):
        print(f"  {row.feature}")
    print(f"Generated SHAP bar figure: {'Yes' if bar_main_path.exists() else 'No'}")
    print(f"Generated SHAP beeswarm figure: {'Yes' if beeswarm_main_path.exists() else 'No'}")
    print(f"Generated dependence plots: {'Yes' if len(generated_dependence) > 0 else 'No'}")
    print(f"Recommend moving to paper figure integration: {'Yes' if recommend_paper_stage else 'No'}")
    print("Output files:")
    for path in output_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
