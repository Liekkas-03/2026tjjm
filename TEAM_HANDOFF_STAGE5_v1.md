# Team Handoff Stage 5

## 1. 这次我完成了什么工作

这轮工作已经把项目从“数据已经能跑模型”推进到了“可以直接交给组员写论文和继续整合结果”的状态。主线工作包括：

1. 基于 `v3 train ready` 数据整理出最终建模数据，形成 `v4`。
2. 对最终建模数据做信息泄露复核，保留 audited 版本，避免把可疑变量直接带进模型。
3. 跑通第一轮 baseline：
   - Logit / Probit
   - LogisticRegression
   - XGBoost
   - LightGBM
4. 做了 Stage 3 结果复核与稳健性分析，重点检查了几个解释风险较高的变量。
5. 做了 Stage 4 正式 SHAP 分析，并比较了三套特征方案。
6. 做了 Stage 5 异质性分析、论文主表主图整理，以及“实证结果”写作提纲和中文草稿。

现在这批提交文件，不是把全部过程结果都交出去，而是只保留“组员必须拿到才能接着写论文”的最小交接包。

## 2. 目前能得出什么结论

### 2.1 总体结论

目前结果整体稳定，论文主线可以概括为：

> 在延迟退休背景下，中老年劳动参与适配性并不只由年龄决定，而是同时受到健康约束、家庭照料与代际支持压力、以及经济压力的共同影响。

### 2.2 比较稳的结果

- `poor_health`、`adl_limit`、`iadl_limit` 稳定负向，说明健康受限会明显压低劳动参与。
- `chronic_count` 整体偏负向，说明慢性病负担越重，继续劳动的倾向越低。
- `co_reside_child`、`care_elder_or_disabled` 多数情况下负向，说明家庭照料责任会挤压劳动参与。
- `economic_pressure_index_v1` 正向，说明经济压力越大，继续劳动的倾向越强。
- `log_medical_expense_w` 负向，说明医疗负担与退出劳动更相关。

### 2.3 需要谨慎解释的结果

- `family_care_index_v1` 在主回归里是正向。
  这个变量不能直接写成“照料责任促进劳动参与”，更合适的说法是：
  “家庭照料与代际支持压力代理”。
- `economic_pressure_index_v1` 正向，可以解释为经济压力推动继续劳动，但不能直接写成“更适合延迟退休”。
- `year_2020` 只能作为年份控制变量，不能当作核心发现展开。
- `depression_high` 为正、`total_cognition_w` 为负，这两个方向要提醒可能存在编码、样本选择或代理效应。
- 所有 `*_missing` 变量都只作为技术性特征使用，不能作 substantive 机制解释。

### 2.4 机器学习与 SHAP 结果

- baseline 中，`LightGBM` 表现最好，AUC 约 `0.885`。
- 正式 SHAP 主模型选 `LightGBM` 是合理的。
- 正文 SHAP 主方案采用 `main_clean`，即去掉：
  - `smokev`
  - `drinkl`
  - `exercise`
  - `totmet_w`

原因不是它性能最高，而是它的解释更干净、代理风险更低。

## 3. 这次提交给组员的文件

这次只保留“必须交接”的文件。组员拿到这批文件之后，应该能够：

- 复现从 Stage 2 到 Stage 5 的核心流程
- 直接使用 audited 数据
- 直接引用论文主表和正文主图
- 直接接着写“实证结果”部分

### 3.1 脚本

用于复现核心流程：

- `data/scripts/07_finalize_model_data_v1.py`
- `data/scripts/08_audit_final_model_data_v1.py`
- `data/scripts/09_train_baseline_models_v1.py`
- `data/scripts/10_stage3_review_robustness_v1.py`
- `data/scripts/11_stage4_shap_analysis_v1.py`
- `data/scripts/12_stage5_heterogeneity_and_paper_ready_v1.py`

### 3.2 最终 audited 数据

只提交 audited 版本，避免组员混用未审计版本：

- `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv`
- `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv`
- `outputs/final_model_data_v1/final_leakage_audit_v1/final_leakage_audit_report_v1.txt`

### 3.3 论文最终可用结果包

这一部分是组员最需要直接使用的内容：

- `outputs/model_stage5_paper_ready_v1/paper_table1_descriptive_stats_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table2_group_labor_rates_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table3_logit_main_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table4_robustness_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table5_ml_performance_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table6_shap_top_features_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table7_heterogeneity_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_figures_list_v1.xlsx`
- `outputs/model_stage5_paper_ready_v1/empirical_results_outline_v1.txt`
- `outputs/model_stage5_paper_ready_v1/empirical_results_draft_v1.txt`
- `outputs/model_stage5_paper_ready_v1/paper_use_recommendation_v1.txt`

### 3.4 正文推荐主图

这批图是正文最推荐直接使用的图：

- `outputs/model_stage5_paper_ready_v1/main_text_figures/fig1_labor_rates_groups.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/fig2_logit_model5_forest.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/fig3_shap_bar_main_clean.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/fig4_shap_beeswarm_main_clean.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/poor_health.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/iadl_limit.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/family_care_index_v1.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/economic_pressure_index_v1.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/log_hhcperc_v1_w.png`
- `outputs/model_stage5_paper_ready_v1/main_text_figures/log_intergen_support_out_w.png`

## 4. 这次不提交的内容

这次我刻意没有把所有中间过程结果都交出去，原因是会让仓库又大又乱，组员也不一定真的需要。

不提交的主要是：

- `outputs/model_training_v1/`
- `outputs/model_stage3_review_v1/`
- `outputs/model_stage4_shap_v1/`
- `outputs/model_stage5_paper_ready_v1/appendix_figures/`
- `outputs/model_stage5_paper_ready_v1/heterogeneity_logit_results_v1.xlsx`
- 非 audited 的 `v4` 数据
- 各类中间审计明细表、列检查表、候选变量表

这些内容不是没用，而是更适合保留在本地作为过程记录，不适合作为“组员必须拿到”的交接材料。

## 5. 组员下一步应该做什么

### 5.1 直接进入论文写作

优先使用：

- `paper_table3_logit_main_final.xlsx`
- `paper_table4_robustness_final.xlsx`
- `paper_table5_ml_performance_final.xlsx`
- `paper_table6_shap_top_features_final.xlsx`
- `paper_table7_heterogeneity_final.xlsx`

以及：

- `fig1_labor_rates_groups.png`
- `fig2_logit_model5_forest.png`
- `fig3_shap_bar_main_clean.png`
- `fig4_shap_beeswarm_main_clean.png`

### 5.2 直接接着写“实证结果”部分

可以把下面两个文件作为写作底稿：

- `empirical_results_outline_v1.txt`
- `empirical_results_draft_v1.txt`

### 5.3 统一表述口径

正文建议统一写法：

- `family_care_index_v1`：
  “家庭照料与代际支持压力代理”
- `economic_pressure_index_v1`：
  “经济压力推动继续劳动倾向”
- `year_2020`：
  “年份控制变量”

不要写成：

- `family_care_index_v1` = 纯照料责任
- SHAP 证明因果关系
- `year_2020` = 政策效果

### 5.4 如果还要继续完善

下一步最值得做的是：

1. 把摘要、引言、文献综述、实证结果和结论整合成完整论文初稿。
2. 统一主表和主图的编号、格式和注释。
3. 对 `depression_high`、`total_cognition_w` 的异常方向补一句解释风险提示。
4. 如果篇幅紧张，再压缩一版正文图表组合。


## 7. 要找什么内容，应该去哪里看

查找导航：

- 如果要找最终建模数据：
  去 `outputs/final_model_data_v1/`
  重点看：
  - `CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv`
  - `CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv`

- 如果要确认这些最终数据能不能放心建模：
  去 `outputs/final_model_data_v1/final_leakage_audit_v1/`
  重点看：
  - `final_leakage_audit_report_v1.txt`

- 如果要看整套流程是怎么跑出来的：
  去 `data/scripts/`
  重点看：
  - `07_finalize_model_data_v1.py`
  - `08_audit_final_model_data_v1.py`
  - `09_train_baseline_models_v1.py`
  - `10_stage3_review_robustness_v1.py`
  - `11_stage4_shap_analysis_v1.py`
  - `12_stage5_heterogeneity_and_paper_ready_v1.py`

- 如果要直接写论文正文表格：
  去 `outputs/model_stage5_paper_ready_v1/`
  重点看：
  - `paper_table1_descriptive_stats_final.xlsx`
  - `paper_table2_group_labor_rates_final.xlsx`
  - `paper_table3_logit_main_final.xlsx`
  - `paper_table4_robustness_final.xlsx`
  - `paper_table5_ml_performance_final.xlsx`
  - `paper_table6_shap_top_features_final.xlsx`
  - `paper_table7_heterogeneity_final.xlsx`

- 如果要直接找正文推荐图：
  去 `outputs/model_stage5_paper_ready_v1/main_text_figures/`
  重点看：
  - `fig1_labor_rates_groups.png`
  - `fig2_logit_model5_forest.png`
  - `fig3_shap_bar_main_clean.png`
  - `fig4_shap_beeswarm_main_clean.png`

- 如果要看正文结果应该怎么写：
  去 `outputs/model_stage5_paper_ready_v1/`
  重点看：
  - `empirical_results_outline_v1.txt`
  - `empirical_results_draft_v1.txt`

- 如果要看图表到底放正文还是附录：
  去 `outputs/model_stage5_paper_ready_v1/`
  重点看：
  - `paper_figures_list_v1.xlsx`
  - `paper_use_recommendation_v1.txt`

- 如果要快速了解这轮到底做了什么、结论是什么、下一步该怎么接：
  先看当前这份文件：
  - `TEAM_HANDOFF_STAGE5_v1.md`
