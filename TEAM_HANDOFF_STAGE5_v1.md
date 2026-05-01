# Team Handoff Stage 6

## 1. 这次完成了什么工作

这轮工作已经把项目从“中老年劳动参与的一般影响因素分析”进一步推进到了“延迟退休背景下的适配性识别与分层支持分析”。

前面 Stage 1-5 已经完成：

1. 数据清洗与变量整理
2. 最终建模数据构造与信息泄露审计
3. Logit / Probit / XGBoost / LightGBM baseline
4. SHAP 可解释分析
5. 异质性分析、论文主表主图整理、实证结果草稿

这次新增的 Stage 6 工作主要是：

1. 构造延迟退休相关年龄组：
   - `retirement_relevant_age`
   - `near_retirement_core`
   - `older_post_retirement`
   - `age_policy_group`
2. 构造延迟退休适配性分层：
   - `suitable_for_flexible_delay`
   - `pressure_driven_work`
   - `constrained_working`
   - `constrained_exit`
   - `potential_labor_supply`
   - `support_priority`
3. 对准退休年龄段做描述性分析
4. 在核心准退休年龄段上重跑 Logit
5. 在全样本中做“健康 / 家庭 / 经济压力 × 准退休年龄段”的交互项模型
6. 生成专门服务于“延迟退休背景”表述的图表、文字草稿和政策建议草稿

也就是说，现在项目主线已经可以从“谁在劳动、哪些变量重要”升级为：

> 在延迟退休背景下，哪些群体具备继续劳动能力，哪些群体属于压力驱动劳动，哪些群体需要弹性退出或保障支持。

## 2. 目前能得出什么结论

### 2.1 总体主线

本文不是延迟退休政策效果评估，也不模拟“退休年龄推到 65 岁会怎样”。  
本文更适合定位为：

> 延迟退休背景下的劳动参与适配性识别。

也就是说，我们关心的不是“政策导致了什么”，而是：

- 谁“能不能继续工作”
- 谁“有没有条件继续工作”
- 谁“是在主动延迟，还是被经济压力推着劳动”
- 谁“更需要弹性退出和保障支持”

### 2.2 Stage 1-5 已经比较稳的结论

- `poor_health`、`adl_limit`、`iadl_limit` 稳定负向，说明健康受限会明显压低劳动参与。
- `chronic_count` 整体偏负向，说明慢性病负担越重，继续劳动的倾向越低。
- `co_reside_child`、`care_elder_or_disabled` 多数情况下负向，说明家庭照料责任会挤压劳动参与。
- `economic_pressure_index_v1` 正向，说明经济压力越大，继续劳动的倾向越强。
- `log_medical_expense_w` 负向，说明医疗负担与退出劳动更相关。

### 2.3 Stage 6 强化后的延迟退休相关结论

- 准退休年龄段样本量为 `9885`，核心准退休年龄段样本量为 `6794`。
- 准退休年龄段劳动参与率约为 `0.6718`。
- 在准退休年龄段内部，人数最多的类型不是“适配继续劳动型”，而是：
  - `constrained_working` 带约束继续劳动型：`4098`
  - `constrained_exit` 健康/家庭约束退出型：`2163`
- `suitable_for_flexible_delay` 适配弹性延迟型约 `1571`
- `potential_labor_supply` 潜在劳动供给型约 `821`
- `support_priority` 保障支持重点型约 `1016`

这说明：

1. 延迟退休背景下，真正“适配继续劳动”的人有，但不是全部。
2. 很多仍在劳动的人，其实是“带约束继续劳动”，不应简单视为适合统一延迟退休。
3. 一部分退出劳动的人并非完全失去劳动能力，而是可能属于“潜在劳动供给型”，如果有更好的健康支持、家庭照护减压和灵活岗位，仍可能转化为有效劳动力供给。
4. 还有一部分人属于“保障支持重点型”，这类群体更需要基本保障，而不是单纯强调继续劳动。

### 2.4 核心准退休年龄段回归结果

在 `near_retirement_core` 样本上，Logit 结果继续支持“健康 + 家庭 + 经济”的三条主线：

- `poor_health` 负向且显著
- `chronic_count` 负向且显著
- `adl_limit`、`iadl_limit` 负向且显著
- `economic_pressure_index_v1` 正向且显著
- `log_medical_expense_w` 负向且显著

这说明在真正接近退休决策边界的人群中，健康能力和经济压力依然是最核心的因素。

### 2.5 交互项模型给出的补充信息

交互项模型重点检验了“这些约束在准退休年龄段是否更明显”。

目前比较有信息量的结果是：

- `poor_health × retirement_relevant_age` 为负且显著
- `iadl_limit × retirement_relevant_age` 为负且显著
- `adl_limit × retirement_relevant_age` 负向，边际显著
- `economic_pressure_index_v1 × retirement_relevant_age` 为负且显著
- `family_care_index_v1 × retirement_relevant_age` 不显著

这说明：

1. 健康约束在准退休年龄段的抑制作用更强。
2. 经济压力虽然总体推动劳动参与，但在准退休年龄段，这种推动并不意味着“适配性更高”，反而更像一种脆弱性信号。
3. `family_care_index_v1` 更像“家庭照料与代际支持压力代理”，不宜解释为单一、线性的照料负担指标。

## 3. 需要谨慎解释的地方

- `family_care_index_v1` 不能直接写成“照料责任促进劳动参与”。
  更合适的写法是：
  “家庭照料与代际支持压力代理”。
- `economic_pressure_index_v1` 正向，不能直接写成“更适合延迟退休”。
  更合理的解释是：
  “经济压力推动继续劳动倾向，其中一部分可能属于被动劳动。”
- `year_2020` 仍然只是年份控制变量，不能解释为政策效果。
- `depression_high` 为正、`total_cognition_w` 为负，这两个方向仍需提醒可能存在编码、样本选择或代理效应。
- 所有 `*_missing` 变量都只作为技术性特征使用，不能作 substantive 机制解释。
- SHAP 只能解释“预测贡献”，不能写成因果关系。

## 4. 这次提交给组员的文件应该怎么理解

如果后面要提交给组员，建议把 Stage 6 也纳入“最终论文包”。核心逻辑是：

- 脚本负责复现
- audited 数据负责统一建模口径
- Stage 5 负责论文主表主图和基础写作
- Stage 6 负责把论文主线明确推向“延迟退休适配性识别”

### 4.1 复现脚本

- `data/scripts/07_finalize_model_data_v1.py`
- `data/scripts/08_audit_final_model_data_v1.py`
- `data/scripts/09_train_baseline_models_v1.py`
- `data/scripts/10_stage3_review_robustness_v1.py`
- `data/scripts/11_stage4_shap_analysis_v1.py`
- `data/scripts/12_stage5_heterogeneity_and_paper_ready_v1.py`
- `data/scripts/13_stage6_retirement_relevance_v1.py`

### 4.2 最终 audited 数据

- `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv`
- `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv`
- `outputs/final_model_data_v1/final_leakage_audit_v1/final_leakage_audit_report_v1.txt`

### 4.3 Stage 5 论文主表主图与写作草稿

重点还是：

- `outputs/model_stage5_paper_ready_v1/paper_table1_descriptive_stats_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table2_group_labor_rates_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table3_logit_main_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table4_robustness_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table5_ml_performance_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table6_shap_top_features_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/paper_table7_heterogeneity_final.xlsx`
- `outputs/model_stage5_paper_ready_v1/empirical_results_outline_v1.txt`
- `outputs/model_stage5_paper_ready_v1/empirical_results_draft_v1.txt`

### 4.4 Stage 6 延迟退休相关结果包

这是这轮新增、最直接服务于最终题目的内容：

- `outputs/model_stage6_retirement_relevance_v1/retirement_suitability_distribution_v1.xlsx`
- `outputs/model_stage6_retirement_relevance_v1/near_retirement_descriptive_tables_v1.xlsx`
- `outputs/model_stage6_retirement_relevance_v1/near_retirement_logit_results_v1.xlsx`
- `outputs/model_stage6_retirement_relevance_v1/retirement_interaction_logit_v1.xlsx`
- `outputs/model_stage6_retirement_relevance_v1/fig_near_retirement_labor_rates.png`
- `outputs/model_stage6_retirement_relevance_v1/fig_retirement_suitability_distribution.png`
- `outputs/model_stage6_retirement_relevance_v1/fig_health_constraints_near_retirement.png`
- `outputs/model_stage6_retirement_relevance_v1/fig_economic_pressure_near_retirement.png`
- `outputs/model_stage6_retirement_relevance_v1/retirement_relevance_writeup_v1.txt`
- `outputs/model_stage6_retirement_relevance_v1/retirement_policy_implications_v1.txt`

## 5. 组员下一步应该怎么写论文

### 5.1 论文主线建议

现在建议把论文主线明确改成：

> 延迟退休背景下的劳动参与适配性识别与分层支持

而不是停留在：

> 中老年劳动参与影响因素分析

### 5.2 结果结构建议

正文结构可以这样排：

1. 描述性统计与劳动参与现状
2. 主回归结果
3. 稳健性检验
4. 机器学习与 SHAP 结果
5. 异质性分析
6. 延迟退休相关性强化分析
7. 小结与政策含义

### 5.3 延迟退休相关部分应该怎么写

重点不是写：

- 政策已经让谁去工作
- 政策造成了什么因果变化

重点应该写：

- 哪些群体具备继续劳动能力
- 哪些群体即使继续劳动，也属于带约束劳动或压力驱动劳动
- 哪些群体更适合弹性退出
- 哪些群体最需要保障支持

### 5.4 政策建议怎么从结果推出

- 对 `suitable_for_flexible_delay`：
  更适合提供灵活岗位、再就业服务、技能培训
- 对 `pressure_driven_work`：
  更应完善养老金、低收入补贴、医疗保障，避免被迫劳动
- 对 `constrained_exit`：
  应允许弹性退出，加强健康管理与照护支持
- 对 `constrained_working`：
  应加强劳动保护、工时弹性、职业健康支持
- 对 `support_priority`：
  应强化长期护理、社区照护和基本生活保障

## 6. 这份交接包的定位

现在这批文件的定位已经不是“模型跑通了”，而是：

> 一套足够让其他组员直接接手，把论文主线明确收束到“延迟退休适配性识别”，并继续往终稿推进的交接包。

换句话说，后面的重点已经不是重新清洗或重新搜模型，而是：

1. 把 Stage 5 和 Stage 6 的表图整合进同一版论文结构
2. 统一措辞
3. 控制解释风险
4. 完成全文整合和终稿排版

## 7. 组员如果要找什么内容，应该去哪里看

- 如果要找最终建模数据：
  去 `outputs/final_model_data_v1/`
  重点看：
  - `CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv`
  - `CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv`

- 如果要确认最终数据能不能放心建模：
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
  - `13_stage6_retirement_relevance_v1.py`

- 如果要直接写论文主表：
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

- 如果要写“延迟退休相关性强化分析”这一节：
  去 `outputs/model_stage6_retirement_relevance_v1/`
  重点看：
  - `retirement_suitability_distribution_v1.xlsx`
  - `near_retirement_descriptive_tables_v1.xlsx`
  - `near_retirement_logit_results_v1.xlsx`
  - `retirement_interaction_logit_v1.xlsx`
  - `retirement_relevance_writeup_v1.txt`
  - `retirement_policy_implications_v1.txt`

- 如果要直接拿图放到“延迟退休背景”这一节：
  去 `outputs/model_stage6_retirement_relevance_v1/`
  重点看：
  - `fig_near_retirement_labor_rates.png`
  - `fig_retirement_suitability_distribution.png`
  - `fig_health_constraints_near_retirement.png`
  - `fig_economic_pressure_near_retirement.png`

- 如果要看正文结果应该怎么写：
  去 `outputs/model_stage5_paper_ready_v1/`
  重点看：
  - `empirical_results_outline_v1.txt`
  - `empirical_results_draft_v1.txt`

- 如果要快速了解这轮到底做了什么、现在结论是什么、下一步该怎么接：
  先看当前这份文件：
  - `TEAM_HANDOFF_STAGE5_v1.md`
