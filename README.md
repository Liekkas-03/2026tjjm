# 项目说明

## 当前阶段

本项目目前已经推进到 **Stage 6：延迟退休相关性强化分析**。

前面已经完成：

1. 数据清洗与变量整理
2. 最终建模数据构造与信息泄露审计
3. Logit / Probit / XGBoost / LightGBM baseline
4. SHAP 可解释分析
5. 异质性分析、论文主表主图整理、实证结果草稿
6. 延迟退休背景下的准退休年龄段、适配性分层与政策相关性强化分析

## 论文主线

当前最推荐的论文主线是：

> 延迟退休背景下中老年劳动参与适配性识别与分层支持研究

重点不是做“延迟退休政策效果评估”，而是回答：

- 哪些群体具备继续劳动能力
- 哪些群体具备继续劳动条件
- 哪些群体属于压力驱动劳动
- 哪些群体更需要弹性退出和保障支持

## 核心数据与结果位置

- 最终 audited 数据：
  - `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv`
  - `outputs/final_model_data_v1/CHARLS_labor_panel_2018_2020_v4_ml_final_audited.csv`

- Stage 5 论文表图与草稿：
  - `outputs/model_stage5_paper_ready_v1/`

- Stage 6 延迟退休相关结果：
  - `outputs/model_stage6_retirement_relevance_v1/`

## 论文前半部分重写建议

如果要修改别的组员写的论文前半部分，请直接看：

- `outputs/model_stage6_retirement_relevance_v1/introduction_rewrite_replacements_v1.md`

这份文件已经按“哪一段替换成哪一段”的形式整理好了，内容与当前代码、数据和结果保持一致。

## 关键提醒

- `family_care_index_v1` 应表述为“家庭照料与代际支持压力代理”
- `economic_pressure_index_v1` 的正向结果不能直接解释为“更适合延迟退休”
- `year_2020` 只作为年份控制变量
- SHAP 只能解释预测贡献，不能作为因果证据
