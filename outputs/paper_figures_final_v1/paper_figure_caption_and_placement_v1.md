# 论文图表使用说明

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
