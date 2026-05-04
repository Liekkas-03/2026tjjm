# 论文表图使用清单

## 正文建议保留的表格

- Table 1：描述性统计表，放在第4节。
- Table 2：分组劳动参与率，放在第4节。
- Table 3：Logit 主回归结果，放在第5节。
- Table 4：稳健性检验结果，放在第5节或第6节前半部分。
- Table 5：机器学习模型性能比较，放在第6节。
- Table 6：SHAP 重要特征表，放在第6节。
- Table 7：异质性结果表，放在第7节。
- 建议在第7节追加一张“准退休年龄段适配性分层表”或直接引用 retirement_suitability 分布表。

## 正文建议保留的图形

- Figure 1：劳动参与率分组柱状图，放在第4节。
- Figure 2：Logit 主回归森林图，放在第5节。
- Figure 3：SHAP bar summary（main_clean），放在第6节。
- Figure 4：SHAP beeswarm（main_clean），放在第6节。
- Figure 5：准退休年龄段适配性分布图，放在第7节。

## 附录建议保留的图形

- SHAP bar summary（no_totmet）。
- SHAP bar summary（full_feature）。
- 其余 dependence plots。
- 详细稳健性附加图表。

## 现有 figure list 中的正文推荐图

- Figure 1：劳动参与率分组柱状图，建议位置：正文。用途：展示年龄组、性别与城乡维度的劳动参与率差异
- Figure 2：Logit Model 5 Odds Ratio Forest Plot，建议位置：正文。用途：可视化主回归结果方向与区间估计
- Figure 3：SHAP Bar Summary (main_clean)，建议位置：正文。用途：展示主 SHAP 模型的全局解释贡献排序
- Figure 4：SHAP Beeswarm (main_clean)，建议位置：正文。用途：展示主 SHAP 模型变量影响方向与样本分布
- Figure 5：Dependence Plot - poor_health，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
- Figure 6：Dependence Plot - iadl_limit，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
- Figure 7：Dependence Plot - family_care_index_v1，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
- Figure 8：Dependence Plot - economic_pressure_index_v1，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
- Figure 9：Dependence Plot - log_hhcperc_v1_w，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
- Figure 10：Dependence Plot - log_intergen_support_out_w，建议位置：正文。用途：展示关键健康、家庭或经济变量的 SHAP 局部关系
