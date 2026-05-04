# 第六节结果摘要：机器学习预测与 SHAP 可解释分析

## 机器学习 baseline 表现

| 模型 | Accuracy | F1 | AUC |
| --- | --- | --- | --- |
| LogisticRegression | 0.7703 | 0.7954 | 0.8530 |
| XGBoost | 0.7979 | 0.8195 | 0.8836 |
| LightGBM | 0.7996 | 0.8210 | 0.8846 |

## GroupKFold 平均表现

| 模型 | 平均Accuracy | 平均F1 | 平均AUC |
| --- | --- | --- | --- |
| LightGBM | 0.7926 | 0.8158 | 0.8772 |
| XGBoost | 0.7933 | 0.8165 | 0.8768 |

## 为什么选择 LightGBM

- 在测试集上，LightGBM 的 AUC 为 0.8846，高于 XGBoost 的 0.8836 和 LogisticRegression 的 0.8530。
- 在 5 折 GroupKFold 中，LightGBM 的平均 AUC 为 0.8772，整体稳定性也略优。
- 因此，正文将 LightGBM 作为主 SHAP 模型是合理的。

## 三套 SHAP 方案表现

| 方案 | 特征数 | Accuracy | F1 | AUC |
| --- | --- | --- | --- | --- |
| main_clean | 43 | 0.7753 | 0.8026 | 0.8499 |
| no_totmet | 46 | 0.7808 | 0.8079 | 0.8564 |
| full_feature | 47 | 0.7996 | 0.8210 | 0.8846 |

## main_clean 前 20 个 SHAP 重要变量

| 排名 | 变量 | mean_abs_shap | 变量角色 |
| --- | --- | --- | --- |
| 1 | age | 0.531658 | demographic_control |
| 2 | urban | 0.460598 | demographic_control |
| 3 | female | 0.382767 | demographic_control |
| 4 | year_2020 | 0.341009 | year_control |
| 5 | iadl_limit | 0.261828 | health_constraints |
| 6 | chronic_count | 0.149347 | health_constraints |
| 7 | medical_burden_missing | 0.137948 | missing_indicator |
| 8 | log_intergen_support_out_w | 0.136200 | economic_pressure |
| 9 | married | 0.122190 | demographic_control |
| 10 | edu_1 | 0.110124 | demographic_control |
| 11 | poor_health | 0.106463 | health_constraints |
| 12 | arthre | 0.093971 | disease_indicator |
| 13 | log_hhcperc_v1_w | 0.085558 | economic_pressure |
| 14 | adl_limit | 0.079089 | health_constraints |
| 15 | poor_health_missing | 0.077363 | missing_indicator |
| 16 | log_intergen_support_in_w | 0.068840 | economic_pressure |
| 17 | total_cognition_missing | 0.066271 | missing_indicator |
| 18 | hearte | 0.065646 | disease_indicator |
| 19 | family_size | 0.058087 | family_care_and_intergenerational_support |
| 20 | hchild | 0.052169 | family_care_and_intergenerational_support |

## SHAP 结果的结构性解读

- 健康变量进入前列的包括：iadl_limit, chronic_count, poor_health, adl_limit。
- 家庭相关变量进入前列的包括：family_size, hchild。
- 经济相关变量进入前列的包括：log_intergen_support_out_w, log_hhcperc_v1_w, log_intergen_support_in_w。
- `family_care_index_v1` 本身没有进入 main_clean 前 20，说明家庭机制在 SHAP 主模型中更多通过家庭结构和代际支持类变量体现。

## 缺失指示变量的处理

- main_clean 前 20 中有 3 个缺失指示变量，分别是：medical_burden_missing(rank 7), poor_health_missing(rank 15), total_cognition_missing(rank 17)。
- 这些变量可以作为技术性特征保留在机器学习模型中，但正文解释时不应把它们当作 substantive 机制。

## 解释风险提示

- SHAP 只能解释“预测解释贡献”，不能解释为因果影响。
- `year_2020` 只作为年份控制变量，不应作为政策效果变量展开。
- `totmet_w` 在全特征 SHAP 中排名第 1（mean_abs_shap=0.865243），但其可能代理劳动活动强度本身，因此不进入正文主 SHAP 解释。
- `depression_high`、`total_cognition_w`、`family_care_index_v1` 仍需谨慎解释。

## 可直接写进论文的 SHAP 结论

- LightGBM 在预测表现和稳定性上均优于其余基线模型，因此被选为主解释模型。
- 在去除行为变量和 `totmet_w` 的 `main_clean` 方案中，年龄、城乡、性别、IADL 受限、慢性病负担、收入与代际支持等变量仍具有较高预测解释贡献，说明健康、人口学和经济支持结构共同塑造了劳动参与状态。
- 机器学习与 SHAP 结果总体支持主回归所揭示的基本机制，但这些结果应被理解为“变量对预测结果的解释贡献”，而非因果效应。

## 可直接写进论文的结论句

机器学习与 SHAP 分析表明，中老年劳动参与不仅受线性回归中可识别的健康、家庭与经济因素影响，而且在非线性预测框架下仍表现出明显的年龄、城乡、健康能力和支持结构差异；因此，可解释机器学习结果更适合作为主回归结论的补充验证，而非替代因果分析。
