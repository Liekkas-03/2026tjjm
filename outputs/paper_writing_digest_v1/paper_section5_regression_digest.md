# 第五节结果摘要：计量模型估计与机制分析

## Logit Model 1-5 的模型设置

- Model_1：基础控制模型：年龄、年龄平方、性别、婚姻、城乡、教育和年份控制变量。
- Model_2：在 Model 1 基础上加入健康约束变量：poor_health、chronic_count、adl_limit、iadl_limit、depression_high、total_cognition_w。
- Model_3：在 Model 2 基础上加入家庭照料与代际支持压力变量：family_care_index_v1、co_reside_child、care_elder_or_disabled。
- Model_4：在 Model 3 基础上加入经济压力变量：economic_pressure_index_v1、log_hhcperc_v1_w、log_medical_expense_w。medical_burden_w 在主回归数据中缺失，未进入该版主模型。
- Model_5：在 Model 4 基础上加入行为控制变量：smokev、drinkl、exercise，构成正文主回归模型。


## Model 5 核心变量结果

| 变量 | 系数 | OR | p值 |
| --- | --- | --- | --- |
| poor_health | -0.2056 | 0.8142 | 1.25e-07 |
| chronic_count | -0.0911 | 0.9129 | 4.28e-24 |
| adl_limit | -0.1418 | 0.8678 | 7.30e-04 |
| iadl_limit | -0.6682 | 0.5126 | 7.40e-61 |
| depression_high | 0.2868 | 1.3322 | 1.80e-14 |
| total_cognition_w | -0.0292 | 0.9712 | 5.19e-07 |
| family_care_index_v1 | 0.1208 | 1.1284 | 9.37e-09 |
| co_reside_child | -0.1496 | 0.8611 | 6.48e-04 |
| care_elder_or_disabled | -0.2382 | 0.7880 | 0.0066 |
| economic_pressure_index_v1 | 0.0739 | 1.0767 | 7.52e-05 |
| log_hhcperc_v1_w | -0.0924 | 0.9118 | 1.11e-05 |
| log_medical_expense_w | -0.0319 | 0.9686 | 2.85e-07 |
| age | 0.1436 | 1.1545 | 1.09e-10 |
| age_squared | -0.0017 | 0.9984 | 2.62e-22 |
| female | -0.7191 | 0.4872 | 4.90e-45 |
| married | 0.3487 | 1.4172 | 1.07e-13 |
| urban | -1.0696 | 0.3431 | 1.10e-192 |
| year_2020 | 0.7993 | 2.2239 | 1.19e-155 |

## Probit 与 Logit 的一致性

- Probit 主模型为 Model 4 口径，对应样本量为 29772，伪 R² 为 0.2082。
- 在健康、家庭和经济三类核心变量中，可直接比较的 12 个变量里，方向一致的有 12 个。
- 因此，Probit 与 Logit 的结果可判断为“总体一致”，主回归结论具有较好的函数形式稳健性。

## 机制分析总结

### 健康约束

- poor_health、chronic_count、adl_limit 和 iadl_limit 在 Model 5 中均为负向且显著，其中 iadl_limit 的抑制作用最强（OR=0.5126，p<0.001），说明健康能力是决定中老年人能否继续劳动的最稳定边界条件。
- depression_high 在主回归中为正向显著（OR=1.3322，p=1.80e-14），total_cognition_w 为负向显著（OR=0.9712，p=5.19e-07），这两个结果与通常直觉不完全一致，正文应明确标注“谨慎解释”。

### 家庭照料与代际支持压力

- family_care_index_v1 在主回归中为正向显著（OR=1.1284，p<0.001），但这一结果不能解释为照料责任促进劳动，而应理解为家庭结构、同住安排和代际支持压力的综合代理；与此同时，co_reside_child 和 care_elder_or_disabled 均为负向，说明直接家庭照护责任仍会压缩劳动参与。

### 经济压力

- economic_pressure_index_v1 为正向显著（OR=1.0767，p<0.001），而 log_hhcperc_v1_w 与 log_medical_expense_w 均为负向显著，表明一部分继续劳动更多体现为经济压力驱动，而非简单意义上的高适配性。

## AME 情况

- 现有最终结果文件未单独输出 AME 表，因此正文不宜报告具体 AME 数值。
- 如果需要保留“边际效应”表述，建议只保留概念性说明，不写具体估计值。

## 可直接写进论文的主回归结论

- 健康约束变量在主回归中表现出最稳定的负向作用，尤其是 IADL 受限、ADL 受限和慢性病负担，显著降低了中老年个体继续参与劳动的概率。
- 家庭变量呈现出“直接照护责任负向、综合家庭压力代理正向”的复杂格局，说明家庭照料与代际支持更多体现为结构性约束，而不能简单视为单一照料负担。
- 经济压力相关变量表明，继续劳动并不必然意味着更高的延迟退休适配性，其中一部分劳动参与更可能带有明显的收入与医疗负担压力驱动特征。

## 可直接写进论文的结论句

主回归结果表明，中老年劳动参与首先受健康能力约束，其次受到家庭照料与代际支持压力以及经济脆弱性的共同影响；因此，在延迟退休背景下，是否继续劳动不能被简单理解为“是否愿意延迟退休”，更应被理解为健康能力、家庭条件与经济压力共同作用下的结果。
