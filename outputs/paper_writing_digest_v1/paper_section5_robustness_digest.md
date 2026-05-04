# 第五节补充摘要：稳健性检验

## R1-R8 各自检验内容

| 模型 | 检验内容 | 样本量 | 伪R² |
| --- | --- | --- | --- |
| R1_no_behavior | Remove behavior controls. | 29772 | 0.2081 |
| R2_no_family_index | Use family component items without family_care_index_v1. | 29772 | 0.2262 |
| R3_no_econ_index | Replace economic_pressure_index_v1 with component variables when available. | 29772 | 0.2275 |
| R4_no_cognition | Drop total_cognition_w. | 29772 | 0.2264 |
| R5_only_2018 | 2018 sample only. | 12336 | 0.1626 |
| R6_only_2020 | 2020 sample only. | 17436 | 0.2527 |
| R7_age_le_80 | Exclude age above 80. | 27893 | 0.2000 |
| R8_no_missing_flags | Drop missing-indicator variables if present. | 29772 | 0.2271 |

## 方向稳定性总结

- poor_health 在 R1-R8 中始终为负向；其中 2018 单年份样本 OR=0.7901，2020 单年份样本 OR=0.8265。
- iadl_limit 在全部稳健性模型中保持显著负向，OR 大致介于 0.4757 到 0.5398 之间。
- family_care_index_v1 在估计到该变量的模型中均为正向，2018 单年份样本 OR=1.0727，2020 单年份样本 OR=1.1867。
- economic_pressure_index_v1 在大多数规格中保持正向；仅 2018 单年份样本中不显著（p=0.1251），提示经济压力的推动作用在年份上存在一定波动。
- log_hhcperc_v1_w 与 log_medical_expense_w 在绝大多数规格中保持负向，其中医疗支出变量在 2020 单年份模型中因零方差未估计。

## 结果变化与需要说明的地方

- 去除行为变量（R1）后，健康约束和经济压力变量方向未变，主结论稳定。
- 去掉 family_care_index_v1、改用其组成项（R2）后，经济压力和健康变量结果仍稳定，说明主结论不依赖单一家庭压力指标。
- 去掉 economic_pressure_index_v1、改用其组成变量（R3）后，收入和医疗支出变量仍保持负向，说明经济机制并非完全依赖综合指数。
- 去掉 total_cognition_w（R4）后，其余健康变量方向基本不变，说明整体健康机制稳定。
- 单年份检验显示，2018 与 2020 的主结论方向一致，但个别变量显著性存在波动，尤其是 economic_pressure_index_v1 在 2018 年样本中不显著。
- 剔除 80 岁以上样本（R7）以及去除缺失指示变量（R8）后，核心结论没有发生方向性变化。

## Probit、LPM 与其他稳健性说明

- Probit 结果与 Logit 总体一致，可作为主要函数形式稳健性证据。
- 当前保留的最终结果文件中未包含 LPM 的独立结果表，因此正文不建议再声称“同时报告了 LPM 结果”，除非后续补充相关输出。
- 单年份、年龄截断、去除代理变量、去除缺失指示变量等检验已经足以支持主结论稳定。

## 可直接写进论文的稳健性结论

- 健康约束变量，尤其是 poor_health 和 iadl_limit，在不同样本、不同变量设定和不同模型规格下均保持显著负向，说明其对劳动参与的抑制作用最为稳健。
- family_care_index_v1 虽在多组稳健性模型中保持正向，但其解释应始终限定为家庭结构、同住安排和代际支持压力的综合代理，而非简单的照料促进效应。
- 经济压力变量总体保持正向，而收入和医疗支出变量保持负向，说明继续劳动中确实包含较强的压力驱动成分。

## 可直接写进论文的结论句

总体来看，多组稳健性检验并未改变主回归的核心方向：健康约束始终构成中老年继续劳动的硬边界，家庭与经济变量则更多体现为条件性和压力性机制，因此本文的主要结论具有较好的稳健性。
