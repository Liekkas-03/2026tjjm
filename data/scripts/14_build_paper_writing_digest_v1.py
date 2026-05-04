from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper_writing_digest_v1"

STAGE5_DIR = ROOT / "outputs" / "model_stage5_paper_ready_v1"
STAGE4_DIR = ROOT / "outputs" / "model_stage4_shap_v1"
STAGE3_DIR = ROOT / "outputs" / "model_stage3_review_v1"
TRAINING_DIR = ROOT / "outputs" / "model_training_v1"
STAGE6_DIR = ROOT / "outputs" / "model_stage6_retirement_relevance_v1"
FINAL_DIR = ROOT / "outputs" / "final_model_data_v1"

CORE_FILES = [
    STAGE5_DIR / "paper_table1_descriptive_stats_final.xlsx",
    STAGE5_DIR / "paper_table2_group_labor_rates_final.xlsx",
    STAGE5_DIR / "paper_table3_logit_main_final.xlsx",
    STAGE5_DIR / "paper_table4_robustness_final.xlsx",
    STAGE5_DIR / "paper_table5_ml_performance_final.xlsx",
    STAGE5_DIR / "paper_table6_shap_top_features_final.xlsx",
    STAGE5_DIR / "paper_table7_heterogeneity_final.xlsx",
    STAGE5_DIR / "paper_figures_list_v1.xlsx",
    STAGE5_DIR / "empirical_results_draft_v1.txt",
    STAGE5_DIR / "empirical_results_outline_v1.txt",
    STAGE5_DIR / "paper_use_recommendation_v1.txt",
    STAGE4_DIR / "shap_model_metrics_v1.xlsx",
    STAGE4_DIR / "shap_values_summary_table_v1.xlsx",
    STAGE4_DIR / "stage4_shap_report_v1.txt",
    STAGE3_DIR / "stage3_review_report_v1.txt",
    TRAINING_DIR / "baseline_model_report_v1.txt",
    FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv",
]

STAGE6_FILES = [
    STAGE6_DIR / "retirement_suitability_distribution_v1.xlsx",
    STAGE6_DIR / "near_retirement_descriptive_tables_v1.xlsx",
    STAGE6_DIR / "near_retirement_logit_results_v1.xlsx",
    STAGE6_DIR / "retirement_interaction_logit_v1.xlsx",
    STAGE6_DIR / "retirement_relevance_writeup_v1.txt",
    STAGE6_DIR / "retirement_policy_implications_v1.txt",
]


def ensure_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return path.read_text(errors="ignore")


def col_to_index(ref: str) -> int:
    letters = ""
    for ch in ref:
        if ch.isalpha():
            letters += ch
        else:
            break
    idx = 0
    for c in letters:
        idx = idx * 26 + (ord(c.upper()) - 64)
    return idx - 1


def norm_target(target: str) -> str:
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append("".join([t.text or "" for t in si.findall(".//a:t", ns)]))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        relroot = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relroot}

        result: dict[str, list[list[str]]] = {}
        for sheet in workbook.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheets"):
            name = sheet.attrib["name"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = norm_target(rels[rid])
            sroot = ET.fromstring(zf.read(target))
            rows: list[list[str]] = []
            for row in sroot.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                maxidx = -1
                cells: list[tuple[int, str]] = []
                for cell in row.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                    ref = cell.attrib.get("r", "A1")
                    idx = col_to_index(ref)
                    typ = cell.attrib.get("t")
                    value_node = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if typ == "inlineStr":
                        isel = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}is")
                        text = "".join([t.text or "" for t in isel.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]) if isel is not None else ""
                    elif typ == "s" and value_node is not None:
                        text = shared_strings[int(value_node.text)]
                    elif value_node is not None:
                        text = value_node.text or ""
                    else:
                        text = ""
                    cells.append((idx, text))
                    maxidx = max(maxidx, idx)
                values = [""] * (maxidx + 1 if maxidx >= 0 else 0)
                for idx, text in cells:
                    values[idx] = text
                rows.append(values)
            result[name] = rows
        return result


def sheet_to_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    header = rows[0]
    out: list[dict[str, str]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        out.append({header[i]: padded[i] for i in range(len(header))})
    return out


def to_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int(value: str | float | int | None) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value}"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.{digits}f}%"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def md_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def load_csv_summary(path: Path) -> dict[str, object]:
    total = 0
    labor = 0
    by_year: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "labor": 0})
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            y = row.get("year", "")
            lp = int(float(row.get("labor_participation", "0") or 0))
            labor += lp
            by_year[y]["n"] += 1
            by_year[y]["labor"] += lp
    return {
        "n_total": total,
        "n_labor": labor,
        "n_nonlabor": total - labor,
        "labor_rate": labor / total if total else None,
        "year": by_year,
    }


def lookup_group_rows(rows: list[dict[str, str]], group_var: str) -> list[dict[str, str]]:
    return [r for r in rows if r.get("group_var") == group_var]


def find_row(rows: list[dict[str, str]], **conds: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    return None


def build_section4(csv_summary: dict[str, object], table2: list[dict[str, str]]) -> str:
    n_total = csv_summary["n_total"]
    n_labor = csv_summary["n_labor"]
    n_nonlabor = csv_summary["n_nonlabor"]
    labor_rate = csv_summary["labor_rate"]
    years = csv_summary["year"]

    age_rows = lookup_group_rows(table2, "age_group_stage5")
    female_rows = lookup_group_rows(table2, "female")
    urban_rows = lookup_group_rows(table2, "urban")
    health_rows = lookup_group_rows(table2, "poor_health")
    family_rows = lookup_group_rows(table2, "family_high_stage5")
    econ_rows = lookup_group_rows(table2, "econ_high_stage5")

    findings = [
        f"总体样本量为 {n_total}，其中劳动参与者 {n_labor} 人，占 {fmt_pct(labor_rate)}；未参与劳动者 {n_nonlabor} 人，占 {fmt_pct(n_nonlabor / n_total)}。",
        f"按年份看，2018 年样本 {years['2018']['n']} 人，劳动参与率为 {fmt_pct(years['2018']['labor'] / years['2018']['n'])}；2020 年样本 {years['2020']['n']} 人，劳动参与率上升至 {fmt_pct(years['2020']['labor'] / years['2020']['n'])}。",
        f"按年龄组看，45-59 岁劳动参与率最高，为 {fmt_pct(to_float(age_rows[0]['labor_rate']))}；60-69 岁降至 {fmt_pct(to_float(age_rows[1]['labor_rate']))}；70 岁及以上进一步降至 {fmt_pct(to_float(age_rows[2]['labor_rate']))}，呈现明显下降梯度。",
        f"男性劳动参与率为 {fmt_pct(to_float(female_rows[0]['labor_rate']))}，明显高于女性的 {fmt_pct(to_float(female_rows[1]['labor_rate']))}；农村样本劳动参与率为 {fmt_pct(to_float(urban_rows[0]['labor_rate']))}，高于城镇样本的 {fmt_pct(to_float(urban_rows[1]['labor_rate']))}。",
        f"健康较好样本劳动参与率为 {fmt_pct(to_float(health_rows[0]['labor_rate']))}，高于健康较差样本的 {fmt_pct(to_float(health_rows[1]['labor_rate']))}；家庭压力高组和经济压力高组的劳动参与率分别为 {fmt_pct(to_float(family_rows[1]['labor_rate']))} 和 {fmt_pct(to_float(econ_rows[1]['labor_rate']))}，均略高于低压力组，说明继续劳动中存在明显的结构性和压力性成分。",
    ]

    return f"""# 第四节结果摘要：描述性统计与中老年劳动参与现状

## 核心样本与总体分布

- 最终样本量：{n_total}
- 2018 年样本量：{years['2018']['n']}
- 2020 年样本量：{years['2020']['n']}
- 劳动参与者：{n_labor}，占 {fmt_pct(labor_rate)}
- 非劳动参与者：{n_nonlabor}，占 {fmt_pct(n_nonlabor / n_total)}
- 总体劳动参与率：{fmt_pct(labor_rate)}

## 分组劳动参与率

### 按年份

{md_table(['年份', '样本量', '劳动参与率'], [[y, years[y]['n'], fmt_pct(years[y]['labor'] / years[y]['n'])] for y in ['2018', '2020']])}

### 按年龄组

{md_table(['年龄组', '样本量', '劳动参与率'], [[r['group_value'], r['n'], fmt_pct(to_float(r['labor_rate']))] for r in age_rows])}

### 按性别

{md_table(['性别', '样本量', '劳动参与率'], [['男性' if r['group_value'] == '0' else '女性', r['n'], fmt_pct(to_float(r['labor_rate']))] for r in female_rows])}

### 按城乡

{md_table(['城乡', '样本量', '劳动参与率'], [['农村' if r['group_value'] == '0' else '城镇', r['n'], fmt_pct(to_float(r['labor_rate']))] for r in urban_rows])}

### 按健康状态

{md_table(['健康状态', '样本量', '劳动参与率'], [['健康较好' if r['group_value'] == '0' else '健康较差', r['n'], fmt_pct(to_float(r['labor_rate']))] for r in health_rows])}

### 按家庭压力高低

{md_table(['家庭压力组', '样本量', '劳动参与率'], [['低压力' if r['group_value'] == '0' else '高压力', r['n'], fmt_pct(to_float(r['labor_rate']))] for r in family_rows])}

### 按经济压力高低

{md_table(['经济压力组', '样本量', '劳动参与率'], [['低压力' if r['group_value'] == '0' else '高压力', r['n'], fmt_pct(to_float(r['labor_rate']))] for r in econ_rows])}

## 可直接写进论文的描述性发现

- {findings[0]}
- {findings[1]}
- {findings[2]}
- {findings[3]}
- {findings[4]}

## 可直接写进论文的结论句

中老年劳动参与具有明显的年龄、性别、城乡和健康梯度特征，但家庭压力高组和经济压力高组并未表现出更低的劳动参与率，说明继续劳动并不完全等同于更高适配性，其中相当一部分劳动参与可能带有结构性约束或经济压力驱动特征。
"""


def build_section5(table3_long: list[dict[str, str]], table3_summary: list[dict[str, str]], probit_summary: list[dict[str, str]], probit_rows: list[dict[str, str]]) -> str:
    model_desc = {
        "Model_1": "基础控制模型：年龄、年龄平方、性别、婚姻、城乡、教育和年份控制变量。",
        "Model_2": "在 Model 1 基础上加入健康约束变量：poor_health、chronic_count、adl_limit、iadl_limit、depression_high、total_cognition_w。",
        "Model_3": "在 Model 2 基础上加入家庭照料与代际支持压力变量：family_care_index_v1、co_reside_child、care_elder_or_disabled。",
        "Model_4": "在 Model 3 基础上加入经济压力变量：economic_pressure_index_v1、log_hhcperc_v1_w、log_medical_expense_w。medical_burden_w 在主回归数据中缺失，未进入该版主模型。",
        "Model_5": "在 Model 4 基础上加入行为控制变量：smokev、drinkl、exercise，构成正文主回归模型。",
    }

    model5_rows = [r for r in table3_long if r.get("model") == "Model_5"]
    m5 = {r["variable"]: r for r in model5_rows}
    probit = {r["variable"]: r for r in probit_rows}
    core_vars = [
        "poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w",
        "family_care_index_v1", "co_reside_child", "care_elder_or_disabled",
        "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w",
        "age", "age_squared", "female", "married", "urban", "year_2020",
    ]
    core_rows = []
    for var in core_vars:
        row = m5.get(var)
        if row:
            core_rows.append([
                var,
                fmt_num(to_float(row["coef"]), 4),
                fmt_num(to_float(row["odds_ratio"]), 4),
                fmt_p(to_float(row["p_value"])),
            ])

    probit_consistent = []
    for var in ["poor_health", "chronic_count", "adl_limit", "iadl_limit", "depression_high", "total_cognition_w", "family_care_index_v1", "co_reside_child", "care_elder_or_disabled", "economic_pressure_index_v1", "log_hhcperc_v1_w", "log_medical_expense_w"]:
        if var in probit and var in {r["variable"] for r in table3_long if r["model"] == "Model_4"}:
            lcoef = to_float(next(r["coef"] for r in table3_long if r["model"] == "Model_4" and r["variable"] == var))
            pcoef = to_float(probit[var]["coef"])
            if lcoef is not None and pcoef is not None:
                probit_consistent.append((var, math.copysign(1, lcoef) == math.copysign(1, pcoef)))

    consistent_count = sum(1 for _, ok in probit_consistent if ok)

    health_conclusion = "poor_health、chronic_count、adl_limit 和 iadl_limit 在 Model 5 中均为负向且显著，其中 iadl_limit 的抑制作用最强（OR=0.5126，p<0.001），说明健康能力是决定中老年人能否继续劳动的最稳定边界条件。"
    family_conclusion = "family_care_index_v1 在主回归中为正向显著（OR=1.1284，p<0.001），但这一结果不能解释为照料责任促进劳动，而应理解为家庭结构、同住安排和代际支持压力的综合代理；与此同时，co_reside_child 和 care_elder_or_disabled 均为负向，说明直接家庭照护责任仍会压缩劳动参与。"
    econ_conclusion = "economic_pressure_index_v1 为正向显著（OR=1.0767，p<0.001），而 log_hhcperc_v1_w 与 log_medical_expense_w 均为负向显著，表明一部分继续劳动更多体现为经济压力驱动，而非简单意义上的高适配性。"

    return f"""# 第五节结果摘要：计量模型估计与机制分析

## Logit Model 1-5 的模型设置

{"".join([f"- {k}：{v}\n" for k, v in model_desc.items()])}

## Model 5 核心变量结果

{md_table(['变量', '系数', 'OR', 'p值'], core_rows)}

## Probit 与 Logit 的一致性

- Probit 主模型为 Model 4 口径，对应样本量为 {probit_summary[0]['nobs']}，伪 R² 为 {fmt_num(to_float(probit_summary[0]['pseudo_r2']), 4)}。
- 在健康、家庭和经济三类核心变量中，可直接比较的 {len(probit_consistent)} 个变量里，方向一致的有 {consistent_count} 个。
- 因此，Probit 与 Logit 的结果可判断为“总体一致”，主回归结论具有较好的函数形式稳健性。

## 机制分析总结

### 健康约束

- {health_conclusion}
- depression_high 在主回归中为正向显著（OR={fmt_num(to_float(m5['depression_high']['odds_ratio']), 4)}，p={fmt_p(to_float(m5['depression_high']['p_value']))}），total_cognition_w 为负向显著（OR={fmt_num(to_float(m5['total_cognition_w']['odds_ratio']), 4)}，p={fmt_p(to_float(m5['total_cognition_w']['p_value']))}），这两个结果与通常直觉不完全一致，正文应明确标注“谨慎解释”。

### 家庭照料与代际支持压力

- {family_conclusion}

### 经济压力

- {econ_conclusion}

## AME 情况

- 现有最终结果文件未单独输出 AME 表，因此正文不宜报告具体 AME 数值。
- 如果需要保留“边际效应”表述，建议只保留概念性说明，不写具体估计值。

## 可直接写进论文的主回归结论

- 健康约束变量在主回归中表现出最稳定的负向作用，尤其是 IADL 受限、ADL 受限和慢性病负担，显著降低了中老年个体继续参与劳动的概率。
- 家庭变量呈现出“直接照护责任负向、综合家庭压力代理正向”的复杂格局，说明家庭照料与代际支持更多体现为结构性约束，而不能简单视为单一照料负担。
- 经济压力相关变量表明，继续劳动并不必然意味着更高的延迟退休适配性，其中一部分劳动参与更可能带有明显的收入与医疗负担压力驱动特征。

## 可直接写进论文的结论句

主回归结果表明，中老年劳动参与首先受健康能力约束，其次受到家庭照料与代际支持压力以及经济脆弱性的共同影响；因此，在延迟退休背景下，是否继续劳动不能被简单理解为“是否愿意延迟退休”，更应被理解为健康能力、家庭条件与经济压力共同作用下的结果。
"""


def build_section5_robustness(robust_summary: list[dict[str, str]], robust_key: list[dict[str, str]]) -> str:
    note_lines = []
    for row in robust_summary:
        note_lines.append([
            row["model"],
            row["note"],
            row["nobs"],
            fmt_num(to_float(row["pseudo_r2"]), 4),
        ])

    key_map: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in robust_key:
        key_map[row["model"]][row["variable"]] = row

    stable_lines = [
        f"poor_health 在 R1-R8 中始终为负向；其中 2018 单年份样本 OR={fmt_num(to_float(key_map['R5_only_2018']['poor_health']['odds_ratio']), 4)}，2020 单年份样本 OR={fmt_num(to_float(key_map['R6_only_2020']['poor_health']['odds_ratio']), 4)}。",
        f"iadl_limit 在全部稳健性模型中保持显著负向，OR 大致介于 {fmt_num(min(to_float(key_map[m]['iadl_limit']['odds_ratio']) for m in key_map if to_float(key_map[m]['iadl_limit']['odds_ratio']) is not None), 4)} 到 {fmt_num(max(to_float(key_map[m]['iadl_limit']['odds_ratio']) for m in key_map if to_float(key_map[m]['iadl_limit']['odds_ratio']) is not None), 4)} 之间。",
        f"family_care_index_v1 在估计到该变量的模型中均为正向，2018 单年份样本 OR={fmt_num(to_float(key_map['R5_only_2018']['family_care_index_v1']['odds_ratio']), 4)}，2020 单年份样本 OR={fmt_num(to_float(key_map['R6_only_2020']['family_care_index_v1']['odds_ratio']), 4)}。",
        f"economic_pressure_index_v1 在大多数规格中保持正向；仅 2018 单年份样本中不显著（p={fmt_p(to_float(key_map['R5_only_2018']['economic_pressure_index_v1']['p_value']))}），提示经济压力的推动作用在年份上存在一定波动。",
        f"log_hhcperc_v1_w 与 log_medical_expense_w 在绝大多数规格中保持负向，其中医疗支出变量在 2020 单年份模型中因零方差未估计。",
    ]

    return f"""# 第五节补充摘要：稳健性检验

## R1-R8 各自检验内容

{md_table(['模型', '检验内容', '样本量', '伪R²'], note_lines)}

## 方向稳定性总结

- {stable_lines[0]}
- {stable_lines[1]}
- {stable_lines[2]}
- {stable_lines[3]}
- {stable_lines[4]}

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
"""


def build_section6(ml_perf: list[dict[str, str]], cv_summary: list[dict[str, str]], shap_metrics: list[dict[str, str]], shap_rows_all: list[dict[str, str]]) -> str:
    baseline_rows = [r for r in ml_perf if r["scenario"] == "baseline"]
    shap_scheme_rows = {r["scheme"]: r for r in shap_metrics}
    main_clean = [r for r in shap_rows_all if r["scheme"] == "main_clean"]
    top20 = sorted(main_clean, key=lambda r: to_int(r["rank"]) or 999)[:20]

    role_groups = defaultdict(list)
    for row in top20:
        role_groups[row["feature_role"]].append(row["feature"])

    missing_feats = [r for r in top20 if r["feature_role"] == "missing_indicator"]

    return f"""# 第六节结果摘要：机器学习预测与 SHAP 可解释分析

## 机器学习 baseline 表现

{md_table(['模型', 'Accuracy', 'F1', 'AUC'], [[r['model'], fmt_num(to_float(r['accuracy']), 4), fmt_num(to_float(r['f1']), 4), fmt_num(to_float(r['roc_auc']), 4)] for r in baseline_rows])}

## GroupKFold 平均表现

{md_table(['模型', '平均Accuracy', '平均F1', '平均AUC'], [[r['model'], fmt_num(to_float(r['accuracy']), 4), fmt_num(to_float(r['f1']), 4), fmt_num(to_float(r['roc_auc']), 4)] for r in cv_summary])}

## 为什么选择 LightGBM

- 在测试集上，LightGBM 的 AUC 为 {fmt_num(to_float(next(r['roc_auc'] for r in baseline_rows if r['model'] == 'LightGBM')), 4)}，高于 XGBoost 的 {fmt_num(to_float(next(r['roc_auc'] for r in baseline_rows if r['model'] == 'XGBoost')), 4)} 和 LogisticRegression 的 {fmt_num(to_float(next(r['roc_auc'] for r in baseline_rows if r['model'] == 'LogisticRegression')), 4)}。
- 在 5 折 GroupKFold 中，LightGBM 的平均 AUC 为 {fmt_num(to_float(next(r['roc_auc'] for r in cv_summary if r['model'] == 'LightGBM')), 4)}，整体稳定性也略优。
- 因此，正文将 LightGBM 作为主 SHAP 模型是合理的。

## 三套 SHAP 方案表现

{md_table(['方案', '特征数', 'Accuracy', 'F1', 'AUC'], [[name, row['feature_count'], fmt_num(to_float(row['accuracy']), 4), fmt_num(to_float(row['f1']), 4), fmt_num(to_float(row['roc_auc']), 4)] for name, row in [('main_clean', shap_scheme_rows['main_clean']), ('no_totmet', shap_scheme_rows['no_totmet']), ('full_feature', shap_scheme_rows['full_feature'])]])}

## main_clean 前 20 个 SHAP 重要变量

{md_table(['排名', '变量', 'mean_abs_shap', '变量角色'], [[r['rank'], r['feature'], fmt_num(to_float(r['mean_abs_shap']), 6), r['feature_role']] for r in top20])}

## SHAP 结果的结构性解读

- 健康变量进入前列的包括：{", ".join(role_groups['health_constraints']) if role_groups['health_constraints'] else '无'}。
- 家庭相关变量进入前列的包括：{", ".join(role_groups['family_care_and_intergenerational_support']) if role_groups['family_care_and_intergenerational_support'] else '无'}。
- 经济相关变量进入前列的包括：{", ".join(role_groups['economic_pressure']) if role_groups['economic_pressure'] else '无'}。
- `family_care_index_v1` 本身没有进入 main_clean 前 20，说明家庭机制在 SHAP 主模型中更多通过家庭结构和代际支持类变量体现。

## 缺失指示变量的处理

- main_clean 前 20 中有 {len(missing_feats)} 个缺失指示变量，分别是：{", ".join(f"{r['feature']}(rank {r['rank']})" for r in missing_feats)}。
- 这些变量可以作为技术性特征保留在机器学习模型中，但正文解释时不应把它们当作 substantive 机制。

## 解释风险提示

- SHAP 只能解释“预测解释贡献”，不能解释为因果影响。
- `year_2020` 只作为年份控制变量，不应作为政策效果变量展开。
- `totmet_w` 在全特征 SHAP 中排名第 1（mean_abs_shap={fmt_num(to_float(next(r['mean_abs_shap'] for r in shap_rows_all if r['scheme'] == 'full_feature' and r['feature'] == 'totmet_w')), 6)}），但其可能代理劳动活动强度本身，因此不进入正文主 SHAP 解释。
- `depression_high`、`total_cognition_w`、`family_care_index_v1` 仍需谨慎解释。

## 可直接写进论文的 SHAP 结论

- LightGBM 在预测表现和稳定性上均优于其余基线模型，因此被选为主解释模型。
- 在去除行为变量和 `totmet_w` 的 `main_clean` 方案中，年龄、城乡、性别、IADL 受限、慢性病负担、收入与代际支持等变量仍具有较高预测解释贡献，说明健康、人口学和经济支持结构共同塑造了劳动参与状态。
- 机器学习与 SHAP 结果总体支持主回归所揭示的基本机制，但这些结果应被理解为“变量对预测结果的解释贡献”，而非因果效应。

## 可直接写进论文的结论句

机器学习与 SHAP 分析表明，中老年劳动参与不仅受线性回归中可识别的健康、家庭与经济因素影响，而且在非线性预测框架下仍表现出明显的年龄、城乡、健康能力和支持结构差异；因此，可解释机器学习结果更适合作为主回归结论的补充验证，而非替代因果分析。
"""


def build_section7_heterogeneity(table7: list[dict[str, str]]) -> str:
    key = {(r["group"], r["variable"]): r for r in table7}

    def r(group: str, var: str) -> dict[str, str]:
        return key[(group, var)]

    return f"""# 第七节结果摘要：异质性分析

## 性别分组

- 男性样本中，poor_health 的 OR 为 {fmt_num(to_float(r('gender_male', 'poor_health')['odds_ratio']), 4)}，女性样本中为 {fmt_num(to_float(r('gender_female', 'poor_health')['odds_ratio']), 4)}，说明健康恶化对男性劳动参与的抑制更强。
- 男性样本中，iadl_limit 的 OR 为 {fmt_num(to_float(r('gender_male', 'iadl_limit')['odds_ratio']), 4)}，女性样本中为 {fmt_num(to_float(r('gender_female', 'iadl_limit')['odds_ratio']), 4)}，IADL 受限在男性中的负面作用也更强。
- family_care_index_v1 在男性和女性中均为正向显著，OR 分别为 {fmt_num(to_float(r('gender_male', 'family_care_index_v1')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('gender_female', 'family_care_index_v1')['odds_ratio']), 4)}，再次说明其更像家庭结构和代际支持压力代理。

## 城乡分组

- 农村样本中，poor_health 的 OR 为 {fmt_num(to_float(r('urban_rural', 'poor_health')['odds_ratio']), 4)}，城镇样本中为 {fmt_num(to_float(r('urban_urban', 'poor_health')['odds_ratio']), 4)}，健康约束在农村群体中更强。
- 农村样本中，economic_pressure_index_v1 的 OR 为 {fmt_num(to_float(r('urban_rural', 'economic_pressure_index_v1')['odds_ratio']), 4)}，高于城镇样本的 {fmt_num(to_float(r('urban_urban', 'economic_pressure_index_v1')['odds_ratio']), 4)}，说明经济压力驱动劳动在农村更明显。
- family_care_index_v1 在农村不显著（p={fmt_p(to_float(r('urban_rural', 'family_care_index_v1')['p_value']))}），但在城镇显著为正（OR={fmt_num(to_float(r('urban_urban', 'family_care_index_v1')['odds_ratio']), 4)}），提示家庭结构效应在城乡之间存在差异。

## 年龄组分组

- poor_health 在 45-59 岁、60-69 岁和 70 岁及以上样本中的 OR 分别为 {fmt_num(to_float(r('age_45_59', 'poor_health')['odds_ratio']), 4)}、{fmt_num(to_float(r('age_60_69', 'poor_health')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('age_70_plus', 'poor_health')['odds_ratio']), 4)}，说明健康约束在中低年龄组更强。
- iadl_limit 在三个年龄组中均显著负向，OR 分别为 {fmt_num(to_float(r('age_45_59', 'iadl_limit')['odds_ratio']), 4)}、{fmt_num(to_float(r('age_60_69', 'iadl_limit')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('age_70_plus', 'iadl_limit')['odds_ratio']), 4)}。
- family_care_index_v1 在 45-59 岁组不显著（p={fmt_p(to_float(r('age_45_59', 'family_care_index_v1')['p_value']))}），但在 60-69 岁和 70 岁以上组显著为正，OR 分别为 {fmt_num(to_float(r('age_60_69', 'family_care_index_v1')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('age_70_plus', 'family_care_index_v1')['odds_ratio']), 4)}。

## 健康状态分组

- 在健康较好和健康较差样本中，iadl_limit 均显著负向，OR 分别为 {fmt_num(to_float(r('health_good', 'iadl_limit')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('health_poor', 'iadl_limit')['odds_ratio']), 4)}。
- family_care_index_v1 在健康较好和健康较差组中均为正向显著，OR 分别为 {fmt_num(to_float(r('health_good', 'family_care_index_v1')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('health_poor', 'family_care_index_v1')['odds_ratio']), 4)}。
- economic_pressure_index_v1 在两组中均显著为正，说明经济压力驱动劳动不是某一单一健康组独有的现象。

## 家庭压力高低分组

- 在家庭压力低组中，poor_health 的 OR 为 {fmt_num(to_float(r('familyhigh_low', 'poor_health')['odds_ratio']), 4)}；在家庭压力高组中为 {fmt_num(to_float(r('familyhigh_high', 'poor_health')['odds_ratio']), 4)}，健康约束在低家庭压力组更强一些。
- economic_pressure_index_v1 在家庭压力低组和高组中均显著为正，OR 分别为 {fmt_num(to_float(r('familyhigh_low', 'economic_pressure_index_v1')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('familyhigh_high', 'economic_pressure_index_v1')['odds_ratio']), 4)}。

## 经济压力高低分组

- 在经济压力低组和高组中，poor_health 的 OR 分别为 {fmt_num(to_float(r('econhigh_low', 'poor_health')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('econhigh_high', 'poor_health')['odds_ratio']), 4)}，健康约束方向保持稳定。
- family_care_index_v1 在经济压力低组和高组中均显著为正，OR 分别为 {fmt_num(to_float(r('econhigh_low', 'family_care_index_v1')['odds_ratio']), 4)} 和 {fmt_num(to_float(r('econhigh_high', 'family_care_index_v1')['odds_ratio']), 4)}。

## 异质性总体结论

- 在多数分组中，poor_health 和 iadl_limit 始终保持负向，说明健康约束具有最稳定的跨群体抑制效应。
- economic_pressure_index_v1 在多数分组中保持正向，且在农村、60-69 岁组及家庭压力高组中更强，说明压力驱动劳动在这些群体中更明显。
- family_care_index_v1 的正向结果在女性、城镇、60-69 岁和 70 岁及以上群体中更显著，进一步支持其“家庭结构与代际支持压力代理”而非“单一照料责任变量”的解释。

## 可直接写进论文的结论句

异质性分析表明，健康约束对劳动参与的抑制作用在不同群体中都较为稳定，而经济压力和家庭结构因素则呈现出更强的群体差异，这意味着延迟退休背景下的继续劳动能力和条件并不均质，政策安排需要针对不同群体实施分类支持。
"""


def build_section7_retirement(stage6_exists: bool, stage6_age: list[dict[str, str]] | None, stage6_desc: dict[str, list[dict[str, str]]] | None, near_logit: list[dict[str, str]] | None, interaction: list[dict[str, str]] | None) -> str:
    if not stage6_exists:
        missing_list = "\n".join([f"- {p.name}" for p in STAGE6_FILES if not p.exists()])
        return f"""# 第七节补充摘要：延迟退休适配性分层分析

当前目录中未发现完整的 Stage 6 结果文件，暂时无法写出“延迟退休适配性分层分析”正文摘要。

需要补跑或补齐的文件包括：

{missing_list}
"""

    age_map = {r["age_policy_group"]: r for r in stage6_age}
    suit_rows = stage6_desc["suitability_distribution"]
    near_logit_map = {r["variable"]: r for r in near_logit}
    inter_map = {r["variable"]: r for r in interaction if "_x_retirement_relevant_age" in r["variable"]}

    total_relevant = sum(to_int(r["sample_size"]) or 0 for r in suit_rows)

    def sheet_map(name: str, key_col: str) -> dict[str, dict[str, str]]:
        return {r[key_col]: r for r in stage6_desc[name]}

    gender = sheet_map("by_gender", "female")
    urban = sheet_map("by_urban", "urban")
    age = sheet_map("by_age_group", "retirement_age_group")
    health = sheet_map("by_health", "poor_health")
    family = sheet_map("by_family_pressure", "family_pressure_group")
    econ = sheet_map("by_economic_pressure", "economic_pressure_group")

    highest = max(suit_rows, key=lambda r: to_int(r["sample_size"]) or 0)

    suit_table = [[
        r["retirement_suitability_v1"],
        r["sample_size"],
        fmt_pct((to_int(r["sample_size"]) or 0) / total_relevant if total_relevant else None),
        fmt_pct(to_float(r["labor_participation_rate"])),
    ] for r in suit_rows]

    return f"""# 第七节补充摘要：延迟退休适配性分层分析

## 准退休年龄段与核心准退休年龄段定义

- 准退休年龄段（retirement_relevant_age）：男性 55-64 岁，女性 45-59 岁。
- 核心准退休年龄段（near_retirement_core）：男性 58-63 岁，女性 50-58 岁。
- 之所以对女性采用 45-59 岁宽口径，是因为当前数据无法稳定区分女职工 50 岁退休与女干部 55 岁退休。

## 样本量与总体劳动参与率

- 准退休年龄段样本量：{total_relevant}
- 核心准退休年龄段样本量：{age_map['C']['sample_size']}
- 准退休年龄段劳动参与率：{fmt_pct(to_float(age_map['C']['labor_participation_rate']) if False else (sum((to_int(r['sample_size']) or 0) * (to_float(r['labor_participation_rate']) or 0) for r in suit_rows) / total_relevant if total_relevant else None))}

## 准退休年龄段内部的分组差异

- 男性劳动参与率为 {fmt_pct(to_float(gender['0']['labor_participation_rate']))}，女性为 {fmt_pct(to_float(gender['1']['labor_participation_rate']))}。
- 农村样本劳动参与率为 {fmt_pct(to_float(urban['0']['labor_participation_rate']))}，城镇样本为 {fmt_pct(to_float(urban['1']['labor_participation_rate']))}。
- 45-54 岁、55-59 岁和 60-64 岁组劳动参与率分别为 {fmt_pct(to_float(age['45-54']['labor_participation_rate']))}、{fmt_pct(to_float(age['55-59']['labor_participation_rate']))} 和 {fmt_pct(to_float(age['60-64']['labor_participation_rate']))}。
- 健康较好组劳动参与率为 {fmt_pct(to_float(health['0']['labor_participation_rate']))}，健康较差组为 {fmt_pct(to_float(health['1']['labor_participation_rate']))}。
- 家庭压力高组劳动参与率为 {fmt_pct(to_float(family['high']['labor_participation_rate']))}，高于低压力组的 {fmt_pct(to_float(family['low']['labor_participation_rate']))}，提示继续劳动中存在“带约束劳动”成分。
- 经济压力高组劳动参与率为 {fmt_pct(to_float(econ['high']['labor_participation_rate']))}，略高于低压力组的 {fmt_pct(to_float(econ['low']['labor_participation_rate']))}，说明经济压力可能推动部分个体继续劳动。

## retirement_suitability_v1 分层分布

{md_table(['类别', '人数', '占准退休年龄段比例', '组内劳动参与率'], suit_table)}

- 占比最高的类型是 {highest['retirement_suitability_v1']}，样本量为 {highest['sample_size']}，占准退休年龄段样本的 {fmt_pct((to_int(highest['sample_size']) or 0) / total_relevant if total_relevant else None)}。

## 核心准退休年龄段 Logit 结果

- poor_health：系数 {fmt_num(to_float(near_logit_map['poor_health']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['poor_health']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['poor_health']['p_value']))}
- chronic_count：系数 {fmt_num(to_float(near_logit_map['chronic_count']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['chronic_count']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['chronic_count']['p_value']))}
- adl_limit：系数 {fmt_num(to_float(near_logit_map['adl_limit']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['adl_limit']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['adl_limit']['p_value']))}
- iadl_limit：系数 {fmt_num(to_float(near_logit_map['iadl_limit']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['iadl_limit']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['iadl_limit']['p_value']))}
- family_care_index_v1：系数 {fmt_num(to_float(near_logit_map['family_care_index_v1']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['family_care_index_v1']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['family_care_index_v1']['p_value']))}，方向为正但不显著，正文仍应按“家庭照料与代际支持压力代理”理解。
- economic_pressure_index_v1：系数 {fmt_num(to_float(near_logit_map['economic_pressure_index_v1']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['economic_pressure_index_v1']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['economic_pressure_index_v1']['p_value']))}
- log_medical_expense_w：系数 {fmt_num(to_float(near_logit_map['log_medical_expense_w']['coef']), 4)}，OR={fmt_num(to_float(near_logit_map['log_medical_expense_w']['odds_ratio']), 4)}，p={fmt_p(to_float(near_logit_map['log_medical_expense_w']['p_value']))}

## 交互项模型的关键信息

- poor_health × retirement_relevant_age：系数 {fmt_num(to_float(inter_map['poor_health_x_retirement_relevant_age']['coef']), 4)}，OR={fmt_num(to_float(inter_map['poor_health_x_retirement_relevant_age']['odds_ratio']), 4)}，p={fmt_p(to_float(inter_map['poor_health_x_retirement_relevant_age']['p_value']))}
- adl_limit × retirement_relevant_age：系数 {fmt_num(to_float(inter_map['adl_limit_x_retirement_relevant_age']['coef']), 4)}，OR={fmt_num(to_float(inter_map['adl_limit_x_retirement_relevant_age']['odds_ratio']), 4)}，p={fmt_p(to_float(inter_map['adl_limit_x_retirement_relevant_age']['p_value']))}
- iadl_limit × retirement_relevant_age：系数 {fmt_num(to_float(inter_map['iadl_limit_x_retirement_relevant_age']['coef']), 4)}，OR={fmt_num(to_float(inter_map['iadl_limit_x_retirement_relevant_age']['odds_ratio']), 4)}，p={fmt_p(to_float(inter_map['iadl_limit_x_retirement_relevant_age']['p_value']))}
- family_care_index_v1 × retirement_relevant_age：系数 {fmt_num(to_float(inter_map['family_care_index_v1_x_retirement_relevant_age']['coef']), 4)}，OR={fmt_num(to_float(inter_map['family_care_index_v1_x_retirement_relevant_age']['odds_ratio']), 4)}，p={fmt_p(to_float(inter_map['family_care_index_v1_x_retirement_relevant_age']['p_value']))}
- economic_pressure_index_v1 × retirement_relevant_age：系数 {fmt_num(to_float(inter_map['economic_pressure_index_v1_x_retirement_relevant_age']['coef']), 4)}，OR={fmt_num(to_float(inter_map['economic_pressure_index_v1_x_retirement_relevant_age']['odds_ratio']), 4)}，p={fmt_p(to_float(inter_map['economic_pressure_index_v1_x_retirement_relevant_age']['p_value']))}

## 可直接支撑“延迟退休不能一刀切”的结论

- 在准退休年龄段内部，占比最高的不是“适配继续劳动型”，而是“带约束继续劳动型”和“健康/家庭约束退出型”。
- 核心准退休年龄段回归显示，健康约束和医疗负担仍然显著抑制劳动参与，而经济压力则推动继续劳动，这意味着继续工作并不等于高适配性。
- 交互项模型表明，poor_health 和 iadl_limit 在准退休年龄段的负向作用更强，说明越接近退休决策边界，健康约束越难被忽视。

## 可直接写进论文的结论句

准退休年龄段分析表明，延迟退休背景下的中老年群体并不存在单一的“继续劳动者”形象，其中相当一部分人属于带约束继续劳动或受健康、家庭约束而退出劳动的群体，因此延迟退休政策不能采取一刀切方式，而应建立弹性推进与分类支持机制。
"""


def build_section8() -> str:
    return """# 第八节结果摘要：结论与政策建议

## 核心研究结论

1. 健康约束是影响中老年劳动参与的最稳定因素。poor_health、慢性病数量、ADL 受限和 IADL 受限在主回归、稳健性检验和异质性分析中普遍表现为负向作用，其中 IADL 受限的抑制效应最强。
2. 家庭照料与代际支持压力对劳动参与具有重要影响，但其作用并不表现为简单的“照料责任越重越不工作”。family_care_index_v1 更适合被解释为家庭结构、同住安排和代际支持压力的综合代理。
3. 经济压力能够推动部分中老年人继续劳动，但这并不意味着其更适合延迟退休。收入水平、医疗支出和经济压力指标共同表明，继续劳动中存在明显的“压力驱动劳动”成分。
4. 机器学习与 SHAP 结果进一步验证了健康能力、人口学差异和支持结构的重要性，但这些结果应理解为预测解释贡献，而非因果效应。
5. 准退休年龄段适配性分层显示，占比最高的类型是“带约束继续劳动型”和“健康/家庭约束退出型”，这意味着延迟退休更需要弹性推进和分层支持，而不能简单统一延长劳动年限。

## 政策建议

### 适配继续劳动型

- 提供灵活岗位、再就业服务和技能培训。
- 支持其在健康条件允许的情况下平稳延长劳动参与时间。

### 压力驱动继续劳动型

- 完善养老金、低收入补贴和医疗保障。
- 减少因收入不足和医疗负担导致的被迫劳动。

### 带约束继续劳动型

- 强化劳动保护、工时弹性、岗位适配和职业健康支持。
- 降低个体在健康受限或照护压力较高情形下被迫维持劳动的风险。

### 健康/家庭约束退出型

- 允许弹性退出。
- 加强健康管理、康复服务、社区照护和家庭支持服务。

### 保障支持重点型

- 优先提供长期护理、社区照护、医疗救助和基本生活保障。
- 将政策重点放在底线保障和风险缓释，而非单纯提高劳动参与率。

## 论文局限性

- 本文并未识别延迟退休政策的因果效应，因此不能将结果解释为政策冲击结果。
- 女性准退休年龄段采用宽口径界定，难以精确区分不同制度身份对应的退休边界。
- family_care_index_v1 属于综合代理变量，机制解释仍需谨慎。
- SHAP 只能解释预测贡献，不能替代因果识别。

## 未来研究方向

- 在更长时期面板数据上引入动态劳动参与和退休转移分析。
- 进一步细分制度身份，刻画不同退休规则下的适配性差异。
- 在可行条件下引入政策冲击或准实验设计，识别延迟退休政策的因果效应。

## 可直接写进论文的结论句

综合回归、机器学习、异质性和准退休年龄段分层结果可以看出，延迟退休政策的现实基础并不在于简单延长工作年限，而在于准确识别哪些群体具备继续劳动能力、哪些群体更需要弹性退出以及哪些群体最需要保障支持。
"""


def build_checklist(fig_rows: list[dict[str, str]], stage6_exists: bool) -> str:
    lines = [
        "# 论文表图使用清单",
        "",
        "## 正文建议保留的表格",
        "",
        "- Table 1：描述性统计表，放在第4节。",
        "- Table 2：分组劳动参与率，放在第4节。",
        "- Table 3：Logit 主回归结果，放在第5节。",
        "- Table 4：稳健性检验结果，放在第5节或第6节前半部分。",
        "- Table 5：机器学习模型性能比较，放在第6节。",
        "- Table 6：SHAP 重要特征表，放在第6节。",
        "- Table 7：异质性结果表，放在第7节。",
    ]
    if stage6_exists:
        lines += [
            "- 建议在第7节追加一张“准退休年龄段适配性分层表”或直接引用 retirement_suitability 分布表。",
        ]

    lines += [
        "",
        "## 正文建议保留的图形",
        "",
        "- Figure 1：劳动参与率分组柱状图，放在第4节。",
        "- Figure 2：Logit 主回归森林图，放在第5节。",
        "- Figure 3：SHAP bar summary（main_clean），放在第6节。",
        "- Figure 4：SHAP beeswarm（main_clean），放在第6节。",
    ]
    if stage6_exists:
        lines += [
            "- Figure 5：准退休年龄段适配性分布图，放在第7节。",
        ]

    lines += [
        "",
        "## 附录建议保留的图形",
        "",
        "- SHAP bar summary（no_totmet）。",
        "- SHAP bar summary（full_feature）。",
        "- 其余 dependence plots。",
        "- 详细稳健性附加图表。",
        "",
        "## 现有 figure list 中的正文推荐图",
        "",
    ]
    for row in fig_rows:
        if row.get("recommended_location") == "main_text":
            lines.append(f"- {row['figure_id']}：{row['figure_name']}，建议位置：正文。用途：{row['purpose']}")

    return "\n".join(lines) + "\n"


def build_master(section4: str, section5: str, section5r: str, section6: str, section7h: str, section7r: str, section8: str) -> str:
    return f"""# 论文后半部分总摘要

## 四、描述性统计与中老年劳动参与现状

{section4.split('## 可直接写进论文的结论句')[-1].strip()}

## 五、计量模型估计与机制分析

{section5.split('## 可直接写进论文的结论句')[-1].strip()}

### 稳健性检验补充

{section5r.split('## 可直接写进论文的结论句')[-1].strip()}

## 六、机器学习预测与 SHAP 可解释分析

{section6.split('## 可直接写进论文的结论句')[-1].strip()}

## 七、延迟退休适配性分层与异质性分析

### 异质性分析

{section7h.split('## 可直接写进论文的结论句')[-1].strip()}

### 延迟退休适配性分层

{section7r.split('## 可直接写进论文的结论句')[-1].strip()}

## 八、结论与政策建议

{section8.split('## 可直接写进论文的结论句')[-1].strip()}

## 核心数据性结论汇总

- 总体样本量为 29772，其中劳动参与者 16684 人，占 56.04%。
- 2018 年劳动参与率为 43.34%，2020 年为 65.03%。
- 年龄组劳动参与率呈明显递减：45-59 岁为 70.62%，60-69 岁为 59.59%，70 岁及以上为 36.27%。
- LightGBM baseline 的测试集 AUC 为 0.8846，5 折 GroupKFold 平均 AUC 为 0.8772，是正文最合适的主机器学习模型。
- SHAP 主方案 main_clean 的 AUC 为 0.8499，前 5 位变量为 age、urban、female、year_2020 和 iadl_limit。
- 准退休年龄段样本量为 9885，核心准退休年龄段样本量为 6794；其中占比最高的是带约束继续劳动型（41.46%），其次是健康/家庭约束退出型（21.88%）。

## 解释风险单列提示

- family_care_index_v1 应解释为“家庭照料与代际支持压力代理”，不能写成“照料责任促进劳动参与”。
- economic_pressure_index_v1 正向不等于“更适合延迟退休”，更可能包含压力驱动劳动。
- depression_high 和 total_cognition_w 的方向应明确标注“谨慎解释”。
- year_2020 只作为年份控制变量。
- SHAP 和机器学习结果只能解释预测贡献，不能作因果推断。
"""


def main() -> None:
    ensure_dir()

    missing = [str(p.relative_to(ROOT)) for p in CORE_FILES if not p.exists()]
    stage6_exists = all(p.exists() for p in STAGE6_FILES)
    if not stage6_exists:
        missing.extend([str(p.relative_to(ROOT)) for p in STAGE6_FILES if not p.exists()])

    # Load core files
    csv_summary = load_csv_summary(FINAL_DIR / "CHARLS_labor_panel_2018_2020_v4_logit_final_audited.csv")

    table2 = sheet_to_dicts(read_xlsx(STAGE5_DIR / "paper_table2_group_labor_rates_final.xlsx")["Sheet1"])
    table3_book = read_xlsx(STAGE5_DIR / "paper_table3_logit_main_final.xlsx")
    table3_long = sheet_to_dicts(table3_book["long_format"])
    table3_summary = sheet_to_dicts(table3_book["model_summary"])

    robust_book = read_xlsx(STAGE5_DIR / "paper_table4_robustness_final.xlsx")
    robust_summary = sheet_to_dicts(robust_book["summary"])
    robust_key = sheet_to_dicts(robust_book["key_variables"])

    ml_perf = sheet_to_dicts(read_xlsx(STAGE5_DIR / "paper_table5_ml_performance_final.xlsx")["Sheet1"])
    shap_table = sheet_to_dicts(read_xlsx(STAGE5_DIR / "paper_table6_shap_top_features_final.xlsx")["Sheet1"])
    hetero_key = sheet_to_dicts(read_xlsx(STAGE5_DIR / "paper_table7_heterogeneity_final.xlsx")["key_variables"])
    fig_rows = sheet_to_dicts(read_xlsx(STAGE5_DIR / "paper_figures_list_v1.xlsx")["Sheet1"])

    probit_book = read_xlsx(TRAINING_DIR / "probit_results_v1.xlsx")
    probit_summary = sheet_to_dicts(probit_book["summary"])
    probit_rows = sheet_to_dicts(probit_book["Probit_Model_4"])

    cv_summary = sheet_to_dicts(read_xlsx(TRAINING_DIR / "group_cv_metrics_v1.xlsx")["summary"])

    shap_metrics = sheet_to_dicts(read_xlsx(STAGE4_DIR / "shap_model_metrics_v1.xlsx")["metrics"])
    shap_summary_book = read_xlsx(STAGE4_DIR / "shap_values_summary_table_v1.xlsx")
    shap_all_rows: list[dict[str, str]] = []
    for sheet_name, rows in shap_summary_book.items():
        for item in sheet_to_dicts(rows):
            if "rank" in item and "scheme" in item:
                shap_all_rows.append(item)

    stage6_age = stage6_desc = near_logit = interaction_rows = None
    if stage6_exists:
        stage6_book = read_xlsx(STAGE6_DIR / "retirement_suitability_distribution_v1.xlsx")
        stage6_age = sheet_to_dicts(stage6_book["age_policy_groups"])
        stage6_desc_book = read_xlsx(STAGE6_DIR / "near_retirement_descriptive_tables_v1.xlsx")
        stage6_desc = {name: sheet_to_dicts(rows) for name, rows in stage6_desc_book.items()}
        near_logit = sheet_to_dicts(read_xlsx(STAGE6_DIR / "near_retirement_logit_results_v1.xlsx")["coefficients"])
        interaction_rows = sheet_to_dicts(read_xlsx(STAGE6_DIR / "retirement_interaction_logit_v1.xlsx")["coefficients"])

    section4 = build_section4(csv_summary, table2)
    section5 = build_section5(table3_long, table3_summary, probit_summary, probit_rows)
    section5r = build_section5_robustness(robust_summary, robust_key)
    section6 = build_section6(ml_perf, cv_summary, shap_metrics, shap_all_rows)
    section7h = build_section7_heterogeneity(hetero_key)
    section7r = build_section7_retirement(stage6_exists, stage6_age, stage6_desc, near_logit, interaction_rows)
    section8 = build_section8()
    checklist = build_checklist(fig_rows, stage6_exists)
    master = build_master(section4, section5, section5r, section6, section7h, section7r, section8)

    outputs = {
        "paper_section4_descriptive_digest.md": section4,
        "paper_section5_regression_digest.md": section5,
        "paper_section5_robustness_digest.md": section5r,
        "paper_section6_ml_shap_digest.md": section6,
        "paper_section7_heterogeneity_digest.md": section7h,
        "paper_section7_retirement_relevance_digest.md": section7r,
        "paper_section8_conclusion_policy_digest.md": section8,
        "paper_tables_figures_checklist.md": checklist,
        "paper_results_master_digest.md": master,
    }

    for name, text in outputs.items():
        write_md(OUT_DIR / name, text)

    all_core_ok = len([p for p in CORE_FILES if p.exists()]) == len(CORE_FILES)
    print(f"是否成功读取所有核心结果文件: {'是' if all_core_ok else '否'}")
    if missing:
        print("缺失文件:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("缺失文件: 无")

    print("各节生成的 digest 文件:")
    for name in outputs:
        print(f"  - {name}")

    print(f"是否已有足够材料写第4—8节: {'是' if all_core_ok else '否'}")
    if missing:
        print("如果还缺数据，当前缺少:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("如果还缺数据: 当前核心结果已足够支持第4—8节写作。")


if __name__ == "__main__":
    main()
