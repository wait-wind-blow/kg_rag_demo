# 实验结果总览（抗菌药物问答）

## 1. 数据集概况

- 语料：`data/docs.jsonl`
  - 文献数：300
  - 平均摘要长度：约 1700 字符（前面统计过）
  - 主题：主要围绕**细菌感染**和**抗菌药物**，尤其是金黄色葡萄球菌及相关耐药性
- 问题集：`data/qa_med_questions.jsonl`
  - 总问题数：14
  - 英文问题：7 个
  - 中文问题：7 个
  - 涵盖场景：
    - 金黄色葡萄球菌 / MRSA
    - 急性单纯性尿路感染（UTI）
    - 社区获得性肺炎（CAP）
    - A 族链球菌性咽炎
    - 非化脓性蜂窝织炎（链球菌 / MSSA）
    - 幽门螺杆菌根除
    - 假单胞菌相关医院获得性肺炎 / 呼吸机相关性肺炎

---

## 2. 单一问题：MRSA 用药基准实验

**问题：**

> Which antibiotics are commonly used to treat Staphylococcus aureus (including MRSA) infections?

**Gold 药物列表大小：17**
=== GOLD 大小 === 17 ['cefazolin', 'ceftaroline', 'cephalexin', 'clindamycin', 'daptomycin', 'dicloxacillin', 'doxycycline', 'flucloxacillin', 'gentamicin', 'linezolid', 'nafcillin', 'oxacillin', 'rifampin', 'teicoplanin', 'tetracycline', 'trimethoprim-sulfamethoxazole', 'vancomycin']
=== BM25 ===
TP=15 FP=0 FN=2  P=1.000 R=0.882 F1=0.938
BM25覆盖： ['cefazolin', 'ceftaroline', 'cephalexin', 'clindamycin', 'daptomycin', 'doxycycline', 'gentamicin', 'linezolid', 'nafcillin', 'oxacillin', 'rifampin', 'teicoplanin', 'tetracycline', 'trimethoprim-sulfamethoxazole', 'vancomycin']

=== KG+PPR ===
TP=17 FP=0 FN=0  P=1.000 R=1.000 F1=1.000
PPR覆盖： ['cefazolin', 'ceftaroline', 'cephalexin', 'clindamycin', 'daptomycin', 'dicloxacillin', 'doxycycline', 'flucloxacillin', 'gentamicin', 'linezolid', 'nafcillin', 'oxacillin', 'rifampin', 'teicoplanin', 'tetracycline', 'trimethoprim-sulfamethoxazole', 'vancomycin']

=== 差异 ===
PPR 额外找到： ['dicloxacillin', 'flucloxacillin']

（后面可以列出完整 gold 列表）

**方法对比（Top-K = 15 或 20）：**

| 方法          | TP | FP | FN | P    | R    | F1    | 备注                      |
|---------------|----|----|----|------|------|-------|---------------------------|
| BM25          |..  |..  |..  |..    |..    |..     | `python eval_compare.py` |
| KG+PPR        |..  |..  |..  |..    |..    |..     |                           |
| 向量 (MiniLM) |..  |..  |..  |..    |..    |..     |                           |
| 向量 (MPNet)  |..  |..  |..  |..    |..    |..     |                           |



✅ 已写出评测结果到：runs\qa_med_eval.csv
🔢 微平均 (micro)：P=0.215  R=0.289  F1=0.246
🔢 宏平均 (macro)：P=0.225  R=0.243  F1=0.202

> 注：这里的数字可以从你之前 `eval_compare.py` 和 `qa_med_eval_vec.py` 输出里抄过来。

---

## 3. 多问题：14 个临床场景问答（当前语料）

### 3.1 纯 BM25

- Micro: P=0.xxx, R=0.xxx, F1=0.xxx  
- Macro: P=0.xxx, R=0.xxx, F1=0.xxx  

🔢 微平均 (micro)：P=0.284  R=0.233  F1=0.256
🔢 宏平均 (macro)：P=0.251  R=0.237  F1=0.187

> 从刚才这次 `qa_med_eval_bm25.py` 的最后几行复制。

### 3.2 纯向量（all-mpnet-base-v2）

- Micro: P=0.210, R=0.244, F1=0.226  
- Macro: P=0.161, R=0.219, F1=0.169

### 3.3 KG+PPR

- Micro: P=..., R=..., F1=...  
- Macro: P=..., R=..., F1=...

> 这个从 `runs/qa_med_eval_kgppr.csv` 对应的评测脚本最后输出里抄。

### 3.4 集成（Ensemble）

- UnionAll：
  - Micro: P=0.209, R=0.422, F1=0.279
  - Macro: P=0.210, R=0.397, F1=0.249
- Majority2：
  - Micro: P=0.257, R=0.300, F1=0.277
  - Macro: P=0.278, R=0.292, F1=0.241

---

## 4. 小结（可以写成论文里的文字草稿）

- 在单一 MRSA 问题上，KG+PPR 相比 BM25 和纯向量能覆盖更多标准药物，F1 更高。
- 在 14 个多场景问题上，整体指标较低，核心原因是：
  - 当前语料主要集中在金黄色葡萄球菌及相关感染；
  - 对 UTI、幽门螺杆菌、假单胞菌等问题，语料覆盖不足；
  - 部分 gold 药物在语料中几乎不出现或极少出现。
- 即便如此，**集成方法（KG+文本检索结合）在召回上有明显提升**，说明结构化知识 + 文本检索是有互补价值的。
