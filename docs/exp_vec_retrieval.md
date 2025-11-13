# 实验日志：语义向量检索（Vec）在抗菌药物推荐任务上的表现

时间：YYYY-MM-DD  
环境：conda 环境 `kg-rag`，Windows，项目路径 `E:\project\kg_rag_demo`

---

## 1. 目的

在同一批医学问题（抗菌药物选择）上，对比：

- 基于知识图谱 + PPR 的检索方法（KG+PPR）  
- 基于语义向量的检索方法（Vec，sentence-transformers/all-MiniLM-L6-v2）

看：**在“给某类感染列出常用抗生素”这个任务上，向量检索能做到什么水平，和 KG+PPR 有什么区别。**

---

## 2. 数据与评测设置

### 2.1 语料

- 文件：`data/docs.jsonl`
- 内容：约 300 篇 PubMed 摘要，主题集中在：
  - 各类细菌感染（如金黄色葡萄球菌、MRSA、肺炎链球菌、铜绿假单胞菌等）
  - 抗菌药物使用与耐药性
  - 相关的临床治疗与疗效分析

该语料主要用于支撑“**细菌感染 → 抗生素选择**”这一类问题。

### 2.2 问题集合

- 文件：`data/qa_med_questions.jsonl`
- 共 14 个问题：
  - 7 个英文问题（`q1`~`q7`）
  - 7 个中文问题（`q1_zh`~`q7_zh`）
- 问题类型：
  - MRSA / MSSA 相关用药
  - 尿路感染一线治疗
  - 社区获得性肺炎常用口服药物
  - 链球菌性咽炎标准用药
  - 蜂窝织炎（非化脓性）门诊口服治疗
  - 幽门螺杆菌根除方案中的常见抗生素
  - 抗假单胞菌经验性治疗药物（如 HAP / VAP）

### 2.3 标准答案（Gold）

- 每个问题都手工标注了一个**标准药物列表**：
  - 例如 MRSA 相关问题的 gold 列表包括：
    - vancomycin, linezolid, daptomycin, teicoplanin,
      ceftaroline, clindamycin, doxycycline,
      trimethoprim-sulfamethoxazole (TMP-SMX),
      tetracycline, cefazolin, cephalexin, oxacillin,
      nafcillin, dicloxacillin, flucloxacillin, gentamicin, rifampin 等
- 评测指标：
  - 每道题都计算：
    - Precision（预测出的药里有多少是对的）
    - Recall（该说的药里你说到了多少）
    - F1（综合评分）
  - 同时计算：
    - 微平均（micro-average）
    - 宏平均（macro-average）

---

## 3. 方法说明：语义向量检索（Vec）

### 3.1 索引构建

脚本：

```bash
python src/build_vec_index_vec.py
