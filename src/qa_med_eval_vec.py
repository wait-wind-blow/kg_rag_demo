import os
import json
import csv
from typing import List, Dict, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# 复用 answer_vec_drugs 里写好的工具函数
from answer_vec_drugs import (
    load_docs,
    load_vec_index,
    vec_search,
    doc_text,
    extract_drugs_from_text,
)

DOCS_PATH = os.path.join("data", "docs.jsonl")
VEC_EMB_PATH = os.path.join("data", "index_vec_emb.npy")
QUEST_PATH = os.path.join("data", "qa_med_questions.jsonl")
OUT_CSV = os.path.join("runs", "qa_med_eval_vec.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 20  # 向量检索的文献数，可以以后调参


def f1_score(p: float, r: float) -> float:
    if p == 0.0 or r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def eval_one_question(
    question: str,
    gold_drugs: List[str],
    docs: List[Dict],
    emb: np.ndarray,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> Tuple[List[str], int, int, int, int, int, float, float, float]:
    """
    对单个问题做：
      - 向量检索 Top-K
      - 把这 K 篇文章拼起来抽药名
      - 和 gold 对比，算 P/R/F1
    """
    # 1) 向量检索
    idx, scores = vec_search(question, model, emb, top_k=top_k)

    # 2) 拼接文本
    big_chunks = []
    for i in idx:
        doc = docs[int(i)]
        big_chunks.append(doc_text(doc))
    big_text = "\n\n".join(big_chunks)

    # 3) 抽药名
    pred_drugs = extract_drugs_from_text(big_text)

    # 4) 计算指标
    gold_set: Set[str] = set(d.lower() for d in gold_drugs)
    pred_set: Set[str] = set(d.lower() for d in pred_drugs)

    tp_set = gold_set & pred_set
    fp_set = pred_set - gold_set
    fn_set = gold_set - pred_set

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    pred_size = len(pred_set)
    gold_size = len(gold_set)

    if pred_size == 0:
        precision = 0.0
    else:
        precision = tp / pred_size

    if gold_size == 0:
        recall = 0.0
    else:
        recall = tp / gold_size

    f1 = f1_score(precision, recall)

    # 为了输出好看，pred_drugs 用原始的大小写顺序
    return pred_drugs, gold_size, pred_size, tp, fp, fn, precision, recall, f1


def load_questions(path: str = QUEST_PATH) -> List[Dict]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qs.append(obj)
    return qs


def main():
    os.makedirs("runs", exist_ok=True)

    print(f"📄 读取问题文件：{QUEST_PATH}")
    questions = load_questions(QUEST_PATH)
    print(f"✅ 共读取 {len(questions)} 个问题")

    print(f"📄 读取语料：{DOCS_PATH}")
    docs = load_docs(DOCS_PATH)
    print(f"✅ 文献条数：{len(docs)}")

    print(f"📦 读取向量索引：{VEC_EMB_PATH}")
    emb = load_vec_index()

    if emb.shape[0] != len(docs):
        raise RuntimeError(
            f"❌ 向量条数 {emb.shape[0]} 和 docs 条数 {len(docs)} 不一致，请确认用同一份 docs.jsonl 重建向量索引。"
        )

    print(f"🧠 加载向量模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    rows = []

    # micro 统计
    micro_tp = micro_fp = micro_fn = 0

    # macro 统计
    macro_p_list = []
    macro_r_list = []
    macro_f1_list = []

    for q in questions:
        qid = q.get("id", "")
        qtext = q.get("question", "")
        gold_drugs = q.get("gold_drugs", [])

        print(f"\n=== 评测问题 {qid} ===")
        print(qtext)

        (
            pred_drugs,
            gold_size,
            pred_size,
            tp,
            fp,
            fn,
            p,
            r,
            f1,
        ) = eval_one_question(qtext, gold_drugs, docs, emb, model, top_k=TOP_K)

        print(
            f"TP={tp} FP={fp} FN={fn}  P={p:.4f} R={r:.4f} F1={f1:.4f}"
        )
        print("预测药物：", ";".join(pred_drugs))

        # 累计 micro
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        # 累计 macro
        macro_p_list.append(p)
        macro_r_list.append(r)
        macro_f1_list.append(f1)

        rows.append(
            {
                "id": qid,
                "question": qtext,
                "gold_size": gold_size,
                "pred_size": pred_size,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{p:.4f}",
                "recall": f"{r:.4f}",
                "f1": f"{f1:.4f}",
                "gold_drugs": ";".join(gold_drugs),
                "pred_drugs": ";".join(pred_drugs),
            }
        )

    # 计算 micro
    if micro_tp + micro_fp == 0:
        micro_p = 0.0
    else:
        micro_p = micro_tp / (micro_tp + micro_fp)

    if micro_tp + micro_fn == 0:
        micro_r = 0.0
    else:
        micro_r = micro_tp / (micro_tp + micro_fn)

    micro_f1 = f1_score(micro_p, micro_r)

    # 计算 macro
    if macro_p_list:
        macro_p = sum(macro_p_list) / len(macro_p_list)
        macro_r = sum(macro_r_list) / len(macro_r_list)
        macro_f1 = sum(macro_f1_list) / len(macro_f1_list)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    print("=" * 80)
    print(f"✅ 已写出评测结果到：{OUT_CSV}")
    print(
        f"🔢 微平均 (micro)：P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}"
    )
    print(
        f"🔢 宏平均 (macro)：P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}"
    )

    # 写 CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "gold_size",
                "pred_size",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "gold_drugs",
                "pred_drugs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
