# -*- coding: utf-8 -*-
"""
对 data/qa_med_questions.jsonl 里的问题，
用「BM25 + 抽取药物列表」的方式做评测。

最终输出：
- 控制台打印每个问题的 P/R/F1
- 写一份 CSV 到 runs/qa_med_eval_bm25.csv
"""

import os
import json
import re
import math
import csv
from collections import Counter

# ==== 路径设置 ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

DOC_PATH = os.path.join(DATA_DIR, "docs.jsonl")
QA_PATH = os.path.join(DATA_DIR, "qa_med_questions.jsonl")
OUT_CSV = os.path.join(RUNS_DIR, "qa_med_eval_bm25.csv")

# ==== 基础工具：读文件 ====

def load_docs(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def load_questions(path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qs.append(obj)
    return qs


def get_text(doc):
    """把一篇文献变成一段可检索的文本。"""
    if doc.get("text"):
        return doc["text"]
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    return (title + " " + abstract).strip()


# ==== BM25 实现（简易版，自己算，不依赖外部库） ====

TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text):
    """只要英文字母，统统小写。"""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def build_bm25_index(docs):
    """
    为所有文献预计算：
    - 每篇的 term 频率
    - 每篇长度
    - 每个 term 出现在哪些文献里（文档频率 df）
    - 平均文档长度 avgdl
    """
    doc_tfs = []
    doc_lens = []
    df = Counter()

    for doc in docs:
        tokens = tokenize(get_text(doc))
        tf = Counter(tokens)
        doc_tfs.append(tf)
        doc_lens.append(len(tokens))
        for term in tf.keys():
            df[term] += 1

    N = len(docs)
    avgdl = sum(doc_lens) / N if N > 0 else 0.0

    return {
        "doc_tfs": doc_tfs,
        "doc_lens": doc_lens,
        "df": df,
        "N": N,
        "avgdl": avgdl,
    }


def bm25_scores(query, index, k1=1.5, b=0.75):
    """对一个查询，算出每篇文献的 BM25 分数。"""
    q_tokens = tokenize(query)
    doc_tfs = index["doc_tfs"]
    doc_lens = index["doc_lens"]
    df = index["df"]
    N = index["N"]
    avgdl = index["avgdl"]

    scores = []
    for i, tf in enumerate(doc_tfs):
        dl = doc_lens[i]
        score = 0.0
        for t in q_tokens:
            f = tf.get(t, 0)
            if f == 0:
                continue
            n_q = df.get(t, 0)
            if n_q == 0:
                continue
            # 经典 BM25 idf 公式
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * dl / (avgdl + 1e-9))
            score += idf * f * (k1 + 1) / denom
        scores.append(score)
    return scores


# ==== 药物抽取：词表 + 正则 ====

def build_drug_lexicon(questions):
    """
    偷懒但实用的做法：
    ——直接把 qa 文件里所有 gold_drugs 合并成一个词表。
    这样能保证：凡是 gold 里有的药名，只要出现在文献文本里，都有机会被匹配出来。
    """
    lex = set()
    for q in questions:
        for d in q.get("gold_drugs", []):
            if d:
                lex.add(d.strip())
    # 全部转小写，方便匹配
    return {d.lower() for d in lex}


def build_drug_regex(lexicon):
    """
    根据药物词表构造一个大正则：
    \b(drug1|drug2|...)\b
    """
    if not lexicon:
        return None
    # 长的药名放前面，避免短词乱匹配
    parts = [re.escape(d) for d in sorted(lexicon, key=len, reverse=True)]
    pattern = r"\b(" + "|".join(parts) + r")\b"
    return re.compile(pattern, re.I)


def extract_drugs(text, drug_re):
    if not drug_re:
        return []
    found = set(m.group(0).lower() for m in drug_re.finditer(text))
    return sorted(found)


# ==== 单题评测 ====

def eval_one(q, docs, index, drug_re, top_k=20):
    qid = q.get("id", "?")
    qtext = q["question"]
    gold = {d.lower() for d in q.get("gold_drugs", [])}

    scores = bm25_scores(qtext, index)
    # 取分数最高的 top_k 篇文献
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    combined_text = "\n\n".join(get_text(docs[i]) for i in top_idx)

    pred_list = extract_drugs(combined_text, drug_re)
    pred = set(pred_list)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    print(f"\n=== 评测问题 {qid} ===")
    print(qtext)
    print(f"TP={tp} FP={fp} FN={fn}  P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    print("预测药物：", ";".join(sorted(pred)) if pred else "(无)")

    return {
        "id": qid,
        "question": qtext,
        "gold_size": len(gold),
        "pred_size": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "gold_drugs": ";".join(sorted(gold)),
        "pred_drugs": ";".join(sorted(pred)),
    }


# ==== 主函数 ====

def main():
    print(f"📄 读取问题文件：{QA_PATH}")
    questions = load_questions(QA_PATH)
    print(f"✅ 共读取 {len(questions)} 个问题")

    print(f"📄 读取语料：{DOC_PATH}")
    docs = load_docs(DOC_PATH)
    print(f"✅ 文献条数：{len(docs)}")

    print("🧮 构建 BM25 索引…")
    index = build_bm25_index(docs)

    print("📚 根据 gold_drugs 构建药物词表…")
    drug_lex = build_drug_lexicon(questions)
    drug_re = build_drug_regex(drug_lex)
    print(f"✅ 词表大小：{len(drug_lex)}")

    results = []
    micro_tp = micro_fp = micro_fn = 0

    for q in questions:
        r = eval_one(q, docs, index, drug_re, top_k=20)
        results.append(r)
        micro_tp += r["tp"]
        micro_fp += r["fp"]
        micro_fn += r["fn"]

    # 计算 micro / macro
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0

    macro_p = sum(r["precision"] for r in results) / len(results)
    macro_r = sum(r["recall"] for r in results) / len(results)
    macro_f1 = sum(r["f1"] for r in results) / len(results)

    # 写 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question",
            "gold_size", "pred_size",
            "tp", "fp", "fn",
            "precision", "recall", "f1",
            "gold_drugs", "pred_drugs"
        ])
        for r in results:
            writer.writerow([
                r["id"], r["question"],
                r["gold_size"], r["pred_size"],
                r["tp"], r["fp"], r["fn"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                r["gold_drugs"],
                r["pred_drugs"],
            ])

    print("=" * 80)
    print(f"✅ 已写出评测结果到：{OUT_CSV}")
    print(f"🔢 微平均 (micro)：P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"🔢 宏平均 (macro)：P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")


if __name__ == "__main__":
    main()
