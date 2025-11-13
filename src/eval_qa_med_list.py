# -*- coding: utf-8 -*-
"""
批量评测医疗问答（英文 + 中文），调用 answer_drugs.py，
对每个问题计算：TP/FP/FN、Precision、Recall、F1，并输出到 CSV。
"""

import json
import os
import subprocess
import csv
from typing import List, Tuple, Set


QA_PATH = "data/qa_med_questions.jsonl"   # 你刚才已经建好的文件
TOP_K = 15                               # 调用 answer_drugs.py 的 K 值
OUT_DIR = "runs"
OUT_CSV = os.path.join(OUT_DIR, "qa_med_eval.csv")


def normalize_drug(name: str) -> str:
    """
    把药名统一成小写、去掉两边空格。
    """
    return name.strip().lower()


def parse_answer_drug_list(stdout: str) -> List[str]:
    """
    从 answer_drugs.py 的输出里，把 “ANSWER (drug list)” 那一行后面的药名提取出来。
    如果没有答案，则返回空列表。
    """
    if not stdout:
        return []

    lines = stdout.splitlines()
    in_answer = False

    for line in lines:
        # 找到答案标题行
        if "ANSWER (drug list)" in line:
            in_answer = True
            continue

        if in_answer:
            text = line.strip()
            # 如果是空行或接下来是分隔线，就结束
            if not text or text.startswith("============"):
                break

            # 如果是那种“未从 Top-K 引文中匹配到常见药名”之类的提示，就视为空答案
            if text.startswith("("):
                return []

            # 正常情况：vancomycin, linezolid, ...
            parts = [normalize_drug(p) for p in text.split(",") if p.strip()]
            return parts

    return []


def eval_one(gold: List[str], pred: List[str]) -> Tuple[int, int, int, float, float, float, List[str]]:
    """
    对单个问题计算：
    - TP / FP / FN
    - Precision / Recall / F1
    并返回去重后的预测列表。
    """
    gold_set: Set[str] = set(normalize_drug(d) for d in gold)
    pred_set: Set[str] = set(normalize_drug(d) for d in pred)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1, sorted(pred_set)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    qa_list = []
    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qa_list.append(obj)

    print(f"共读取到 {len(qa_list)} 个问题（英文 + 中文）")

    rows = []

    sum_tp = sum_fp = sum_fn = 0
    sum_p = sum_r = sum_f1 = 0.0
    n = 0

    for item in qa_list:
        qid = item["id"]
        question = item["question"]
        gold_drugs = item["gold_drugs"]

        print("\n" + "=" * 80)
        print(f"[{qid}] 问题：{question}")
        print(f"金标准药物列表 ({len(gold_drugs)}): {gold_drugs}")

        # 调用 answer_drugs.py
        cmd = [
            "python",
            "src/answer_drugs.py",
            question,
            str(TOP_K),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        stdout = proc.stdout
        if proc.returncode != 0:
            print(f"⚠️ answer_drugs.py 运行出错，returncode={proc.returncode}")
            print("stdout:", stdout)
            print("stderr:", proc.stderr)
            pred_drugs = []
        else:
            # 把完整输出打印一遍方便你看
            print(stdout)
            # 解析答案药物列表
            pred_drugs = parse_answer_drug_list(stdout)
            print(f"解析出的预测药物列表 ({len(pred_drugs)}): {pred_drugs}")

        tp, fp, fn, p, r, f1, pred_unique = eval_one(gold_drugs, pred_drugs)

        print(f"👉 本题结果：TP={tp} FP={fp} FN={fn}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

        rows.append({
            "id": qid,
            "question": question,
            "gold_size": len(gold_drugs),
            "pred_size": len(pred_unique),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": f"{p:.4f}",
            "recall": f"{r:.4f}",
            "f1": f"{f1:.4f}",
            "gold_drugs": ";".join(sorted(set(normalize_drug(d) for d in gold_drugs))),
            "pred_drugs": ";".join(pred_unique),
        })

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_p += p
        sum_r += r
        sum_f1 += f1
        n += 1

    # 写 CSV
    fieldnames = [
        "id", "question",
        "gold_size", "pred_size",
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "gold_drugs", "pred_drugs",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 计算整体指标（微平均 + 宏平均）
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    macro_p = sum_p / n if n > 0 else 0.0
    macro_r = sum_r / n if n > 0 else 0.0
    macro_f1 = sum_f1 / n if n > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"✅ 已写出评测结果到：{OUT_CSV}")
    print(f"🔢 微平均 (micro)：P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"🔢 宏平均 (macro)：P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")


if __name__ == "__main__":
    main()
