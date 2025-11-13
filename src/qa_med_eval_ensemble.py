import csv
from collections import Counter
from pathlib import Path


def parse_drug_list(s: str):
    """把 'a;b;c' 这种字符串变成 ['a','b','c']，处理空和'(无)'."""
    if s is None:
        return []
    s = s.strip()
    if not s or s == "(无)":
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def eval_one(gold, pred):
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1)


def load_csv_by_id(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"]
            rows[qid] = row
    return rows


def main():
    base_dir = Path(__file__).resolve().parents[1]
    runs_dir = base_dir / "runs"

    bm25_path = runs_dir / "qa_med_eval_bm25.csv"
    vec_path = runs_dir / "qa_med_eval_vec.csv"
    kg_path = runs_dir / "qa_med_eval.csv"

    if not (bm25_path.exists() and vec_path.exists() and kg_path.exists()):
        print("❌ 找不到某些输入文件，请确认以下文件都存在：")
        print(f"  - {bm25_path}")
        print(f"  - {vec_path}")
        print(f"  - {kg_path}")
        return

    print("📄 读取 BM25 / Vec / KG+PPR 评测结果…")
    bm25_rows = load_csv_by_id(bm25_path)
    vec_rows = load_csv_by_id(vec_path)
    kg_rows = load_csv_by_id(kg_path)

    # 用 BM25 的 gold_drugs 作为“金标准”（三份文件应该都是同一批问题）
    qids = sorted(bm25_rows.keys())

    out_path = runs_dir / "qa_med_eval_ensemble.csv"
    out_f = out_path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "id",
        "method",
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
    ]
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()

    # 用来做 micro / macro 统计
    stats = {
        "UnionAll": {"tp": 0, "fp": 0, "fn": 0, "p_list": [], "r_list": [], "f1_list": []},
        "Majority2": {"tp": 0, "fp": 0, "fn": 0, "p_list": [], "r_list": [], "f1_list": []},
    }

    for qid in qids:
        row_b = bm25_rows[qid]
        row_v = vec_rows.get(qid, {})
        row_k = kg_rows.get(qid, {})

        question = row_b.get("question", "")
        gold = parse_drug_list(row_b.get("gold_drugs", ""))

        bm_pred = parse_drug_list(row_b.get("pred_drugs", ""))
        vec_pred = parse_drug_list(row_v.get("pred_drugs", ""))
        kg_pred = parse_drug_list(row_k.get("pred_drugs", ""))

        # --- 方案1：三家药物取并集（谁说过就算谁） ---
        union3 = sorted(set(bm_pred) | set(vec_pred) | set(kg_pred))
        res_u = eval_one(gold, union3)
        stats["UnionAll"]["tp"] += res_u["tp"]
        stats["UnionAll"]["fp"] += res_u["fp"]
        stats["UnionAll"]["fn"] += res_u["fn"]
        stats["UnionAll"]["p_list"].append(res_u["p"])
        stats["UnionAll"]["r_list"].append(res_u["r"])
        stats["UnionAll"]["f1_list"].append(res_u["f1"])

        writer.writerow(
            dict(
                id=qid,
                method="UnionAll",
                question=question,
                gold_size=len(gold),
                pred_size=len(union3),
                tp=res_u["tp"],
                fp=res_u["fp"],
                fn=res_u["fn"],
                precision=f"{res_u['p']:.4f}",
                recall=f"{res_u['r']:.4f}",
                f1=f"{res_u['f1']:.4f}",
                gold_drugs=";".join(gold),
                pred_drugs=";".join(union3),
            )
        )

        # --- 方案2：至少两家同意（多数表决，≥2/3） ---
        cnt = Counter(bm_pred + vec_pred + kg_pred)
        maj2 = sorted([d for d, c in cnt.items() if c >= 2])
        res_m = eval_one(gold, maj2)
        stats["Majority2"]["tp"] += res_m["tp"]
        stats["Majority2"]["fp"] += res_m["fp"]
        stats["Majority2"]["fn"] += res_m["fn"]
        stats["Majority2"]["p_list"].append(res_m["p"])
        stats["Majority2"]["r_list"].append(res_m["r"])
        stats["Majority2"]["f1_list"].append(res_m["f1"])

        writer.writerow(
            dict(
                id=qid,
                method="Majority2",
                question=question,
                gold_size=len(gold),
                pred_size=len(maj2),
                tp=res_m["tp"],
                fp=res_m["fp"],
                fn=res_m["fn"],
                precision=f"{res_m['p']:.4f}",
                recall=f"{res_m['r']:.4f}",
                f1=f"{res_m['f1']:.4f}",
                gold_drugs=";".join(gold),
                pred_drugs=";".join(maj2),
            )
        )

    out_f.close()
    print(f"✅ 已写出 ensemble 结果到：{out_path}")

    # 计算整体 micro / macro
    def summarize(name):
        s = stats[name]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0

        p_macro = sum(s["p_list"]) / len(s["p_list"]) if s["p_list"] else 0.0
        r_macro = sum(s["r_list"]) / len(s["r_list"]) if s["r_list"] else 0.0
        f1_macro = sum(s["f1_list"]) / len(s["f1_list"]) if s["f1_list"] else 0.0

        print(f"\n📊 方法：{name}")
        print(f"   Micro: P={p_micro:.3f}  R={r_micro:.3f}  F1={f1_micro:.3f}")
        print(f"   Macro: P={p_macro:.3f}  R={r_macro:.3f}  F1={f1_macro:.3f}")

    summarize("UnionAll")
    summarize("Majority2")


if __name__ == "__main__":
    main()
