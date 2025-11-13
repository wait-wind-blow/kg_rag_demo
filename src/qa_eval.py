# src/qa_eval.py
# 功能：批量读 data/qas.jsonl，调用我们现有的“两步检索(线性版)”计算命中率：
# “Top-K 段落的拼接文本，是否包含任一标准答案子串”

import os, json, sys
from retrieve import load_index, build_embeddings, activate_entities, rank_paragraphs

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    os.chdir(root)

    topk = int(sys.argv[1]) if len(sys.argv) >= 2 else 5

    meta, M, C = load_index()
    model, sent_emb = build_embeddings(meta["sents"])

    qas_path = os.path.join("data", "qas.jsonl")
    qas = [json.loads(l) for l in open(qas_path, "r", encoding="utf-8") if l.strip()]

    hit = 0
    for i, ex in enumerate(qas, 1):
        q = ex["question"]
        answers = [a.lower() for a in ex.get("answers", []) if a]

        act_e = activate_entities(q, model, sent_emb, M, meta, R=50, sim_th=0.35, rounds=1)
        results = rank_paragraphs(q, model, sent_emb, C, meta, act_e, alpha=0.3, topk=topk)

        bag = " ".join([meta["doc_texts"].get(pid, "") for pid, _ in results]).lower()
        ok = any(a in bag for a in answers)
        hit += 1 if ok else 0
        print(f"[{i}/{len(qas)}] {'HIT ' if ok else 'MISS'}  Q: {q}")

    print(f"\n简单命中率 (Top-{topk} 段落包含任一答案子串)：{hit}/{len(qas)} = {hit/len(qas):.2%}")

if __name__ == "__main__":
    main()
