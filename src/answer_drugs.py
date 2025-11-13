# src/answer_drugs.py —— 更稳的药名抽取：同义词表 + 文本归一化 + 宽松匹配

import os, sys, re
from collections import Counter
from retrieve import load_index, build_embeddings, activate_entities
from ppr_retrieve import rank_paragraphs_ppr

# 统一到“规范名”: [各种写法/别名]
ALIASES = {
    "vancomycin": ["vancomycin"],
    "linezolid": ["linezolid"],
    "daptomycin": ["daptomycin"],
    "teicoplanin": ["teicoplanin"],
    "ceftaroline": ["ceftaroline"],
    "clindamycin": ["clindamycin"],
    "doxycycline": ["doxycycline"],
    "trimethoprim-sulfamethoxazole": [
        "trimethoprim-sulfamethoxazole",
        "trimethoprim sulfamethoxazole",
        "trimethoprim/sulfamethoxazole",
        "tmp-smx", "co-trimoxazole", "cotrimoxazole"
    ],
    "nafcillin": ["nafcillin"],
    "oxacillin": ["oxacillin"],
    "dicloxacillin": ["dicloxacillin"],
    "flucloxacillin": ["flucloxacillin"],
    "cefazolin": ["cefazolin"],
    "cephalexin": ["cephalexin"],
    "gentamicin": ["gentamicin"],
    "rifampin": ["rifampin", "rifampicin"],
    "tetracycline": ["tetracycline"],
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def boundary_loose_contains(text_norm: str, alias_norm: str) -> bool:
    # 宽松边界：前后不是字母数字即可
    pat = re.compile(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", re.I)
    return bool(pat.search(text_norm))

def main():
    if len(sys.argv) < 2:
        print('用法：python src\\answer_drugs.py "你的问题" [topk]')
        return
    query = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) >= 3 else 12

    # 切到工程根
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    os.chdir(root)

    meta, M, C = load_index()
    model, sent_emb = build_embeddings(meta["sents"])
    seeds = activate_entities(query, model, sent_emb, M, meta, R=100, sim_th=0.25, rounds=1)
    results = rank_paragraphs_ppr(query, model, sent_emb, M, C, meta, seeds, topk=topk)

    texts = [meta["doc_texts"].get(pid, "") for pid, _ in results]
    texts_norm = [normalize_text(t) for t in texts]

    hits = Counter()
    for t in texts_norm:
        for canon, variants in ALIASES.items():
            for v in variants:
                if boundary_loose_contains(t, normalize_text(v)):
                    hits[canon] += 1
                    break  # 该规范名已命中一次就够

    print("\n================= QUERY =================")
    print(query)
    print("=============== ANSWER (drug list) =============")
    if hits:
        ordered = [name for name, _ in hits.most_common()]
        print(", ".join(ordered))
    else:
        print("(Top-K 未匹配到药名；可以把 K 提到 20，或再精炼语料)")

    print("============= CITATIONS (Top-K) =============")
    for i, (pid, sc) in enumerate(results, 1):
        text = meta["doc_texts"].get(pid, "")
        short = (text[:260] + "…") if len(text) > 260 else text
        print(f"[{i}] pid={pid}  score={sc:.4f}\n    {short}\n")

if __name__ == "__main__":
    main()
