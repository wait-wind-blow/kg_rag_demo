# -*- coding: utf-8 -*-
# BM25 基线 + 药名抽取（与 ppr+kw 的 answer_drugs.py 对照）
import json, re, sys
from rank_bm25 import BM25Okapi

# —— 简单的同义词/别名归一化（和你之前的一致，足够覆盖 MRSA/MSSA 常见用药）——
ALIASES = {
    "vancomycin": ["vancomycin", "vanco"],
    "linezolid": ["linezolid", "zyvox"],
    "daptomycin": ["daptomycin", "cubicin"],
    "teicoplanin": ["teicoplanin"],
    "ceftaroline": ["ceftaroline", "tazef", "teflaro"],
    "clindamycin": ["clindamycin"],
    "doxycycline": ["doxycycline"],
    "tetracycline": ["tetracycline", "minocycline", "minocyclin"],
    "trimethoprim-sulfamethoxazole": [
        "trimethoprim-sulfamethoxazole","trimethoprim sulfamethoxazole",
        "tmp-smx","tmp smx","co-trimoxazole","cotrimoxazole","bactrim","septra"
    ],
    "oxacillin": ["oxacillin", "cloxacillin", "flucloxacillin"],
    "nafcillin": ["nafcillin"],
    "cefazolin": ["cefazolin"],
    "cephalexin": ["cephalexin", "keflex"],
    "dicloxacillin": ["dicloxacillin"],
    "gentamicin": ["gentamicin"],
    "rifampin": ["rifampin", "rifampicin"],
    # 你数据里出现过的新药也加上
    "levonadifloxacin": ["levonadifloxacin"],
}

def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def contains(text_norm: str, alias_norm: str) -> bool:
    # 宽松包含，兼容连字符/空格/斜杠等写法
    return alias_norm in text_norm

def load_docs(path="data/docs.jsonl"):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            pid = j.get("pid") or j.get("id") or "?"
            text = j.get("text") or j.get("abstract") or ""
            if text:
                docs.append((pid, text))
    if not docs:
        raise RuntimeError("data/docs.jsonl 为空，先运行 prepare_pubmed.py")
    return docs

def tokenize(s: str):
    # 非字母数字都当分隔
    return re.findall(r"[A-Za-z0-9\-+/\.]+", s.lower())

def main():
    if len(sys.argv) < 2:
        print("用法：python src\\answer_drugs_bm25.py \"你的问题\" [K]")
        sys.exit(1)
    query = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    docs = load_docs()
    corpus_tokens = [tokenize(text) for _, text in docs]
    bm25 = BM25Okapi(corpus_tokens)

    scores = bm25.get_scores(tokenize(query))
    # 取分数 Top-K 索引
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
    top = [(docs[i][0], docs[i][1], scores[i]) for i in top_idx]

    # 抽药名
    found = []
    found_set = set()
    all_text = " \n ".join([t for _, t, _ in top])
    all_text_norm = norm(all_text)

    for canon, alias_list in ALIASES.items():
        for a in alias_list:
            if contains(all_text_norm, norm(a)):
                if canon not in found_set:
                    found.append(canon)
                    found_set.add(canon)
                break

    print("\n================ QUERY =================")
    print(query)
    print("=============== ANSWER (drug list) =============")
    if found:
        print(", ".join(found))
    else:
        print("(未从 Top-K 引文中匹配到常见药名；可增大 K)")
    print("============= CITATIONS (Top-K) =============")
    for pid, text, sc in top:
        snippet = re.sub(r"\s+", " ", text.strip())
        if len(snippet) > 220:
            snippet = snippet[:217] + "…"
        print(f"[pid={pid}]  score={sc:.4f}\n  {snippet}\n")

if __name__ == "__main__":
    main()
