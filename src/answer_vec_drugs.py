import argparse
import json
import os
from typing import List, Dict, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DOCS_PATH = os.path.join("data", "docs.jsonl")
VEC_EMB_PATH = os.path.join("data", "index_vec_emb.npy")


# 你关心的“标准药物名”列表（和 qa_med_questions.jsonl 里的 gold 一致）
CANON_DRUGS: List[str] = [
    # Staph / MRSA 相关
    "cefazolin",
    "ceftaroline",
    "cephalexin",
    "clindamycin",
    "daptomycin",
    "dicloxacillin",
    "doxycycline",
    "flucloxacillin",
    "gentamicin",
    "linezolid",
    "nafcillin",
    "oxacillin",
    "rifampin",
    "teicoplanin",
    "tetracycline",
    "trimethoprim-sulfamethoxazole",
    "vancomycin",

    # 尿路感染
    "fosfomycin",
    "nitrofurantoin",

    # 呼吸道 / 肺炎
    "amoxicillin",
    "azithromycin",
    "clarithromycin",

    # 咽炎
    "penicillin v",
    "benzathine penicillin g",

    # 蜂窝织炎
    "amoxicillin-clavulanate",

    # 幽门螺杆菌
    "levofloxacin",
    "metronidazole",

    # 抗假单胞菌
    "cefepime",
    "ceftazidime",
    "ciprofloxacin",
    "imipenem-cilastatin",
    "meropenem",
    "piperacillin-tazobactam",
]


# 一点简单的别名（主要是连字符 / 斜杠 / 大写问题）
DRUG_SYNONYMS: Dict[str, List[str]] = {
    "trimethoprim-sulfamethoxazole": [
        "trimethoprim-sulfamethoxazole",
        "trimethoprim / sulfamethoxazole",
        "trimethoprim-sulphamethoxazole",
        "co-trimoxazole",
        "cotrimoxazole",
        "tmp-smx",
        "tmp / smx",
    ],
    "amoxicillin-clavulanate": [
        "amoxicillin-clavulanate",
        "amoxicillin / clavulanate",
        "amox-clav",
        "co-amoxiclav",
    ],
    "penicillin v": [
        "penicillin v",
        "penicillin vk",
    ],
    "benzathine penicillin g": [
        "benzathine penicillin g",
        "benzathine benzylpenicillin",
    ],
    # 其他没写别名的，就用名字本身做匹配
}


def build_drug_pattern_map() -> Dict[str, List[str]]:
    """把所有药名和别名都变成小写，用来做包含匹配。"""
    pat = {}
    for d in CANON_DRUGS:
        base = d.lower()
        pats = [base]
        extra = DRUG_SYNONYMS.get(d, [])
        pats.extend([e.lower() for e in extra])
        pat[d] = pats
    return pat


DRUG_PATTERNS = build_drug_pattern_map()


def load_docs(path: str = DOCS_PATH) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def doc_text(doc: Dict) -> str:
    """从一条 doc 里拼一个 '标题 + 摘要/正文' 的长文本。"""
    parts = []
    title = doc.get("title") or doc.get("Title")
    abstract = doc.get("abstract") or doc.get("Abstract")
    text = doc.get("text") or doc.get("Text")

    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    elif text:
        parts.append(text)

    return "\n\n".join(parts)


def load_vec_index() -> np.ndarray:
    emb = np.load(VEC_EMB_PATH)
    return emb


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return mat / norms


def vec_search(
    query: str,
    model: SentenceTransformer,
    emb: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """用向量相似度做 Top-K 检索。"""
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    emb_norm = normalize_matrix(emb)
    scores = emb_norm @ q_vec
    idx = np.argsort(-scores)[:top_k]
    return idx, scores[idx]


def extract_drugs_from_text(text: str) -> List[str]:
    """在一段大文本里，看看有哪些药名 / 别名出现过。"""
    txt = text.lower()
    found: Set[str] = set()

    for canon_name, patterns in DRUG_PATTERNS.items():
        for p in patterns:
            if p in txt:
                found.add(canon_name)
                break

    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="要检索的问题")
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="向量检索 Top-K 文献数（默认 15）",
    )
    args = parser.parse_args()

    # 1. 读文献 + 向量
    docs = load_docs(DOCS_PATH)
    emb = load_vec_index()

    if emb.shape[0] != len(docs):
        raise RuntimeError(
            f"向量条数 {emb.shape[0]} 和 docs 条数 {len(docs)} 不一致，"
            f"请确认 build_vec_index_vec.py 用的也是 {DOCS_PATH}。"
        )

    # 2. 加载向量模型
    model_name = "sentence-transformers/all-mpnet-base-v2"

    print(f"🧠 加载向量模型：{model_name}")
    model = SentenceTransformer(model_name)

    # 3. 做向量检索
    idx, scores = vec_search(args.query, model, emb, top_k=args.k)

    # 4. 把 Top-K 文献拼成一个“大作文”，从里面抽药名
    big_chunks = []
    for i in idx:
        doc = docs[int(i)]
        big_chunks.append(doc_text(doc))
    big_text = "\n\n".join(big_chunks)

    drugs = extract_drugs_from_text(big_text)

    # 5. 打印结果
    print("\n================ QUERY =================")
    print(args.query)

    print("=============== VEC+DRUGS ANSWER =============")
    if drugs:
        print(", ".join(drugs))
    else:
        print("(没有在 Top-K 文献中找到常见药名)")

    print("============= CITATIONS (Top-K) =============")
    for rank, i in enumerate(idx):
        doc = docs[int(i)]
        pmid = doc.get("pmid") or doc.get("PMID") or "?"
        title = (doc.get("title") or doc.get("Title") or "").strip()
        if len(title) > 180:
            title = title[:177] + "..."
        print(f"[{rank+1}] pid=pmid_{pmid}  score={scores[rank]:.4f}")
        print(f"    {title}\n")


if __name__ == "__main__":
    main()
