# src/ppr_retrieve.py  —— PPR + 关键词加权 + “必须含药名”硬过滤 + 负面词惩罚（强化版）

import os, sys, json
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from retrieve import load_index, build_embeddings, activate_entities, encode

# —— 领域同义词 ——
STAPH_SYNONYMS = [
    "staphylococcus", "staphylococcal", "staph",
    "s. aureus", "staphylococcus aureus",
    "mrsa", "mssa"
]

# —— 常见抗葡萄球菌抗生素（覆盖多种写法）——
ANTIBIOTIC_TERMS = [
    # 抗青霉素酶青霉素 & 一代头孢
    "nafcillin", "oxacillin", "dicloxacillin", "flucloxacillin", "cloxacillin",
    "cefazolin", "cephalexin",
    # 抗 MRSA 的一线/二线
    "vancomycin", "linezolid", "daptomycin", "teicoplanin", "ceftaroline",
    # 口服常用
    "clindamycin", "doxycycline",
    "trimethoprim-sulfamethoxazole", "tmp-smx", "co-trimoxazole", "cotrimoxazole",
    "trimethoprim", "sulfamethoxazole",
    # 其他可能出现
    "rifampin", "rifampicin", "gentamicin", "tetracycline"
]

# 不想排前面的“跑偏词”
NEG_TERMS = [
    "diabetes", "cancer", "immunotherapy", "exposure", "association",
    "risk", "microbiota", "observational", "cohort"
]

# —— 超参数（可调）——
REQUIRE_ANTIBIOTIC = True   # ⚠️ 必须命中“药名”才优先
BETA  = 0.30                # 语义相似度权重
GAMMA = 1.20                # 关键词权重（越大越“听话”）
DELTA = 0.50                # 负面词惩罚

def _row_norm(mat: csr_matrix):
    row_sum = np.array(mat.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    return diags(inv) @ mat

def entity_ppr_scores(M: csr_matrix, C: csr_matrix, seed_entities, alpha=0.15, iters=50, tol=1e-6):
    W = (M.T @ M) + (C.T @ C)
    W.setdiag(0); W.eliminate_zeros()
    P = _row_norm(W)

    E = W.shape[0]
    v = np.zeros(E, dtype=np.float32)
    seeds = list(seed_entities) if seed_entities else []
    if seeds:
        v[seeds] = 1.0 / len(seeds)
    else:
        deg = np.array(W.sum(axis=1)).ravel()
        v = deg / (deg.sum() + 1e-12) if deg.sum() > 0 else np.ones(E) / max(E, 1)

    r = v.copy()
    for _ in range(iters):
        r_next = (1 - alpha) * (P @ r) + alpha * v
        if np.linalg.norm(r_next - r, 1) < tol:
            r = r_next; break
        r = r_next
    return r / (r.sum() + 1e-12)

def _hits(text: str, vocab):
    t = text.lower()
    return sum(1 for k in vocab if k in t)

def rank_paragraphs_ppr(query, model, sent_emb, M, C, meta, activated_entities, beta=BETA, gamma=GAMMA, delta=DELTA, topk=5):
    # 段落聚合向量
    para_sids = {}
    for i, pid in enumerate(meta["sent_docid"]):
        para_sids.setdefault(pid, []).append(i)
    para_ids = list(dict.fromkeys(meta["docs"]))

    P_vecs = np.zeros((len(para_ids), sent_emb.shape[1]), dtype=np.float32)
    para_texts = []
    for i, pid in enumerate(para_ids):
        sids = para_sids.get(pid, [])
        if sids:
            P_vecs[i] = sent_emb[sids].mean(axis=0)
        para_texts.append(meta["doc_texts"].get(pid, ""))
    P_vecs = normalize(P_vecs)

    # 查询扩展（含 staph 词就拼上同义词+药名）
    q_low = query.lower()
    need_expand = any(x in q_low for x in ["staphyl", "aureus", "mrsa", "mssa"])
    expanded = (query + " " + " ".join(STAPH_SYNONYMS + ANTIBIOTIC_TERMS)) if need_expand else query
    qv = encode(model, expanded)

    # 语义相似度（归一化）
    sim = (P_vecs @ qv)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)

    # PPR 覆盖
    r_ent = entity_ppr_scores(M, C, activated_entities, alpha=0.15, iters=50)
    cov_ppr = (C @ r_ent)
    cov_ppr = cov_ppr / (cov_ppr.max() + 1e-12)

    # 关键词：Staph 命中 *2 + 药名 *1；负面词惩罚
    kw_staph = np.array([_hits(t, STAPH_SYNONYMS) for t in para_texts], dtype=np.float32)
    kw_ab    = np.array([_hits(t, ANTIBIOTIC_TERMS) for t in para_texts], dtype=np.float32)
    kw = 2 * kw_staph + kw_ab
    if kw.max() > 0: kw = kw / kw.max()

    neg = np.array([_hits(t, NEG_TERMS) for t in para_texts], dtype=np.float32)
    if neg.max() > 0: neg = neg / neg.max()

    score = beta * sim + (1 - beta) * cov_ppr + gamma * kw - delta * neg

    # —— 硬过滤：优先在“命中药名”的集合内排序；若太少再回退全量 ——
    if REQUIRE_ANTIBIOTIC:
        idx_ab = np.where(kw_ab > 0)[0]
        if len(idx_ab) >= max(topk, 3):
            order = idx_ab[np.argsort(score[idx_ab])[::-1][:topk]]
        else:
            order = np.argsort(score)[::-1][:topk]
    else:
        order = np.argsort(score)[::-1][:topk]

    return [(para_ids[i], float(score[i])) for i in order]

def pretty_print(query, results, meta, title="PPR+KW(FILTER) TOP-K"):
    print("\n================= QUERY =================")
    print(query)
    print(f"=============== {title} =============")
    for rank, (pid, sc) in enumerate(results, 1):
        text = meta["doc_texts"].get(pid, "")
        short = (text[:320] + "…") if len(text) > 320 else text
        print(f"[{rank}] pid={pid}  score={sc:.4f}\n    {short}\n")

def main():
    if len(sys.argv) < 2:
        print('用法：python src\\ppr_retrieve.py "你的问题" [topk]')
        sys.exit(0)
    query = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    meta, M, C = load_index()
    model, sent_emb = build_embeddings(meta["sents"])

    # 注意：activate_entities 需要 meta
    seeds = activate_entities(query, model, sent_emb, M, meta, R=100, sim_th=0.25, rounds=1)
    results = rank_paragraphs_ppr(query, model, sent_emb, M, C, meta, seeds, beta=BETA, gamma=GAMMA, delta=DELTA, topk=topk)
    pretty_print(query, results, meta, title="PPR+KW(FILTER) TOP-K")

if __name__ == "__main__":
    main()
