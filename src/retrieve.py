# src/retrieve.py
# 功能：加载索引（index_meta.json / index_tri_graph.npz），
#       对一个查询做“两步检索”，打印 Top-K 段落与得分。

import json, sys, os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

def load_index():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    os.chdir(root)

    meta = json.load(open("index_meta.json", "r", encoding="utf-8"))
    Z = np.load("index_tri_graph.npz")
    M = csr_matrix((Z['M_data'], Z['M_indices'], Z['M_indptr']), shape=tuple(Z['M_shape']))
    C = csr_matrix((Z['C_data'], Z['C_indices'], Z['C_indptr']), shape=tuple(Z['C_shape']))
    return meta, M, C

def build_embeddings(sents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = normalize(model.encode(sents, convert_to_numpy=True))
    return model, emb

def encode(model, text):
    return normalize(model.encode([text], convert_to_numpy=True))[0]

def activate_entities(query, model, sent_emb, M, meta, R=50, sim_th=0.35, rounds=1):
    """从问题出发 → 找相似句 → 点亮这些句子里的实体；可迭代 rounds 轮"""
    qv = encode(model, query)
    sim = (sent_emb @ qv)
    # 选阈值以上的句子 + Top-R
    top_idx = np.argsort(sim)[-R:]
    S = set(np.where(sim >= sim_th)[0]) | set(top_idx)

    # 句子 -> 实体
    act_e = set()
    for sid in S:
        row = M.getrow(sid)
        act_e.update(row.indices.tolist())

    # 迭代扩一小圈：根据已激活实体，再反找包含它们的句子再过一轮相似度
    for _ in range(rounds - 1):
        if not act_e:
            break
        rows = M[:, list(act_e)].sum(axis=1).A1
        cand = np.where(rows > 0)[0]
        sim2 = (sent_emb[cand] @ qv)
        more = set(cand[np.argsort(sim2)[-R:]])
        S |= more
        for sid in more:
            row = M.getrow(sid)
            act_e.update(row.indices.tolist())

    return act_e

def rank_paragraphs(query, model, sent_emb, C, meta, activated_entities, alpha=0.3, topk=8):
    # 段落向量：把每段的所有句子向量取平均
    para_sids = {}
    for i, pid in enumerate(meta["sent_docid"]):
        para_sids.setdefault(pid, []).append(i)
    para_ids = list(dict.fromkeys(meta["docs"]))  # 保序去重

    P = np.zeros((len(para_ids), sent_emb.shape[1]), dtype=np.float32)
    pid2row = {pid: i for i, pid in enumerate(para_ids)}
    for i, pid in enumerate(para_ids):
        sids = para_sids.get(pid, [])
        if sids:
            P[i] = sent_emb[sids].mean(axis=0)
    P = normalize(P)

    qv = encode(model, query)
    sim = (P @ qv)

    if activated_entities:
        E = list(activated_entities)
        # C 是 (段落行 × 实体列)
        cov = normalize(C[:, E].astype(np.float32), norm='l1', axis=1).sum(axis=1).A1
    else:
        cov = np.zeros(len(para_ids), dtype=np.float32)

    score = alpha * sim + (1 - alpha) * cov
    order = np.argsort(score)[::-1][:topk]
    results = [(para_ids[i], float(score[i])) for i in order]
    return results

def pretty_print(query, results, meta):
    print("\n================= QUERY =================")
    print(query)
    print("=============== TOP-K HITS =============")
    for rank, (pid, sc) in enumerate(results, 1):
        text = meta["doc_texts"].get(pid, "")
        short = (text[:220] + "…") if len(text) > 220 else text
        print(f"[{rank}] pid={pid}  score={sc:.4f}\n    {short}\n")

def main():
    if len(sys.argv) < 2:
        print("用法：python src\\retrieve.py \"你的问题\" [topk]")
        print("示例：python src\\retrieve.py \"What was the nationality of Beatrice I's husband?\" 5")
        sys.exit(0)
    query = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    meta, M, C = load_index()
    model, sent_emb = build_embeddings(meta["sents"])

    act_e = activate_entities(
        query=query,
        model=model,
        sent_emb=sent_emb,
        M=M,
        meta=meta,
        R=50,
        sim_th=0.35,
        rounds=1  # 初学者建议先设 1；要“多跳”可改 2
    )

    results = rank_paragraphs(
        query=query,
        model=model,
        sent_emb=sent_emb,
        C=C,
        meta=meta,
        activated_entities=act_e,
        alpha=0.3,
        topk=topk
    )

    pretty_print(query, results, meta)

if __name__ == "__main__":
    main()
