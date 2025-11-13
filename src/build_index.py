# src/build_index.py
# 功能：读取 data/docs.jsonl 里的段落 -> 分句 -> 用 en_core_sci_md 抽实体
#       生成稀疏矩阵 M(句子x实体)、C(段落x实体) 并保存到项目根目录

import json, re, sys, os
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def split_sentences(text: str):
    """超简单分句：按 . ? ! 后的空格切。你也可以替换成更强的分句器。"""
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def make_csr(pairs, n_rows, n_cols):
    """把 (row, col) 对转成稀疏矩阵"""
    if pairs:
        rows, cols = zip(*pairs)
    else:
        rows, cols = [], []
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

def load_md():
    """优先使用你已安装的 en_core_sci_md；加载失败就报错（按你要求不自动降级）"""
    try:
        import en_core_sci_md
        nlp = en_core_sci_md.load()
        print("✅ 已加载 en_core_sci_md 0.5.4")
        return nlp
    except Exception as e:
        print("❌ 没找到 en_core_sci_md，请先安装：")
        print("   python -m pip install <本地路径或官方 tar.gz>")
        raise

def main():
    # 0) 切到工程根目录（保证输出文件落在根目录）
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    os.chdir(root)

    # 1) 加载 NER
    nlp = load_md()

    # 2) 读取段落
    docs_path = os.path.join("data", "docs.jsonl")
    try:
        docs = [json.loads(l) for l in open(docs_path, "r", encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        print("❌ 找不到 data/docs.jsonl。请创建后再运行。")
        sys.exit(1)
    if not docs:
        print("❌ data/docs.jsonl 为空。")
        sys.exit(1)

    ent2id = {}           # 实体字符串 -> ID
    sents = []            # 所有句子文本
    sent_docid = []       # 每个句子对应的段落ID
    para_ent_pairs = []   # (段落索引, 实体ID)
    sent_ent_pairs = []   # (句子索引, 实体ID)
    doc_ids = []          # 段落ID（与 docs 顺序一致）
    doc_texts = {}        # 段落ID -> 原文

    print(f"🔧 共 {len(docs)} 个段落，开始分句 + 抽实体…（模型：en_core_sci_md）")
    for di, d in enumerate(tqdm(docs)):
        pid = d["id"]; text = d["text"]
        doc_ids.append(pid)
        doc_texts[pid] = text

        ents_para = set()
        cur_sents = split_sentences(text)
        for sent in cur_sents:
            sid = len(sents)
            sents.append(sent)
            sent_docid.append(pid)

            # NER：实体只要字符串，不做复杂规范化
            doc = nlp(sent)
            ents_sent = set(e.text.strip() for e in doc.ents if e.text.strip())
            for e in ents_sent:
                if e not in ent2id:
                    ent2id[e] = len(ent2id)
                sent_ent_pairs.append((sid, ent2id[e]))
            ents_para |= ents_sent

        for e in ents_para:
            para_ent_pairs.append((di, ent2id[e]))

    # 3) 稀疏矩阵
    M = make_csr(sent_ent_pairs, n_rows=len(sents), n_cols=len(ent2id))
    C = make_csr(para_ent_pairs, n_rows=len(docs),  n_cols=len(ent2id))

    # 4) 保存到工程根目录
    np.savez_compressed(
        "index_tri_graph.npz",
        M_data=M.data, M_indices=M.indices, M_indptr=M.indptr, M_shape=M.shape,
        C_data=C.data, C_indices=C.indices, C_indptr=C.indptr, C_shape=C.shape
    )
    meta = {
        "docs": doc_ids,
        "doc_texts": doc_texts,
        "sents": sents,
        "sent_docid": sent_docid,
        "ent2id": ent2id
    }
    with open("index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("✅ 索引完成：")
    print(f"   句子数 = {len(sents)}")
    print(f"   实体数 = {len(ent2id)}")
    print(f"   段落数 = {len(docs)}")
    print("   已生成 index_tri_graph.npz 与 index_meta.json")

if __name__ == "__main__":
    main()
