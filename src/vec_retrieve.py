import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_vec_index(data_dir: Path):
    emb_path = data_dir / "index_vec_emb.npy"
    meta_path = data_dir / "index_vec_meta.json"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("❌ 找不到向量索引文件，请先运行 build_vec_index_vec.py")

    print(f"📥 加载向量：{emb_path}")
    emb = np.load(emb_path)

    print(f"📥 加载元信息：{meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return emb, meta


def vec_search(query, model, emb, meta, top_k=5):
    # 计算 query 的向量
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # 点积 = 余弦相似度（因为已 normalize）
    scores = emb @ q_emb  # (N,) 向量

    # 取 top-k
    idx = np.argsort(-scores)[:top_k]

    results = []
    for rank, i in enumerate(idx, start=1):
        item = meta[i]
        results.append(
            {
                "rank": rank,
                "pid": item.get("pid"),
                "score": float(scores[i]),
                "text": item.get("text", "")[:400].replace("\n", " "),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="查询问题（英文）")
    parser.add_argument(
        "--k", type=int, default=5, help="返回前多少条（top-k）文献"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # 1. 加载索引
    emb, meta = load_vec_index(data_dir)

    # 2. 加载同一个向量模型（要和 build_vec_index_vec.py 里的一致）
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"🧠 加载向量模型：{model_name}")
    model = SentenceTransformer(model_name)

    # 3. 做检索
    print("\n================ QUERY =================")
    print(args.query)
    print("=============== VEC TOP-K =============")
    results = vec_search(args.query, model, emb, meta, top_k=args.k)
    for r in results:
        print(f"[{r['rank']}] pid={r['pid']}  score={r['score']:.4f}")
        print(f"    {r['text']}...")
        print()


if __name__ == "__main__":
    main()
