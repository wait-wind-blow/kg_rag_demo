import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_docs(jsonl_path):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # 尝试多种字段名，避免 KeyError
            pid = obj.get("pid") or obj.get("pmid") or obj.get("id")
            text = (
                obj.get("text")
                or obj.get("abstract")
                or obj.get("body")
                or obj.get("content")
            )

            if not text:
                continue

            docs.append({"pid": pid, "text": text})
    return docs


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    jsonl_path = data_dir / "docs.jsonl"

    if not jsonl_path.exists():
        print(f"❌ 找不到 {jsonl_path}，先确认已经准备好语料。")
        return

    print(f"📄 读取语料：{jsonl_path}")
    docs = load_docs(jsonl_path)
    print(f"✅ 共读取 {len(docs)} 篇文献")

    # 选择一个比较轻的英文向量模型
    # 换成医学领域向量模型（可以根据需要再改）
    model_name = "sentence-transformers/all-mpnet-base-v2"

    print(f"🧠 加载向量模型：{model_name}")
    model = SentenceTransformer(model_name)

    texts = [d["text"] for d in docs]

    print("⚙️ 开始计算文献向量（embedding）…")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 方便后面用点积=相似度
    )

    out_emb = data_dir / "index_vec_emb.npy"
    out_meta = data_dir / "index_vec_meta.json"

    print(f"💾 保存向量到：{out_emb}")
    np.save(out_emb, embeddings)

    print(f"💾 保存元信息到：{out_meta}")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("✅ 向量索引构建完成！")


if __name__ == "__main__":
    main()
