# src/answer.py
# 作用：基于我们已有的索引与两步检索，输出一个“答案 + 引用段落”。
# 说明：为了先跑通，这里用非常简单的规则法从证据文本里“抠答案”（例如国籍词）。
#      后面你需要的话，我们再换成大模型生成＆严格引用。

import os, sys, re, json
import numpy as np
from retrieve import load_index, build_embeddings, activate_entities, rank_paragraphs

# 一小撮常见“国籍/民族”形容词（demo 用，够我们先跑通）
DEMONYMS = [
    "German","French","Italian","Spanish","English","American","Chinese","Japanese","Korean","Russian","Indian",
    "Canadian","Brazilian","Mexican","Turkish","Dutch","Swedish","Norwegian","Finnish","Danish","Swiss",
    "Polish","Greek","Portuguese","Austrian","Hungarian","Irish","Scottish","Welsh","Czech","Slovak",
    "Ukrainian","Romanian","Bulgarian","Serbian","Croatian","Bosnian","Slovenian","Lithuanian","Latvian",
    "Estonian","Icelandic","Australian","New Zealander","Egyptian","Israeli","Iranian","Iraqi","Pakistani",
    "Bangladeshi","Sri Lankan","Thai","Vietnamese","Indonesian","Malaysian","Filipino"
]
# 一些国家名 -> 常见国籍词 映射（demo 用）
COUNTRY_TO_DEMONYM = {
    "Germany":"German","France":"French","Italy":"Italian","Spain":"Spanish","England":"English","United Kingdom":"English",
    "USA":"American","United States":"American","China":"Chinese","Japan":"Japanese","Korea":"Korean","Russia":"Russian",
    "India":"Indian","Austria":"Austrian","Netherlands":"Dutch","Switzerland":"Swiss","Portugal":"Portuguese","Greece":"Greek"
}

def pick_answer_from_text(question: str, evidence_text: str):
    # 1) 先匹配 DEMONYMS（国籍形容词）
    for dem in DEMONYMS:
        if re.search(rf"\b{re.escape(dem)}\b", evidence_text, flags=re.IGNORECASE):
            return dem

    # 2) 再看看是否出现了国家名，映射成国籍词
    for country, dem in COUNTRY_TO_DEMONYM.items():
        if re.search(rf"\b{re.escape(country)}\b", evidence_text, flags=re.IGNORECASE):
            return dem

    # 3) 特例：句式 “He was a/an XXX king/man/woman/…” 抠出 XXX
    m = re.search(r"\b(?:He|She|They)\s+was\s+(?:an?\s+)?([A-Z][a-z]+)\b", evidence_text)
    if m:
        return m.group(1)

    # 4) 实在不行，返回 insufficient
    return "insufficient"

def main():
    if len(sys.argv) < 2:
        print('用法：python src\\answer.py "你的问题" [topk]')
        print('示例：python src\\answer.py "What was the nationality of Beatrice I\'s husband?" 5')
        sys.exit(0)

    query = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    # 加载索引与句向量
    meta, M, C = load_index()
    model, sent_emb = build_embeddings(meta["sents"])

    # 两步检索
    act_e = activate_entities(query, model, sent_emb, M, meta, R=50, sim_th=0.35, rounds=1)
    results = rank_paragraphs(query, model, sent_emb, C, meta, act_e, alpha=0.3, topk=topk)

    # 拼接证据文本
    evidences = []
    for pid, sc in results:
        text = meta["doc_texts"].get(pid, "")
        evidences.append((pid, sc, text))
    bag = " ".join(t for _, _, t in evidences)

    # 规则法“抠答案”
    answer = pick_answer_from_text(query, bag)

    # 打印
    print("\n================= QUERY =================")
    print(query)
    print("=============== ANSWER =================")
    print(answer)
    print("============= CITATIONS ================")
    for rank, (pid, sc, text) in enumerate(evidences, 1):
        short = (text[:220] + "…") if len(text) > 220 else text
        print(f"[{rank}] pid={pid}  score={sc:.4f}\n    {short}\n")

if __name__ == "__main__":
    main()
