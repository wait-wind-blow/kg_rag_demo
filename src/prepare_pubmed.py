# src/prepare_pubmed.py —— 三重兜底抓取版
# 功能：从 PubMed 批量抓“题目+摘要”，写入 data/docs.jsonl（覆盖旧文件）

import os, sys, time, json, html
from typing import List, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

from Bio import Entrez

# 必填：真实邮箱（NCBI 要求）
Entrez.email = "windwait0@gmail.com"
# 可填你的 NCBI API Key（没有就留空）
Entrez.api_key = os.environ.get("NCBI_API_KEY", "")

# 主题关键词（可根据需要调整）
QUERY = '(pneumonia[Title/Abstract]) OR ("Streptococcus pneumoniae"[Title/Abstract]) OR (pneumococcal[Title/Abstract]) OR (antibiotic[Title/Abstract])'

UA_HDR = {"User-Agent": f"kg-rag-demo/1.0 ({Entrez.email})"}

# ============== 搜索（先 XML，失败转 JSON） =================
def search_pmids(query: str, retmax: int) -> List[str]:
    # 1) Entrez XML
    for attempt in range(3):
        try:
            h = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance", retmode="xml")
            r = Entrez.read(h)
            ids = r.get("IdList", [])
            if ids:
                return ids
        except Exception as e:
            print(f"⚠️ esearch XML 尝试 {attempt+1}/3 失败：{e}")
            time.sleep(0.7 * (attempt + 1))
    # 2) 备用 JSON
    try:
        params = dict(db="pubmed", term=query, retmax=str(retmax), sort="relevance", retmode="json")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(params)
        data = urlopen(Request(url, headers=UA_HDR), timeout=30).read()
        j = json.loads(data)
        return j.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"❌ esearch JSON 也失败：{e}")
        return []

# ============== 解析 XML 的小工具 =================
def _parse_pubmed_xml(xbytes: bytes) -> Iterable[dict]:
    """解析 PubMed efetch 的 XML：提取 PMID、Title、AbstractText"""
    root = ET.fromstring(xbytes)
    # PubmedArticleSet / PubmedArticle
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        art_node = art.find(".//MedlineCitation/Article")
        if art_node is None or not pmid:
            continue
        # 标题
        title_el = art_node.find("./ArticleTitle")
        title = ""
        if title_el is not None:
            title = "".join(title_el.itertext()).strip()
        # 摘要（可能多段）
        abstract_texts = []
        for at in art_node.findall("./Abstract/AbstractText"):
            abstract_texts.append("".join(at.itertext()).strip())
        abstract = " ".join([t for t in abstract_texts if t])
        text = " ".join([title, abstract]).strip()
        if not text:
            continue
        text = html.unescape(text).replace("\n", " ").strip()
        yield {"id": f"pmid_{pmid}", "text": text}

# ============== 抓取一批：三重兜底 =================
def _efetch_entrez_xml(id_list: List[str]) -> List[dict]:
    """优先：Biopython Entrez XML"""
    try:
        h = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
        r = Entrez.read(h)  # 先用 biopython 解析，若失败再走我们自己的解析
        out = []
        for art in r.get("PubmedArticle", []):
            pmid = str(art["MedlineCitation"]["PMID"])
            art_info = art["MedlineCitation"]["Article"]
            title = " ".join(art_info.get("ArticleTitle", "")) if isinstance(art_info.get("ArticleTitle", ""), list) else str(art_info.get("ArticleTitle", ""))
            abs_parts = art_info.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join([str(x) for x in abs_parts]) if isinstance(abs_parts, list) else str(abs_parts)
            text = " ".join([title, abstract]).strip()
            if text:
                text = html.unescape(text).replace("\n", " ").strip()
                out.append({"id": f"pmid_{pmid}", "text": text})
        return out
    except Exception as e:
        print(f"  ↪️ Entrez XML 失败：{e}")
        return []

def _efetch_http_xml(id_list: List[str]) -> List[dict]:
    """备用1：HTTP 直连 XML + 手动解析"""
    try:
        params = dict(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)
        data = urlopen(Request(url, headers=UA_HDR), timeout=40).read()
        return list(_parse_pubmed_xml(data))
    except Exception as e:
        print(f"  ↪️ HTTP XML 失败：{e}")
        return []

def _efetch_http_text(id_list: List[str]) -> List[dict]:
    """备用2：HTTP 直连 TEXT（MEDLINE 文本），用粗略规则抽题目与摘要"""
    try:
        params = dict(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)
        raw = urlopen(Request(url, headers=UA_HDR), timeout=40).read().decode("utf-8", errors="ignore")
        # 简单按 PMID 分块
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        out = []
        for blk in blocks:
            lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
            pmid, title, abstract = None, "", ""
            for ln in lines:
                if ln.startswith("PMID-"):
                    pmid = ln.split("PMID-")[-1].strip()
                elif ln.startswith("TI  -") or ln.startswith("TI -"):
                    title = ln.split("TI", 1)[-1].split("-", 1)[-1].strip()
                elif ln.startswith("AB  -") or ln.startswith("AB -"):
                    abstract += " " + ln.split("AB", 1)[-1].split("-", 1)[-1].strip()
            txt = " ".join([title, abstract]).strip()
            if pmid and txt:
                out.append({"id": f"pmid_{pmid}", "text": txt})
        return out
    except Exception as e:
        print(f"  ↪️ HTTP TEXT 失败：{e}")
        return []

def fetch_abstracts(pmids: List[str]) -> Iterable[dict]:
    BATCH = 20  # 更小批次，提高成功率
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i+BATCH]
        print(f"… 抓取批次 {i//BATCH+1} / {((len(pmids)-1)//BATCH)+1} （{len(chunk)} 篇）")
        # 1) Entrez XML
        out = _efetch_entrez_xml(chunk)
        if not out:
            # 2) HTTP XML
            out = _efetch_http_xml(chunk)
        if not out:
            # 3) HTTP TEXT
            out = _efetch_http_text(chunk)
        if not out:
            print(f"⛔ 三种方式都失败，跳过这一批（示例 IDs: {chunk[:3]}…）")
        else:
            for item in out:
                yield item
        time.sleep(0.4)  # 轻微限速，遵守礼仪

# ============== 主流程 =================
def main():
    retmax = int(sys.argv[1]) if len(sys.argv) >= 2 else 300

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    os.chdir(root)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "docs.jsonl")

    print(f"🔎 搜索 PubMed：{QUERY}")
    pmids = search_pmids(QUERY, retmax=retmax)
    print(f"✅ 命中 PMIDs：{len(pmids)}")
    if not pmids:
        print("❌ 一个 PMID 都没拿到，退出。")
        sys.exit(1)

    print("⏬ 开始抓取摘要（多通道兜底）…")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in fetch_abstracts(pmids):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ 完成：写入 {n} 条到 {out_path}")
    if n == 0:
        print("⚠️ 没抓到摘要，可能网络被墙/限流。可稍后重试或换关键词。")
    else:
        print("   现在执行：python src\\build_index.py  重建索引")

if __name__ == "__main__":
    main()
