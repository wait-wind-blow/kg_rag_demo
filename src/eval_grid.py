# -*- coding: utf-8 -*-
# 批量扫不同K，比较 BM25 vs KG+PPR 的 P/R/F1，并输出CSV（Windows 编码安全版）
import os, csv, subprocess, locale, re

# —— Gold 集（固定 17 项）——
GOLD = {
    "vancomycin","linezolid","daptomycin","teicoplanin","ceftaroline",
    "clindamycin","doxycycline","tetracycline","trimethoprim-sulfamethoxazole",
    "oxacillin","nafcillin","dicloxacillin","flucloxacillin",
    "cefazolin","cephalexin","gentamicin","rifampin"
}

QUESTION = "Which antibiotics are commonly used to treat Staphylococcus aureus (including MRSA) infections?"
K_LIST = [5, 10, 15, 20, 30, 50]

def safe_decode(b: bytes) -> str:
    """优先 UTF-8；失败则用 Windows 本地编码（mbcs/系统首选编码），忽略非法字节"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("mbcs", errors="ignore")
        except LookupError:
            return b.decode(locale.getpreferredencoding(False), errors="ignore")

def run_cmd(cmd: str) -> str:
    """不要用 text=True/encoding，直接拿字节再自己解码，避免 UnicodeDecodeError"""
    cp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_bytes = cp.stdout or b""
    return safe_decode(out_bytes)

def parse_answer_list(stdout: str):
    """
    在输出里定位
      'ANSWER (drug list)'
    下一行的逗号分隔药名；如果出现多次，取最后一段。
    """
    lines = stdout.splitlines()
    hit_idx = -1
    for i, line in enumerate(lines):
        if "ANSWER (drug list)" in line:
            hit_idx = i
    if hit_idx == -1:
        return []
    if hit_idx + 1 >= len(lines):
        return []
    raw = lines[hit_idx + 1].strip()
    if raw.startswith("("):  # 没匹配到药名
        return []
    # 拆分并标准化
    drugs = [re.sub(r"\s+", " ", x.strip().lower()) for x in raw.split(",")]
    return [d for d in drugs if d]

def prf1(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return tp, fp, fn, p, r, f1

def main():
    os.makedirs("runs", exist_ok=True)
    out_csv = "runs/compare_grid.csv"
    rows = []

    for K in K_LIST:
        # —— BM25 —— #
        bm_out = run_cmd(f'python src\\answer_drugs_bm25.py "{QUESTION}" {K}')
        bm_list = parse_answer_list(bm_out)
        bm_set = set(bm_list)
        tp, fp, fn, p, r, f1 = prf1(bm_set, GOLD)
        rows.append(["BM25", K, tp, fp, fn, round(p,3), round(r,3), round(f1,3), ";".join(sorted(bm_set))])

        # —— KG+PPR —— #
        pp_out = run_cmd(f'python src\\answer_drugs.py "{QUESTION}" {K}')
        pp_list = parse_answer_list(pp_out)
        pp_set = set(pp_list)
        tp, fp, fn, p, r, f1 = prf1(pp_set, GOLD)
        rows.append(["KG+PPR", K, tp, fp, fn, round(p,3), round(r,3), round(f1,3), ";".join(sorted(pp_set))])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method","K","TP","FP","FN","Precision","Recall","F1","found"])
        w.writerows(rows)

    print(f"✅ 已写出 {out_csv}")
    for r in rows:
        print(r)

if __name__ == "__main__":
    main()
