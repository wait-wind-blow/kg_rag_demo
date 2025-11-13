# -*- coding: utf-8 -*-
# 比较 BM25 vs KG+PPR 的药名召回：Precision / Recall / F1
import re

# —— 统一同义词/大小写/连字符 —— #
ALIASES = {
    "vancomycin": ["vancomycin","vanco"],
    "linezolid": ["linezolid","zyvox"],
    "daptomycin": ["daptomycin","cubicin"],
    "teicoplanin": ["teicoplanin"],
    "ceftaroline": ["ceftaroline","teflaro","tazef"],
    "clindamycin": ["clindamycin"],
    "doxycycline": ["doxycycline"],
    "tetracycline": ["tetracycline","minocycline","minocyclin"],
    "trimethoprim-sulfamethoxazole": [
        "trimethoprim-sulfamethoxazole","trimethoprim sulfamethoxazole",
        "tmp-smx","tmp smx","co-trimoxazole","cotrimoxazole","bactrim","septra"
    ],
    "oxacillin": ["oxacillin"],
    "nafcillin": ["nafcillin"],
    "dicloxacillin": ["dicloxacillin"],
    "flucloxacillin": ["flucloxacillin"],
    "cefazolin": ["cefazolin"],
    "cephalexin": ["cephalexin","keflex"],
    "gentamicin": ["gentamicin"],
    "rifampin": ["rifampin","rifampicin"],
    # 可按需扩展： "levonadifloxacin": ["levonadifloxacin"],
}

# —— 论文里常见“金葡菌/MRSA 用药黄金列表（评测 Gold）” —— #
GOLD = set([
    "vancomycin","linezolid","daptomycin","teicoplanin","ceftaroline",
    "clindamycin","doxycycline","tetracycline","trimethoprim-sulfamethoxazole",
    "oxacillin","nafcillin","dicloxacillin","flucloxacillin",
    "cefazolin","cephalexin","gentamicin","rifampin"
])

def canonize(name:str)->str:
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # 映射到规范名
    for canon, alist in ALIASES.items():
        for a in alist:
            if s == a.lower():
                return canon
    return s

def normset(lst):
    return set(canonize(x) for x in lst)

def prf1(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp+fp) if (tp+fp) else 0.0
    rec  = tp / (tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return tp, fp, fn, prec, rec, f1

def main():
    # === 把下面两行替换成你刚才跑出来的结果（无需顺序一致） ===
    bm25_pred = [
        "vancomycin","linezolid","daptomycin","teicoplanin","ceftaroline",
        "clindamycin","doxycycline","tetracycline","trimethoprim-sulfamethoxazole",
        "oxacillin","nafcillin","cefazolin","cephalexin","gentamicin","rifampin"
    ]
    ppr_pred = [
        "vancomycin","linezolid","clindamycin","trimethoprim-sulfamethoxazole",
        "tetracycline","daptomycin","gentamicin","rifampin","ceftaroline",
        "teicoplanin","oxacillin","doxycycline","cefazolin","flucloxacillin",
        "cephalexin","nafcillin","dicloxacillin"
    ]
    bm25 = normset(bm25_pred)
    ppr  = normset(ppr_pred)
    gold = set(GOLD)

    print("=== GOLD 大小 ===", len(gold), sorted(gold))
    print("\n=== BM25 ===")
    tp, fp, fn, p, r, f1 = prf1(bm25, gold)
    print(f"TP={tp} FP={fp} FN={fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("BM25覆盖：", sorted(bm25))
    print("\n=== KG+PPR ===")
    tp, fp, fn, p, r, f1 = prf1(ppr, gold)
    print(f"TP={tp} FP={fp} FN={fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print("PPR覆盖：", sorted(ppr))
    print("\n=== 差异 ===")
    print("PPR 额外找到：", sorted(ppr - bm25))
    print("BM25 额外找到：", sorted(bm25 - ppr))

if __name__ == "__main__":
    main()
