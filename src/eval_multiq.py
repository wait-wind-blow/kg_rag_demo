# -*- coding: utf-8 -*-
import subprocess, csv, locale, re

GOLD = {
    "vancomycin","linezolid","daptomycin","teicoplanin","ceftaroline",
    "clindamycin","doxycycline","tetracycline","trimethoprim-sulfamethoxazole",
    "oxacillin","nafcillin","dicloxacillin","flucloxacillin",
    "cefazolin","cephalexin","gentamicin","rifampin"
}

QUESTIONS = [
    "Which antibiotics are commonly used to treat Staphylococcus aureus (including MRSA) infections?",
    "List common antibiotics for MRSA and MSSA infections.",
    "What drugs are typically used to treat Staphylococcus aureus, including MRSA?",
    "For skin/soft tissue infections due to S. aureus, which antibiotics are commonly prescribed?",
    "Name first-line antibiotics for MSSA vs MRSA."
]
K = 15  # 你可以改成 30 再跑一次

def safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("mbcs", errors="ignore")
        except:
            return b.decode(locale.getpreferredencoding(False), errors="ignore")

def run_cmd(cmd: str) -> str:
    cp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return safe_decode(cp.stdout or b"")

def parse_answer_list(stdout: str):
    lines = stdout.splitlines()
    hit = -1
    for i, line in enumerate(lines):
        if "ANSWER (drug list)" in line:
            hit = i
    if hit == -1 or hit+1 >= len(lines):
        return []
    raw = lines[hit+1].strip()
    if raw.startswith("("):
        return []
    return [re.sub(r"\s+"," ",x.strip().lower()) for x in raw.split(",") if x.strip()]

def prf1(pred, gold):
    pred = set(pred); gold = set(gold)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1

def one_method(name, cmd):
    out = run_cmd(cmd)
    lst = parse_answer_list(out)
    p,r,f1 = prf1(lst, GOLD)
    return p,r,f1,lst

def main():
    sums = {'BM25':[0,0,0], 'KG+PPR':[0,0,0]}
    n = len(QUESTIONS)
    for q in QUESTIONS:
        p_bm, r_bm, f1_bm, lst_bm = one_method('BM25',   f'python src\\answer_drugs_bm25.py "{q}" {K}')
        p_pp, r_pp, f1_pp, lst_pp = one_method('KG+PPR', f'python src\\answer_drugs.py "{q}" {K}')
        print(f"\nQ: {q}\n  [BM25]   P={p_bm:.3f} R={r_bm:.3f} F1={f1_bm:.3f}")
        print(f"  [KG+PPR] P={p_pp:.3f} R={r_pp:.3f} F1={f1_pp:.3f}")
        sums['BM25'][0]+=p_bm; sums['BM25'][1]+=r_bm; sums['BM25'][2]+=f1_bm
        sums['KG+PPR'][0]+=p_pp; sums['KG+PPR'][1]+=r_pp; sums['KG+PPR'][2]+=f1_pp

    print("\n=== Macro Average ===")
    for m in ['BM25','KG+PPR']:
        p = sums[m][0]/n; r = sums[m][1]/n; f1 = sums[m][2]/n
        print(f"{m}: P={p:.3f} R={r:.3f} F1={f1:.3f} (K={K}, n={n})")

if __name__ == "__main__":
    main()
