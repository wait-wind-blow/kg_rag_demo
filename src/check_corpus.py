# src/check_corpus.py
import os, json, re

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path = os.path.join(root, "data", "docs.jsonl")

staph_pat = re.compile(r'(?i)staphylococcus|staphylococcal|s\. aureus|mrsa|mssa')
drug_pat  = re.compile(r'(?i)vancomycin|linezolid|daptomycin|teicoplanin|ceftaroline|clindamycin|doxycycline|trimethoprim(-|/)?sulfamethoxazole|tmp-smx|co-?trimoxazole|cotrimoxazole|nafcillin|oxacillin|dicloxacillin|flucloxacillin|cefazolin|cephalexin|gentamicin|rifampin|rifampicin|tetracycline')

tot=nonempty=staph=drug=0
lengths=[]; examples=[]

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        tot += 1
        try:
            o = json.loads(line)
        except Exception:
            continue
        txt = (o.get("text") or "").strip()
        pid = o.get("pid", "?")
        if txt:
            nonempty += 1
            L = len(txt); lengths.append(L)
            hit_s = bool(staph_pat.search(txt))
            hit_d = bool(drug_pat.search(txt))
            if hit_s: staph += 1
            if hit_d: drug += 1
            if hit_s and len(examples) < 5:
                examples.append((pid, txt[:220].replace("\n"," ")))

print(f"文件: {path}")
print(f"总条数: {tot}")
print(f"有正文的条数: {nonempty}")
print(f"含 Staph 关键词的条数: {staph}")
print(f"含药名的条数: {drug}")
if lengths:
    lens_sorted = sorted(lengths)
    med = lens_sorted[len(lens_sorted)//2]
    print(f"正文长度：平均 {sum(lengths)//len(lengths)} 字符，中位 {med}，最大 {max(lengths)}")
print("\n示例（最多5条，命中 Staph）：")
for pid, snippet in examples:
    print(f"- {pid}: {snippet}...")
