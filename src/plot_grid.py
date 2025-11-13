# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt

rows = []
with open('runs/compare_grid.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

Ks = sorted({int(r['K']) for r in rows})
def pick(method, K, field):
    for r in rows:
        if r['method'] == method and int(r['K']) == K:
            return float(r[field])
    return 0.0

f1_bm = [pick('BM25',   K, 'F1') for K in Ks]
f1_pp = [pick('KG+PPR', K, 'F1') for K in Ks]
r_bm  = [pick('BM25',   K, 'Recall') for K in Ks]
r_pp  = [pick('KG+PPR', K, 'Recall') for K in Ks]
p_bm  = [pick('BM25',   K, 'Precision') for K in Ks]
p_pp  = [pick('KG+PPR', K, 'Precision') for K in Ks]

# F1
plt.figure()
plt.plot(Ks, f1_bm, marker='o', label='BM25 F1')
plt.plot(Ks, f1_pp, marker='o', label='KG+PPR F1')
plt.xlabel('Top-K'); plt.ylabel('F1'); plt.title('F1 vs K'); plt.legend(); plt.grid(True)
plt.savefig('runs/curve_f1.png', dpi=200, bbox_inches='tight')

# Recall
plt.figure()
plt.plot(Ks, r_bm, marker='o', label='BM25 Recall')
plt.plot(Ks, r_pp, marker='o', label='KG+PPR Recall')
plt.xlabel('Top-K'); plt.ylabel('Recall'); plt.title('Recall vs K'); plt.legend(); plt.grid(True)
plt.savefig('runs/curve_recall.png', dpi=200, bbox_inches='tight')

# Precision
plt.figure()
plt.plot(Ks, p_bm, marker='o', label='BM25 Precision')
plt.plot(Ks, p_pp, marker='o', label='KG+PPR Precision')
plt.xlabel('Top-K'); plt.ylabel('Precision'); plt.title('Precision vs K'); plt.legend(); plt.grid(True)
plt.savefig('runs/curve_precision.png', dpi=200, bbox_inches='tight')

print('✅ 图已保存到 runs\\curve_f1.png / curve_recall.png / curve_precision.png')
