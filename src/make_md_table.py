# -*- coding: utf-8 -*-
import csv

in_csv = 'runs/compare_grid.csv'
out_md = 'runs/compare_grid.md'

rows = list(csv.DictReader(open(in_csv, encoding='utf-8')))
lines = []
lines.append('| Method | K | Precision | Recall | F1 |')
lines.append('|---|---:|---:|---:|---:|')
for r in rows:
    lines.append(f"| {r['method']} | {r['K']} | {float(r['Precision']):.3f} | {float(r['Recall']):.3f} | {float(r['F1']):.3f} |")

open(out_md, 'w', encoding='utf-8').write('\n'.join(lines))
print('✅ 已写出', out_md)
