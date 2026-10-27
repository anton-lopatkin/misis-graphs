#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==2.2.3", "pyarrow==25.0.1"]
# ///
import json
from collections import Counter
from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent
RUNS = ('raw', 'clean-default-chunks', 'clean')
ALPHABETS = ('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def first_letter(title):
    head = title.strip()[:1].upper()
    if any(head in alphabet for alphabet in ALPHABETS):
        return head
    return 'цифра' if head.isdigit() else 'другое'


def counts(run):
    output = STAGE / run / 'output'
    entities = pd.read_parquet(output / 'entities.parquet')
    relationships = pd.read_parquet(output / 'relationships.parquet')
    vertices = Counter(first_letter(title) for title in entities['title'])
    edges = Counter()
    for source, target in zip(relationships['source'], relationships['target']):
        for letter in {first_letter(source), first_letter(target)}:
            edges[letter] += 1
    return vertices, edges


def main():
    tables = {run: counts(run) for run in RUNS}
    letters = [*''.join(ALPHABETS), 'цифра', 'другое']
    rows = []
    for letter in letters:
        row = {'letter': letter}
        for run in RUNS:
            vertices, edges = tables[run]
            row[f'vertices_{run}'] = vertices[letter]
            row[f'edges_{run}'] = edges[letter]
        if any(value for key, value in row.items() if key != 'letter'):
            rows.append(row)
    (STAGE / 'letters.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    groups = {'русские': ALPHABETS[0], 'латиница': ALPHABETS[1]}
    totals = [{'letter': name, **{key: sum(row[key] for row in rows if row['letter'] in members) for key in rows[0] if key != 'letter'}}
              for name, members in groups.items()]
    width = max(map(len, RUNS)) + 2
    print(f'{"":<9}{"вершин":^{width * len(RUNS)}}{"рёбер":^{width * len(RUNS)}}')
    print(f'{"буква":<9}' + ''.join(f'{run:>{width}}' for run in RUNS) * 2)
    for row in [*rows, *totals]:
        print(f'{row["letter"]:<9}' + ''.join(f'{row[f"vertices_{run}"]:>{width}}' for run in RUNS)
              + ''.join(f'{row[f"edges_{run}"]:>{width}}' for run in RUNS))


if __name__ == '__main__':
    main()
