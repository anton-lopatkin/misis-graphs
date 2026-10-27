#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["razdel==0.5.0", "pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142"]
# ///
import json
import re
from collections import Counter

import pymorphy3
from razdel import tokenize

from run import OUT, PROTECTED, PROTECTED_KINDS, SOURCE, STAGE, TABLE
from stage import check, share, write_json

morph = pymorphy3.MorphAnalyzer()
MAX_SPLIT_ABBREVIATIONS = 0
ABBREVIATION = re.compile(r'(?<![\w-])[A-ZА-ЯЁ]{2,6}(?![\w-])')
LECTURE_METRICS = {'terms': ('steel_grade', 'designation', 'phase', 'compound'), 'formulas': ('formula',),
                   'numbers_with_units': ('number_with_unit',), 'numbers': ('number',)}


def protected_items(text):
    return [(match.group(0), match.lastgroup, match.start(), match.end()) for match in PROTECTED.finditer(text)]


def main():
    metrics, vocabulary = {}, Counter()
    for source in sorted(SOURCE.glob('*.chunks.json')):
        book = source.name.removesuffix('.chunks.json')
        texts = [piece['text'] for piece in json.loads(source.read_text(encoding='utf-8'))]
        chunks = [json.loads(line) for line in (OUT / f'{book}.jsonl').read_text(encoding='utf-8').splitlines()]
        tokens = [token for chunk in chunks for sentence in chunk['sentences'] for token in sentence]
        tokens += [token for chunk in chunks for table in chunk['tables'] for row in table for cell in row for token in cell]
        vocabulary.update(token['text'] for token in tokens)
        found, split_without_protection, abbreviations_found, abbreviations_split = Counter(), Counter(), 0, 0
        for text, chunk in zip(texts, chunks):
            chunk_tokens = {token['text'] for sentence in chunk['sentences'] for token in sentence}
            chunk_tokens |= {token['text'] for table in chunk['tables'] for row in table for cell in row for token in cell}
            prose_and_cells = TABLE.sub(lambda match: match.group(0).replace('|', '\n'), text)
            boundaries = {token.start for token in tokenize(prose_and_cells)}
            abbreviations = ABBREVIATION.findall(PROTECTED.sub(' ', prose_and_cells))
            abbreviations_found += len(abbreviations)
            abbreviations_split += sum(not any(abbreviation in token for token in chunk_tokens) for abbreviation in abbreviations)
            for item, kind, start, end in protected_items(prose_and_cells):
                found[kind] += 1
                split_without_protection[kind] += any(start < boundary < end for boundary in boundaries)
        whole = Counter(token['kind'] for token in tokens)
        words = [token['text'] for token in tokens if token['kind'] == 'word']
        metrics[book] = {
            'chunks': len(chunks),
            'sentences': sum(len(chunk['sentences']) for chunk in chunks),
            'tokens': len(tokens),
            'protected': {kind: {'found': found[kind], 'whole_tokens': whole[kind], 'split_without_protection': split_without_protection[kind]}
                          for kind in PROTECTED_KINDS},
            'out_of_vocabulary': share(sum(not morph.word_is_known(word) for word in words), len(words), 3),
            'abbreviations': {'found': abbreviations_found, 'split': abbreviations_split},
        }
    abbreviations = json.loads((OUT / 'abbreviations.json').read_text(encoding='utf-8'))
    metrics['vocabulary'] = {'tokens': len(vocabulary), 'abbreviations': len(abbreviations)}
    for book in (key for key in metrics if key != 'vocabulary'):
        values = metrics[book]
        print(f"{book:<9} {values['chunks']} chunks, {values['sentences']} sentences, {values['tokens']} tokens, "
              f"OOV {values['out_of_vocabulary']['share']:.1%}")
        for metric, kinds in LECTURE_METRICS.items():
            for kind in kinds:
                counts = values['protected'][kind]
                print(f"  {metric:<19} {kind:<17} {counts['whole_tokens']}/{counts['found']} whole, razdel alone would split {counts['split_without_protection']}")
        print(f"  {'abbreviations':<19} {values['abbreviations']['found'] - values['abbreviations']['split']}/{values['abbreviations']['found']} whole")
    print(f"vocabulary {metrics['vocabulary']['tokens']} distinct tokens, {metrics['vocabulary']['abbreviations']} abbreviations")
    checks = [check(f'{book}.{kind}.whole', counts['whole_tokens'], '==', counts['found'])
              for book, values in metrics.items() if book != 'vocabulary' for kind, counts in values['protected'].items()]
    checks += [check(f'{book}.abbreviations.split', values['abbreviations']['split'], '<=', MAX_SPLIT_ABBREVIATIONS)
               for book, values in metrics.items() if book != 'vocabulary']
    write_json(STAGE / 'metrics.json', metrics | {'checks': checks})


if __name__ == '__main__':
    main()
