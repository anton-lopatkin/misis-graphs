#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["jiwer==3.1.0", "pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142"]
# ///
import difflib
import json
import re

import jiwer

from run import MARKER, NUMBER, OUT, SOURCE, STAGE, book_context, clean, morph
from stage import check, share, write_json

IMAGE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
MIN_WORD_LETTERS = 4
MARKUP = re.compile(r'!\[[^\]]*\]\([^)]*\)|</?\w+[^<>\n]*>')
HYPHEN_BREAK = re.compile(r'([А-Яа-яЁё]+)[-­]\s+([а-яё]+)')
WORD = re.compile(rf'[А-Яа-яЁё]{{{MIN_WORD_LETTERS},}}')
MIN_QUALITY = 0.75
NUMBER_WITH_UNIT = re.compile(r'-?\d+(?:,\d+)?(?:[–-]-?\d+(?:,\d+)?)? ?(?:°C/с|°C|°С|К/с|МПа|ГПа|кН|Н/мм²|Н/мм|H/mm²|H/mm|кгс|Дж/см²|МДж/м²|мкм|мм|см|нм|мин|%)(?![\w/])')


def join_hyphen_break(match):
    joined = match[1] + match[2]
    return joined if morph.word_is_known(joined.lower()) else match[0]


def unknown_word_share(text):
    words = [word.lower() for word in WORD.findall(HYPHEN_BREAK.sub(join_hyphen_break, MARKUP.sub(' ', text)))]
    return share(sum(not morph.word_is_known(word) for word in words), len(words))


def fragment_source(paragraphs, start, end):
    return '\n\n'.join(paragraph for paragraph in paragraphs[start:end] if not IMAGE.fullmatch(paragraph.strip()))


def edits(source, target):
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    return {(i1, i2, tuple(target[j1:j2])) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != 'equal'}


def against_gold(fragments, books):
    sources, outputs, golds = [], [], []
    made = needed = right = worse = 0
    for fragment in fragments:
        text, context = books[fragment['book']]
        source = fragment_source(text.split('\n\n'), *fragment['paragraphs'])
        source, output, gold = (' '.join(part.split()) for part in (source, clean(source, context, []), fragment['gold']))
        made_edits, needed_edits = edits(source.split(), output.split()), edits(source.split(), gold.split())
        made, needed, right = made + len(made_edits), needed + len(needed_edits), right + len(made_edits & needed_edits)
        worse += jiwer.cer(gold, output) > jiwer.cer(gold, source)
        sources.append(source), outputs.append(output), golds.append(gold)
    precision, recall = right / made, right / needed
    before, after = jiwer.cer(golds, sources), jiwer.cer(golds, outputs)
    return {
        'fragments': len(fragments),
        'cer': {'before': round(before, 4), 'after': round(after, 4), 'reduction': round(1 - after / before, 3)},
        'wer': {'before': round(jiwer.wer(golds, sources), 4), 'after': round(jiwer.wer(golds, outputs), 4)},
        'fragments_worse': worse,
        'edits': {'made': made, 'needed': needed, 'right': right},
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f05': round(1.25 * precision * recall / (0.25 * precision + recall), 3),
    }


def preservation(text, context):
    removed = []
    cleaned = clean(text, context, removed)
    noise = '\n'.join(removed)
    count = {
        'numbers': lambda part: len(NUMBER.findall(MARKER.sub(' ', part))),
        'numbers_with_units': lambda part: len(NUMBER_WITH_UNIT.findall(MARKER.sub(' ', part))),
        'formulas': lambda part: len(re.findall(r'⟦MATH_\d+⟧', part)),
        'tables': lambda part: len(re.findall(r'⟦TABLE_\d+⟧', part)),
    }
    return {kind: {'before': counter(text), 'after': counter(cleaned), 'removed_with_noise': counter(noise)}
            for kind, counter in count.items()} | {'images_removed': len(IMAGE.findall(text))}


def main():
    books = {}
    for source in sorted(SOURCE.glob('*.txt')):
        text = source.read_text(encoding='utf-8')
        books[source.stem] = text, book_context(text)
    metrics = {'gold': against_gold(json.loads((STAGE / 'gold.json').read_text(encoding='utf-8')), books)}
    for book, (text, context) in books.items():
        metrics[book] = {
            'preservation': preservation(text, context),
            'unknown_word_share': {'before': unknown_word_share(text)['share'],
                                   'after': unknown_word_share((OUT / f'{book}.txt').read_text(encoding='utf-8'))['share']},
        }
    gold = metrics['gold']
    print(f"gold      CER {gold['cer']['before']:.2%} → {gold['cer']['after']:.2%} (−{gold['cer']['reduction']:.0%}), "
          f"WER {gold['wer']['before']:.2%} → {gold['wer']['after']:.2%}, fragments worse {gold['fragments_worse']}")
    print(f"          P {gold['precision']:.1%}, R {gold['recall']:.1%}, F0.5 {gold['f05']:.1%} "
          f"({gold['edits']['right']} right of {gold['edits']['made']} made, {gold['edits']['needed']} needed)")
    for book in books:
        kept = metrics[book]['preservation']
        print(f"{book:<9} " + ', '.join(f"{kind} {kept[kind]['before']} → {kept[kind]['after']} (+{kept[kind]['removed_with_noise']} with noise)"
                                         for kind in ('numbers', 'numbers_with_units', 'formulas', 'tables'))
              + f", images removed {kept['images_removed']}, unknown words {metrics[book]['unknown_word_share']['before']:.2%} → "
              f"{metrics[book]['unknown_word_share']['after']:.2%}")
    checks = [check(f'gold.{name}', gold[name], '>=', MIN_QUALITY) for name in ('precision', 'recall', 'f05')]
    write_json(STAGE / 'metrics.json', metrics | {'checks': checks})


if __name__ == '__main__':
    main()
