#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142"]
# ///
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pymorphy3

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '02-marking' / 'out'
OUT = STAGE / 'out'

MARKER = re.compile(r'⟦(?:TABLE|MATH)_\d+⟧')
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
CITATION = r'\[\d+(?:\s*[,–-]\s*\d+)*(?:\s*и\s*др\.)?\]'
NOISE = [
    (re.compile(r'</?(?:small|span|div)\b[^>]*>'), ''),
    (re.compile(r'!\[[^\]]*\]\([^)]*\)'), ''),
    (re.compile(rf'(?:,?\s*например,?)?\s+в\s+{CITATION}'), ''),
    (re.compile(rf' ?{CITATION}'), ''),
    (re.compile(r' ?\((?:см\.\s*)?(?:рис|табл)\.\s*[\d.,\s–-]+(?:[а-яa-z](?:,\s*[а-яa-z])*)?\)'), ''),
    (re.compile(r'((?:соотношени|уравнени|выражени|формул)[а-яё]*(?: [А-ЯЁ][а-яё]+)?) \(\d{1,2}\)'), r'\1'),
    (re.compile(r' ?\((?:см\.|например,?|и др\.)?\s*\)'), ''),
    (re.compile(r'^Таблица\s*\d+(?:\.\d+)?\.?[ \t]*', re.M), ''),
]
CAPTION = re.compile(r'Рис\.\s*\d+(?:\.\d+)?\.|Пояснения см\. в тексте')
LEGEND = re.compile(r'(?:[^\n.;]{0,60}:\s*)?(?:[а-яa-zIVX\d]{1,3}(?:\s*[,–-]\s*\d+)?\s*[—–-]\s|\(?×\s?\d)')
FORMULA_BLOCK = re.compile(r'⟦MATH_\d+⟧')
RUNNING_HEADER = re.compile(r'^(?:#+\s*)?(Глава \d+\.\s[^\n]+?)\s*$', re.M)
HYPHEN_BREAK = re.compile(r'([А-Яа-яЁё]+)-\s+(?!(?:и|или|либо)\b)([А-Яа-яЁё]+)')
INNER_HYPHEN = re.compile(r'(?<![\w-])([А-Яа-яЁё]+)-([а-яё]+)(?![\w-])')
MIXED_SCRIPT = re.compile(r'\b(?=\w*[А-Яа-яЁё])(?=\w*[A-Za-z])\w+\b')
LOOKALIKES = 'АВСЕНКМОРТХаесорху', 'ABCEHKMOPTXaecopxy'
TO_LATIN, TO_CYRILLIC = str.maketrans(LOOKALIKES[0], LOOKALIKES[1]), str.maketrans(LOOKALIKES[1], LOOKALIKES[0])
FORMULA = re.compile(r'(\$[^$]+\$)')
GREEK_TO_CYRILLIC = str.maketrans('ΑΒΓΕΗΚΜΝΟΠΡΤΦΧ', 'АВГЕНКМНОПРТФХ')
LEGEND_LINE = re.compile(r'(?:\d+(?:\s*[,–-]\s*\d+)*\s*—\s+[^;\n]{1,80}(?:;\s*|$)){2,}')

morph = pymorphy3.MorphAnalyzer()


def known(word):
    return morph.word_is_known(word.lower())


def unify_script(match):
    token = match.group(0)
    if known(token.translate(TO_CYRILLIC)):
        return token.translate(TO_CYRILLIC)
    only_lookalikes = all(letter in LOOKALIKES[0] for letter in re.findall(r'[А-Яа-яЁё]', token))
    unified = token.translate(TO_LATIN if only_lookalikes else TO_CYRILLIC)
    return token if MIXED_SCRIPT.fullmatch(unified) else unified


def join_inner_hyphen(match):
    left, right = match.groups()
    return left + right if known(left + right) and not (known(left) and known(right)) else match.group(0)


def book_context(text):
    titles = set(RUNNING_HEADER.findall(text))
    titles = {title for title in titles if not any(title != other and title.startswith(other) for other in titles)}
    vocabulary = Counter(re.findall(r'[а-яё]+(?:-[а-яё]+)*', text.lower()))
    return {'titles': sorted(titles, key=len, reverse=True), 'vocabulary': vocabulary}


def word_repairs(context):
    vocabulary = context['vocabulary']

    def join_hyphen_break(match):
        left, right = match.groups()
        joined, hyphenated = left + right, f'{left}-{right}'
        if known(joined) or vocabulary[joined.lower()] > vocabulary[hyphenated.lower()]:
            return joined
        return hyphenated if vocabulary[hyphenated.lower()] or (known(left) and known(right)) else joined

    return [
        (HYPHEN_BREAK, join_hyphen_break),
        (INNER_HYPHEN, join_inner_hyphen),
        (re.compile(r'\\([~*_#])'), r'\1'),
        (re.compile(r'(?<![\w\\])(?:H\s?/\s?(?:mm|MM|мм)|Н\s?/\s?(?:mm|MM))(?![A-Za-zА-Яа-яЁё])'), 'Н/мм'),
        (re.compile(r'\b\d{2}[0-9A-ZА-ЯЁΑ-Ω]*[Α-Ω][0-9A-ZА-ЯЁΑ-Ω]*\b'),
         lambda match: match.group(0).translate(GREEK_TO_CYRILLIC).translate(TO_CYRILLIC)),
        (MIXED_SCRIPT, unify_script),
        (re.compile(r'\bСтЗ(?=[А-Яа-яЁё]{0,3}\b)'), 'Ст3'),
        (re.compile(r'(?<!\w)(Мп|Мо)(?!\w)'), lambda match: {'Мп': 'Mn', 'Мо': 'Mo'}[match.group(0)]),
        (re.compile(r'(?<!\w)(НИС|ДWTT)(?!\w)'), lambda match: {'НИС': 'HIC', 'ДWTT': 'DWTT'}[match.group(0)]),
    ]


def substitute(text, rules, removed=None):
    for pattern, replacement in rules:
        def replace(match):
            result = match.expand(replacement) if isinstance(replacement, str) else replacement(match)
            if removed is not None:
                removed.append(match.group(0).replace(result, '', 1))
            return result
        text = pattern.sub(replace, text)
    return text


def drop_legend_lines(text, removed):
    lines = text.split('\n')
    removed += [line for line in lines if LEGEND_LINE.fullmatch(line.strip())]
    return '\n'.join(line for line in lines if not LEGEND_LINE.fullmatch(line.strip()))


def drop_figures(paragraphs, removed):
    kept, after_caption = [], False
    for paragraph in paragraphs:
        if CAPTION.match(paragraph):
            if kept and FORMULA_BLOCK.fullmatch(kept[-1]):
                removed.append(kept.pop())
            removed.append(paragraph)
            after_caption = True
        elif after_caption and LEGEND.match(paragraph):
            removed.append(paragraph)
        else:
            after_caption = after_caption and bool(MARKER.fullmatch(paragraph))
            kept.append(paragraph)
    return kept


def drop_repeated_headings(paragraphs, removed):
    seen, kept = set(), []
    for paragraph in paragraphs:
        if paragraph.startswith('#') and paragraph in seen:
            removed.append(paragraph)
            continue
        seen.add(paragraph)
        kept.append(paragraph)
    return kept


def join_broken_paragraphs(paragraphs):
    joined = []
    for paragraph in paragraphs:
        previous = joined[-1] if joined else ''
        if previous and not previous.startswith('#') and previous[-1] not in '.!?:;' and re.match(r'[а-яё]', paragraph):
            joined[-1] = f'{previous} {paragraph}'
        else:
            joined.append(paragraph)
    return joined


def clean(text, context, removed):
    text = substitute(text, NOISE, removed)
    if context['titles']:
        header = re.compile(r'^(?:#+\s*)?(?:' + '|'.join(map(re.escape, context['titles'])) + r')(?:[ \t]*\n|[ \t]+|$)', re.M)
        text = substitute(text, [(header, '')], removed)
    paragraphs = [paragraph.strip() for paragraph in drop_legend_lines(text, removed).split('\n\n') if paragraph.strip()]
    paragraphs = join_broken_paragraphs(drop_repeated_headings(drop_figures(paragraphs, removed), removed))
    text = substitute('\n\n'.join(paragraphs), word_repairs(context))
    return re.sub(r' *\n *', '\n', re.sub(r'[ \t]+', ' ', text)).strip() + '\n'


def clean_tables(markers, context):
    def clean_cells(table):
        return ''.join(part if FORMULA.fullmatch(part) else substitute(part, word_repairs(context))
                       for part in FORMULA.split(table))
    return {marker: clean_cells(original) if marker.startswith('⟦TABLE') else original for marker, original in markers.items()}


def protected(text):
    return Counter(MARKER.findall(text)) + Counter(NUMBER.findall(MARKER.sub(' ', text)))


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    for source in sorted(SOURCE.glob('*.txt')):
        text = source.read_text(encoding='utf-8')
        markers = json.loads((SOURCE / f'{source.stem}.markers.json').read_text(encoding='utf-8'))
        context, removed = book_context(text), []
        cleaned = clean(text, context, removed)
        lost = protected(text) - protected(cleaned) - protected('\n'.join(removed))
        if lost:
            raise SystemExit(f'{source.stem}: numbers or markers lost outside removed noise: {dict(lost.most_common(10))}')
        (OUT / source.name).write_text(cleaned, encoding='utf-8')
        write_json(OUT / f'{source.stem}.markers.json', clean_tables(markers, context), indent=1)
        print(f'{source.stem:<9} {len(text)} → {len(cleaned)} characters, {len(removed)} noise fragments removed')


if __name__ == '__main__':
    main()
