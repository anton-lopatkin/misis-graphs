#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["razdel==0.5.0"]
# ///
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from string import ascii_lowercase

from razdel import tokenize

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '05-chunking' / 'out'
OUT = STAGE / 'out'

TABLE = re.compile(r'<table.*?</table>|(?m:^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*$)', re.S)
UNITS = r'(?:°C/с|°C|МПа·м¹ᐟ²|МПа|ГПа|Па|кН|Дж/см²|Дж/м²|МДж/м²|мкм|мм|см|нм|мин|ч|%)'
PROTECTED_KINDS = {
    'formula': r'\$\$.*?\$\$|\$[^$\n]+\$',
    'number_with_unit': rf'[−-]?\d+(?:,\d+)?(?:–[−-]?\d+(?:,\d+)?)? ?{UNITS}(?![\w/])',
    'steel_grade': r'\b\d{1,3}[А-Я][0-9А-Я]*(?:-[А-Я]+)?\b|\b[XК]\d{2,3}\b',
    'compound': r'\b[A-Z][a-z]?\([A-Z](?:, ?[A-Z])+\)',
    'phase': r'\(?[γαδ](?: ?[+→] ?[γαδ])+\)?',
    'designation': r'[^\W\d_][\w′]*(?:_(?:\([^()\s]*\)|[^\W_]+)|[₀-₉⁰-⁹⁻]+)',
    'number': r'(?<![\w,])[−-]?\d+(?:,\d+)?(?:–[−-]?\d+(?:,\d+)?)?(?![\w,])',
}
PROTECTED = re.compile('|'.join(f'(?P<{kind}>{pattern})' for kind, pattern in PROTECTED_KINDS.items()), re.S)
SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|\n+')
NO_SENTENCE_END = re.compile(r'(?:^|\s)(?:[а-яё]{1,3}|[А-ЯЁA-Z])\.$|(?:^|\n)[ \t]*\d{1,2}\.$')
FORMULA = re.compile(PROTECTED_KINDS['formula'], re.S)
ABBREVIATION = re.compile(r'[A-ZА-ЯЁ]{2,6}')


def placeholder(index):
    letters = ''
    while True:
        letters, index = ascii_lowercase[index % 26] + letters, index // 26
        if not index:
            return f'qq{letters}qq'


def kind_of(token):
    if ABBREVIATION.fullmatch(token):
        return 'abbreviation'
    if token.isalpha():
        return 'word'
    return 'numeric' if token[:1].isdigit() else 'symbol'


def sentences(text):
    pieces, start = [], 0
    for match in SENTENCE_END.finditer(text):
        if match.group(0).strip(' ') or not NO_SENTENCE_END.search(text[max(0, match.start() - 8):match.start()]):
            pieces.append(text[start:match.start()])
            start = match.end()
    return [piece for piece in pieces + [text[start:]] if piece.strip()]


def tokenized(text):
    protected = {}

    def protect(match):
        key = placeholder(len(protected))
        protected[key] = (match.group(0), match.lastgroup or 'formula')
        return f' {key} '

    def sentence_tokens(sentence):
        return [{'text': protected[token.text][0], 'kind': protected[token.text][1]} if token.text in protected
                else {'text': token.text, 'kind': kind_of(token.text)} for token in tokenize(PROTECTED.sub(protect, sentence))]
    return [sentence_tokens(sentence) for sentence in sentences(FORMULA.sub(protect, text))]


def table_cells(table):
    rows = [row.strip().strip('|').split('|') for row in table.splitlines() if row.strip() and not re.fullmatch(r'[\s|:-]+', row)]
    return [[[token for sentence in tokenized(cell.strip()) for token in sentence] for cell in row] for row in rows]


def tokenize_chunk(piece):
    tables = [table_cells(match.group(0)) for match in TABLE.finditer(piece['text'])]
    prose = TABLE.sub('\n', piece['text'])
    result = {'id': piece['id'], 'sentences': tokenized(prose), 'tables': tables}
    if re.sub(r'\s', '', prose) != re.sub(r'\s', '', ''.join(token['text'] for sentence in result['sentences'] for token in sentence)):
        raise SystemExit(f"{piece['id']}: tokens do not reproduce the text")
    return result


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    abbreviations = Counter()
    for source in sorted(SOURCE.glob('*.chunks.json')):
        book = source.name.removesuffix('.chunks.json')
        chunks = [tokenize_chunk(piece) for piece in json.loads(source.read_text(encoding='utf-8'))]
        abbreviations.update(token['text'] for chunk in chunks for sentence in chunk['sentences'] for token in sentence if token['kind'] == 'abbreviation')
        (OUT / f'{book}.jsonl').write_text(''.join(json.dumps(chunk, ensure_ascii=False) + '\n' for chunk in chunks), encoding='utf-8')
        print(f'{book:<9} {len(chunks)} chunks, {sum(len(chunk["sentences"]) for chunk in chunks)} sentences')
    write_json(OUT / 'abbreviations.json', dict(abbreviations.most_common()), indent=1)


if __name__ == '__main__':
    main()
