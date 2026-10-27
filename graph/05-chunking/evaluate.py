#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken==0.12.0", "langchain-text-splitters==0.3.8"]
# ///
import json
import re
import statistics
from bisect import bisect_right
from itertools import accumulate

from run import MAX_CHUNK_TOKENS, OUT, SOURCE, STAGE, encoding, source_text, tokens
from stage import check, share, write_json

RAW_INPUT = STAGE.parent / '01-extraction' / 'out'
GRAPHRAG_CHUNK_TOKENS = 1200
GRAPHRAG_OVERLAP_TOKENS = 100
MIN_CHUNK_TOKENS = 512
MAX_TOKENS_STD = 200
MIN_OVERLAP_SHARE = 0.10
MAX_OVERLAP_SHARE = 0.20
MAX_BROKEN_SENTENCES = 0.05
MAX_BROKEN_PARAGRAPHS = 0.02
PROTECTED = re.compile(r'<table.*?</table>|^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*$|\$\$.*?\$\$|\$[^$\n]+\$', re.S | re.M)
PARAGRAPH_BREAK = re.compile(r'\n+')
SENTENCE_BREAK = re.compile(r'(?<=[.!?…])\s+|\n+')


def protected_spans(text):
    return [match.span() for match in PROTECTED.finditer(text)]


def is_table(text):
    return text.startswith('<table') or text.startswith('|')


def split_spans(text, breaks, protected):
    pieces, cursor = [], 0
    for match in breaks.finditer(text):
        if not any(start <= match.start() < end for start, end in protected):
            pieces.append((cursor, match.start()))
            cursor = match.end()
    pieces.append((cursor, len(text)))
    return [(left, right) for left, right in pieces if text[left:right].strip()]


def graphrag_chunks(text):
    token_ids = encoding.encode(text)
    byte_offsets = [0] + list(accumulate(len(encoding.decode_single_token_bytes(token)) for token in token_ids))
    character_byte_offsets = [0] + list(accumulate(len(character.encode()) for character in text))
    to_character = lambda offset: bisect_right(character_byte_offsets, offset) - 1
    chunks, start = [], 0
    while start < len(token_ids):
        end = min(start + GRAPHRAG_CHUNK_TOKENS, len(token_ids))
        chunks.append({'start': to_character(byte_offsets[start]), 'end': to_character(byte_offsets[end]),
                       'text': encoding.decode(token_ids[start:end]), 'overlap_tokens': GRAPHRAG_OVERLAP_TOKENS if start else 0})
        if end == len(token_ids):
            break
        start += GRAPHRAG_CHUNK_TOKENS - GRAPHRAG_OVERLAP_TOKENS
    return chunks


def broken_share(items, chunks):
    spans = sorted((piece['start'], piece['end']) for piece in chunks)
    starts = [start for start, _ in spans]

    def whole_somewhere(item):
        index = bisect_right(starts, item[0]) - 1
        return any(start <= item[0] and item[1] <= end for start, end in spans[max(0, index - 3):index + 1])
    return share(sum(not whole_somewhere(item) for item in items), len(items))


def chunking_metrics(text, chunks, size_limit):
    protected = protected_spans(text)
    sizes = [tokens(piece['text']) for piece in chunks]
    overlap = statistics.mean(piece['overlap_tokens'] for piece in chunks[1:])
    return {
        'chunks': len(chunks),
        'tokens_mean': round(statistics.mean(sizes)),
        'tokens_std': round(statistics.pstdev(sizes)),
        'tokens_min': min(sizes),
        'tokens_max': max(sizes),
        'overlap_share': round(overlap / size_limit, 3),
        'sentences': broken_share(split_spans(text, SENTENCE_BREAK, protected), chunks),
        'paragraphs': broken_share(split_spans(text, PARAGRAPH_BREAK, protected), chunks),
        'formulas': broken_share([span for span in protected if not is_table(text[span[0]:])], chunks),
        'tables': broken_share([span for span in protected if is_table(text[span[0]:])], chunks),
    }


def main():
    metrics = {}
    for source in sorted(SOURCE.glob('*.txt')):
        book, text = source.stem, source_text(source.stem)
        raw_text = (RAW_INPUT / f'{book}.txt').read_text(encoding='utf-8')
        chunks = json.loads((OUT / f'{book}.chunks.json').read_text(encoding='utf-8'))
        metrics[book] = {'graphrag_on_raw': chunking_metrics(raw_text, graphrag_chunks(raw_text), GRAPHRAG_CHUNK_TOKENS),
                         'recursive_on_lemmas': chunking_metrics(text, chunks, MAX_CHUNK_TOKENS)}
        print(book)
        for name, values in metrics[book].items():
            print(f"  {name:<16} {values['chunks']} chunks, {values['tokens_mean']} ± {values['tokens_std']} tokens "
                  f"({values['tokens_min']}–{values['tokens_max']}), overlap {values['overlap_share']:.0%}; broken " +
                  ', '.join(f"{kind} {values[kind]['count']}/{values[kind]['total']}" for kind in ('sentences', 'paragraphs', 'formulas', 'tables')))
    checks = []
    for book, values in metrics.items():
        ours = values['recursive_on_lemmas']
        checks += [check(f'{book}.tokens_mean', ours['tokens_mean'], '>=', MIN_CHUNK_TOKENS),
                   check(f'{book}.tokens_max', ours['tokens_max'], '<=', MAX_CHUNK_TOKENS),
                   check(f'{book}.tokens_std', ours['tokens_std'], '<=', MAX_TOKENS_STD),
                   check(f'{book}.overlap_share', ours['overlap_share'], '>=', MIN_OVERLAP_SHARE),
                   check(f'{book}.overlap_share', ours['overlap_share'], '<=', MAX_OVERLAP_SHARE),
                   check(f'{book}.sentences.share', ours['sentences']['share'], '<=', MAX_BROKEN_SENTENCES),
                   check(f'{book}.paragraphs.share', ours['paragraphs']['share'], '<=', MAX_BROKEN_PARAGRAPHS),
                   check(f'{book}.formulas.count', ours['formulas']['count'], '==', 0),
                   check(f'{book}.tables.count', ours['tables']['count'], '==', 0)]
    write_json(STAGE / 'metrics.json', metrics | {'checks': checks})


if __name__ == '__main__':
    main()
