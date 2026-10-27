#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken==0.12.0", "langchain-text-splitters==0.3.8"]
# ///
import json
import re
import shutil
import sys
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '04-normalization' / 'out' / 'lemmas'
OUT = STAGE / 'out'
MAX_CHUNK_TOKENS = 1024
OVERLAP_TOKENS = 250
MARKER = re.compile(r'⟦(?:TABLE|MATH)_\d+⟧')

encoding = tiktoken.get_encoding('o200k_base')
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='o200k_base', chunk_size=MAX_CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS)


def tokens(text):
    return len(encoding.encode(text))


def chunk(text):
    chunks, previous_start, previous_end = [], -1, 0
    for piece in splitter.split_text(text):
        start = text.find(piece, previous_start + 1)
        if start < 0:
            raise SystemExit(f'chunk not found in the source text: {piece[:80]!r}')
        chunks.append({'start': start, 'end': start + len(piece), 'text': piece,
                       'overlap_tokens': tokens(text[start:previous_end]) if start < previous_end else 0})
        previous_start, previous_end = start, start + len(piece)
    return chunks


def source_text(book):
    markers = json.loads((SOURCE / f'{book}.markers.json').read_text(encoding='utf-8'))
    return MARKER.sub(lambda match: markers[match.group(0)], (SOURCE / f'{book}.txt').read_text(encoding='utf-8'))


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    for source in sorted(SOURCE.glob('*.txt')):
        chunks = chunk(source_text(source.stem))
        write_json(OUT / f'{source.stem}.chunks.json', [{'id': f'{source.stem}-{index:04d}', **piece} for index, piece in enumerate(chunks, 1)], indent=1)
        print(f'{source.stem:<9} {len(chunks)} chunks')


if __name__ == '__main__':
    main()
