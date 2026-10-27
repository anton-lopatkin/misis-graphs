#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
import argparse
import json
import re
import subprocess
from pathlib import Path

STAGE = Path(__file__).resolve().parent
EXTRACTION = STAGE.parent / '01-extraction' / 'out'
NORMALIZATION = STAGE.parent / '04-normalization' / 'out' / 'lemmas'
MARKER = re.compile(r'⟦(?:TABLE|MATH)_\d+⟧')


def extracted_texts():
    return {source.stem: source.read_text(encoding='utf-8') for source in sorted(EXTRACTION.glob('*.txt'))}


def preprocessed_texts():
    texts = {}
    for source in sorted(NORMALIZATION.glob('*.markers.json')):
        book = source.name.removesuffix('.markers.json')
        markers = json.loads(source.read_text(encoding='utf-8'))
        text = (NORMALIZATION / f'{book}.txt').read_text(encoding='utf-8')
        texts[book] = MARKER.sub(lambda match: markers[match.group(0)], text)
    return texts


SOURCES = {'raw': extracted_texts, 'clean-default-chunks': preprocessed_texts, 'clean': preprocessed_texts}


def prepare_input(run):
    directory = STAGE / run / 'input'
    directory.mkdir(exist_ok=True)
    for book, text in SOURCES[run]().items():
        (directory / f'{book}.txt').write_text(text, encoding='utf-8')


def run_script(name, *arguments):
    subprocess.run(['uv', 'run', '--script', str(STAGE / name), *arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description='Build a GraphRAG graph from extracted or preprocessed text.')
    parser.add_argument('run', choices=SOURCES, help='raw: extracted text, clean-default-chunks: preprocessed text with default GraphRAG chunks, clean: preprocessed text with stage 05 chunks')
    args = parser.parse_args()
    prepare_input(args.run)
    run_script('index.py', args.run)
    run_script('export_graphml.py', args.run)
    run_script('evaluate.py', args.run)


if __name__ == '__main__':
    main()
