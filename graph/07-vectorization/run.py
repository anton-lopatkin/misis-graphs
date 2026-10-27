#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn==1.7.2", "scipy==1.16.2"]
# ///
import json
import re
import shutil
import sys
from pathlib import Path

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '06-tokenization' / 'out'
OUT = STAGE / 'out'
SEPARATE_KINDS = {'formula', 'number_with_unit'}
UNIT = re.compile(r'[^\d\s,–−-][^\d]*$')


def features(tokens):
    words = [token['text'] if token['kind'] != 'word' else token['text'].lower() for token in tokens if token['kind'] != 'symbol']
    return words + [f'{left} {right}' for left, right in zip(words, words[1:])]


def separate_features(token):
    unit = UNIT.search(token['text']) if token['kind'] == 'number_with_unit' else None
    return [token['text']] + ([unit.group(0).strip()] if unit else [])


def documents(chunk):
    tokens = [token for sentence in chunk['sentences'] for token in sentence]
    found = [{'chunk': chunk['id'], 'kind': 'text', 'features': features(tokens)}]
    for table in chunk['tables']:
        header, *rows = [[token for cell in row for token in cell] for row in table]
        found.append({'chunk': chunk['id'], 'kind': 'table_header', 'features': features(header)})
        found += [{'chunk': chunk['id'], 'kind': 'table_row', 'features': features(row)} for row in rows]
    found += [{'chunk': chunk['id'], 'kind': token['kind'], 'features': separate_features(token)}
              for token in tokens if token['kind'] in SEPARATE_KINDS]
    return [document for document in found if document['features']]


def vectorize(document_features):
    vectorizer = TfidfVectorizer(analyzer=lambda features: features, sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(document_features)


def load_documents():
    chunks = [json.loads(line) for path in sorted(SOURCE.glob('*.jsonl')) for line in path.read_text(encoding='utf-8').splitlines()]
    return [document for chunk in chunks for document in documents(chunk)]


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    found = load_documents()
    vectorizer, matrix = vectorize([document['features'] for document in found])
    save_npz(OUT / 'vectors.npz', matrix)
    write_json(OUT / 'documents.json', [{'chunk': document['chunk'], 'kind': document['kind']} for document in found], indent=1)
    write_json(OUT / 'vocabulary.json', vectorizer.get_feature_names_out().tolist(), indent=1)
    print(f'{matrix.shape[0]} documents × {matrix.shape[1]} features')


if __name__ == '__main__':
    main()
