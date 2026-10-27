#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["scikit-learn==1.7.2", "scipy==1.16.2", "numpy==2.3.3", "razdel==0.5.0", "pint==0.24.4", "pymorphy3==2.0.6",
#                 "pymorphy3-dicts-ru==2.4.417150.4580142", "pylatexenc==2.10", "pandas==2.2.3", "lxml==5.3.0",
#                 "httpx==0.28.1", "python-dotenv==1.2.1"]
# ///
import importlib.util
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from run import STAGE, features, load_documents, vectorize
import openrouter
from stage import Memo, check, memoized_answers, write_json

CHUNKS = STAGE.parent / '05-chunking' / 'out'
QUESTION_COUNT = 100
QUESTION_SEED = 42
MAX_TABLE_SHARE = 0.5
TOP_K = 10
MIN_HIT_RATE = 0.9
MIN_QUALITY = 0.75
VARIANCE_SHARE = 0.9
BATCH_SIZE = 1
PARALLEL_REQUESTS = 8
QUESTION_MODEL = {'model': 'deepseek/deepseek-v4-flash', 'reasoning': {'enabled': False}}
QUESTION_PROMPT = """Below are numbered fragments of a Russian textbook on metallurgy of steels; the text is lemmatized.
For each fragment write one natural Russian question that a metallurgy specialist might ask and that the fragment answers.
Use domain terms naturally, but do not copy phrases from the fragment. The question must be answerable from the fragment alone
and must not refer to the fragment, figures or tables. Answer with a JSON list of exactly {count} questions in order and nothing else.

{items}"""
TABLE = re.compile(r'(?m:^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*$)')


def stage_module(name):
    spec = importlib.util.spec_from_file_location(name, STAGE.parent / name / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalization, tokenization = stage_module('04-normalization'), stage_module('06-tokenization')


def query_features(question):
    text = normalization.lemmatize(normalization.unify(question, []))
    return features([token for sentence in tokenization.tokenized(text) for token in sentence])


def ask_batch(client, fragments):
    items = '\n\n'.join(f'{number}. {fragment}' for number, fragment in enumerate(fragments, 1))
    prompt = QUESTION_PROMPT.format(items=items, count=len(fragments))
    is_valid = lambda answer: isinstance(answer, list) and len(answer) == len(fragments) and all(isinstance(item, str) for item in answer)
    try:
        return openrouter.ask_list(client, QUESTION_MODEL, prompt, is_valid)
    except openrouter.NoValidAnswer:
        return [None] * len(fragments)


def load_questions():
    chunks = [chunk for path in sorted(CHUNKS.glob('*.chunks.json')) for chunk in json.loads(path.read_text(encoding='utf-8'))]
    prose = [chunk for chunk in chunks if sum(map(len, TABLE.findall(chunk['text']))) / len(chunk['text']) < MAX_TABLE_SHARE]
    sample = random.Random(QUESTION_SEED).sample(prose, QUESTION_COUNT)
    memo = Memo(STAGE / 'questions.json')
    with openrouter.session() as client, ThreadPoolExecutor(PARALLEL_REQUESTS) as pool:
        answers = memoized_answers(memo, 'question', (QUESTION_MODEL, QUESTION_PROMPT), [chunk['text'] for chunk in sample],
                                   lambda batch: ask_batch(client, batch), BATCH_SIZE, pool)
    memo.save()
    return [chunk['id'] for chunk in chunks], {chunk['id']: question for chunk, question in zip(sample, answers) if question}


def neighbours(chunk_id, chunk_ids):
    book, number = chunk_id.rsplit('-', 1)
    return {f'{book}-{int(number) + offset:04d}' for offset in (-1, 0, 1)} & set(chunk_ids)


def retrieval(documents, vectorizer, matrix, chunk_ids, questions):
    owner = np.array([chunk_ids.index(document['chunk']) for document in documents])
    text_rows = np.array([document['kind'] == 'text' for document in documents])
    ranks, own_similarity, other_similarity = {'strict': [], 'with_neighbours': []}, [], []
    for chunk_id, question in questions.items():
        scores = (matrix @ vectorizer.transform([query_features(question)]).T).toarray().ravel()
        chunk_scores = np.zeros(len(chunk_ids))
        np.maximum.at(chunk_scores, owner, scores)
        ranking = [chunk_ids[index] for index in np.argsort(-chunk_scores, kind='stable')]
        text_scores = dict(zip(owner[text_rows], scores[text_rows]))
        own_similarity.append(text_scores[chunk_ids.index(chunk_id)])
        other_similarity.append(np.mean([score for index, score in text_scores.items() if chunk_ids[index] != chunk_id]))
        for variant, relevant in (('strict', {chunk_id}), ('with_neighbours', neighbours(chunk_id, chunk_ids))):
            ranks[variant].append(next(rank for rank, candidate in enumerate(ranking, 1) if candidate in relevant))
    metrics = {variant: {f'hit_rate_at_{TOP_K}': round(sum(rank <= TOP_K for rank in found) / len(found), 3),
                         f'mrr_at_{TOP_K}': round(sum(1 / rank for rank in found if rank <= TOP_K) / len(found), 3),
                         f'ndcg_at_{TOP_K}': round(sum(1 / math.log2(rank + 1) for rank in found if rank <= TOP_K) / len(found), 3)}
               for variant, found in ranks.items()}
    metrics['cosine_similarity'] = {'question_and_its_chunk': round(float(np.mean(own_similarity)), 3),
                                    'question_and_other_chunks': round(float(np.mean(other_similarity)), 3)}
    return metrics


def dimensionality(matrix):
    dense = matrix.toarray()
    singular = np.linalg.svd(dense - dense.mean(axis=0), compute_uv=False)
    explained = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    return {'documents': matrix.shape[0], 'features': matrix.shape[1], 'non_zero_share': round(matrix.nnz / np.prod(matrix.shape), 5),
            'pca_components_for_90_percent': int(np.searchsorted(explained, VARIANCE_SHARE) + 1)}


def main():
    documents = load_documents()
    vectorizer, matrix = vectorize([document['features'] for document in documents])
    chunk_ids, questions = load_questions()
    metrics = {'questions': len(questions), 'retrieval': retrieval(documents, vectorizer, matrix, chunk_ids, questions),
               'dimensionality': dimensionality(matrix)}
    print(f"{metrics['questions']} questions")
    for variant in ('strict', 'with_neighbours'):
        print(f'  {variant:<16} ' + ', '.join(f'{name} {value:.1%}' for name, value in metrics['retrieval'][variant].items()))
    similarity = metrics['retrieval']['cosine_similarity']
    print(f"  cosine similarity: question and its chunk {similarity['question_and_its_chunk']}, other chunks {similarity['question_and_other_chunks']}")
    print(f"  dimensionality: {metrics['dimensionality']}")
    checks = [check(f'{variant}.hit_rate_at_{TOP_K}', metrics['retrieval'][variant][f'hit_rate_at_{TOP_K}'], '>', MIN_HIT_RATE)
              for variant in ('strict', 'with_neighbours')]
    checks += [check(f'{variant}.{name}_at_{TOP_K}', metrics['retrieval'][variant][f'{name}_at_{TOP_K}'], '>=', MIN_QUALITY)
               for variant in ('strict', 'with_neighbours') for name in ('mrr', 'ndcg')]
    write_json(STAGE / 'metrics.json', metrics | {'checks': checks})


if __name__ == '__main__':
    main()
