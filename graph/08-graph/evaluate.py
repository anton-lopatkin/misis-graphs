#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["networkx==3.7", "pandas==2.2.3", "pyarrow==25.0.1", "lancedb==0.37.1", "numpy==2.3.3", "lxml==5.3.0",
#                 "pint==0.24.4", "pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142", "pylatexenc==2.10",
#                 "httpx==0.28.1", "python-dotenv==1.2.1"]
# ///
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median

import lancedb
import networkx as nx
import numpy as np
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
sys.path.insert(0, str(ROOT / 'shared'))
from stage import Memo, check, memoized_answers, share, write_json
import openrouter

spec = importlib.util.spec_from_file_location('normalization', ROOT / '04-normalization' / 'run.py')
normalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalization)
morph = normalization.morph

ENTITY_TYPES = {'СТАЛЬ', 'ЭЛЕМЕНТ', 'ФАЗА', 'СТРУКТУРА', 'СВОЙСТВО', 'РЕЖИМ', 'ОБРАБОТКА', 'ПРЕВРАЩЕНИЕ', 'ИСПЫТАНИЕ',
                'ИЗДЕЛИЕ', 'ОБОРУДОВАНИЕ', 'СТАНДАРТ', 'ДЕФЕКТ'}
PRONOUNS = {'он', 'она', 'оно', 'они', 'его', 'ее', 'её', 'их', 'это', 'этот', 'тот', 'данный', 'такой', 'указанный', 'последний', 'который'}
FUNCTION_PARTS_OF_SPEECH = {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}
UNITS = r'(?:°C|°С|МПа|ГПа|Н/ММ²|%|ММ|МКМ|НМ|СМ|МИН|Ч|К/С|°C/С|ДЖ/СМ²|PPM)'
NOISE = {
    'reference': re.compile(r'(?:РИС(?:УНОК)?|ТАБЛ(?:ИЦА|А)?)\.?(?:\s*(?:И\s+)?[\d.,–-]+)*\s*[А-ЯA-Z]?'),
    'markup': re.compile(r'.*(?:[\\$]|[_^]\{|</?SUB>).*', re.I),
    'number_with_unit': re.compile(rf'[−-]?\d+(?:[,.]\d+)?(?:\s*[–-]\s*\d+(?:[,.]\d+)?)?\s*{UNITS}?|{UNITS}'),
    'single_symbol': re.compile(r'[^\s\d]'),
    'formula_variable': re.compile(r'[A-Za-zА-ЯЁΑ-Ωα-ω](?:_?[\d₀-₉⁰-⁹]{1,3}|_[A-Za-zА-ЯЁ]{1,3})'),
}
STRENGTH_CLASS = re.compile(r'[XKКSСCE]\d{2,3}')
DISPLAY_FORMULA = re.compile(r'\$\$(.*?)\$\$', re.S)
WORD = re.compile(r'(?<![_\w])[А-ЯЁа-яё]{3,}')
TABLE = re.compile(r'<table.*?</table>|(?:^\|[^\n]*\|[ \t]*(?:\n|$))+', re.S | re.M)
TABLE_CELL = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>|(?<=\|)([^|\n]+)(?=\|)', re.S)
DOMAIN = ('стали для труб большого диаметра и металлоконструкций: химический состав и микролегирование, '
          'горячая и контролируемая прокатка, ускоренное охлаждение, структура и свойства')
DOMAIN_CONCEPTS = 40
EMBEDDING_MODEL = 'openai/text-embedding-3-large'
SYNONYM_SIMILARITY = 0.8
COVERAGE_ENTITIES = 5
COVERAGE_RELATIONS = 10
EXAMPLES = 10
JUDGE = {'model': 'deepseek/deepseek-v4-flash', 'reasoning': {'enabled': False}}
TEXT_JUDGE_SAMPLE = 100
DIRECT_RELATIONS = 30
HALO_WIDTH = 3
MAX_FRAGMENTS = 2
PARALLEL_REQUESTS = 16
SEED = 42
TEXT_JUDGE_PROMPT = """Below is a term from a knowledge graph that an LLM built from a Russian textbook on metallurgy of steels,
its description in the graph, its neighbourhood collected by traversing the graph ({statements} statements "source —
target: description") and the textbook fragments the term was extracted from.
1. Count how many statements of the neighbourhood contradict the fragments, that is state the opposite or different
facts. A statement the fragments do not mention is not a contradiction.
2. Count the facts the fragments state about the term: what it is, its properties, its relations to other terms. Count
how many of these facts are present in the description or in the neighbourhood.
Answer with a JSON list of three integers [contradictions, facts, facts_present], for example [0, 5, 3], and nothing else.

Term: {term}: {description}

Neighbourhood:
{halo}

Fragments:
{fragments}"""


CONCEPTS_PROMPT = """List the {count} key concepts of the domain: {domain}. A key concept is a term a specialist expects
a knowledge base on the domain to define and relate to other terms. Write each concept in Russian, in the nominative case.
Answer with a JSON list of {count} distinct strings and nothing else."""
COVERAGE_PROMPT = """Below are a concept of the domain "{domain}" and the entities of a knowledge graph closest to it, each
with its description and its neighbourhood collected by traversing the graph (statements "source — target: description").
Answer 2 if the graph contains the concept together with its definition and relations to other concepts of the domain,
1 if the graph contains the concept without definition or relations, 0 if the graph does not contain the concept.
Answer with a JSON list of one integer, for example [1], and nothing else.

Concept: {concept}

Entities:
{entities}"""


class Graph:
    def __init__(self, run):
        output = run / 'output'
        self.entities = pd.read_parquet(output / 'entities.parquet')
        self.relationships = pd.read_parquet(output / 'relationships.parquet')
        self.types = dict(zip(self.entities.title, self.entities.type))
        self.edges = {frozenset((source, target)): (description, weight) for source, target, description, weight
                      in zip(*(self.relationships[column] for column in ('source', 'target', 'description', 'weight')))}
        self.adjacency = defaultdict(set)
        for source, target in zip(self.relationships.source, self.relationships.target):
            if source != target:
                self.adjacency[source].add(target)
                self.adjacency[target].add(source)
        vectors = lancedb.connect(output / 'lancedb').open_table('entity_description').to_pandas()
        by_id = dict(zip(vectors.id, vectors.vector))
        self.titles = list(self.entities.title)
        self.vectors = np.array([by_id[entity_id] for entity_id in self.entities.id], dtype=np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def breadth_first(self, start, depth=None):
        levels, queue = {start: 0}, deque([start])
        while queue:
            node = queue.popleft()
            if depth is not None and levels[node] == depth:
                continue
            for neighbour in self.adjacency[node]:
                if neighbour not in levels:
                    levels[neighbour] = levels[node] + 1
                    queue.append(neighbour)
        return levels

    def components(self):
        seen, found = set(), []
        for title in self.titles:
            if title not in seen:
                component = self.breadth_first(title)
                seen.update(component)
                found.append(list(component))
        return sorted(found, key=len, reverse=True)


def indicators(graph, components):
    structure = nx.Graph()
    structure.add_nodes_from(graph.titles)
    structure.add_edges_from((source, target) for source in graph.adjacency for target in graph.adjacency[source])
    degrees = [len(graph.adjacency[title]) for title in graph.titles]
    return {
        'entities': len(graph.titles),
        'relationships': structure.number_of_edges(),
        'connected_components': len(components),
        'largest_component': len(components[0]),
        'isolated_entities': sum(len(component) == 1 for component in components),
        'bridges': sum(1 for _ in nx.bridges(structure)),
        'independent_cycles': structure.number_of_edges() - structure.number_of_nodes() + len(components),
        'degree': {'average': round(sum(degrees) / len(degrees), 2), 'median': median(degrees), 'max': max(degrees)},
    }


def by_type(graph):
    pairs = Counter(' — '.join(sorted((graph.types[source], graph.types[target])))
                    for source in graph.adjacency for target in graph.adjacency[source] if source < target)
    return {'entities': dict(Counter(graph.types.values()).most_common()),
            'entities_outside_type_list': share(sum(type_ not in ENTITY_TYPES for type_ in graph.types.values()), len(graph.types)),
            'relationships': dict(pairs.most_common())}


def noise_kind(title, entity_type):
    if entity_type == 'ЭЛЕМЕНТ' or STRENGTH_CLASS.fullmatch(title):
        return None
    for kind, pattern in NOISE.items():
        if pattern.fullmatch(title.strip()):
            return kind
    words = title.lower().split()
    if all(re.fullmatch(r'[а-яё]+', word) and (word in PRONOUNS or morph.parse(word)[0].tag.POS in FUNCTION_PARTS_OF_SPEECH) for word in words):
        return 'function_word'
    return None


def noise(graph):
    found = defaultdict(list)
    for title in graph.titles:
        if kind := noise_kind(title, graph.types[title]):
            found[kind].append(title)
    return {**share(sum(map(len, found.values())), len(graph.titles)),
            'by_kind': {kind: {'count': len(titles), 'examples': sorted(titles)[:EXAMPLES]} for kind, titles in found.items()}}


def is_pronoun(title):
    words = title.lower().split()
    return bool(words) and all(word in PRONOUNS for word in words)


def numbers(title):
    return sorted(re.findall(r'\d+', unicodedata.normalize('NFKC', title)))


def coreferences(graph):
    similarity = graph.vectors @ graph.vectors.T
    np.fill_diagonal(similarity, 0)
    left, right = np.where(np.triu(similarity) >= SYNONYM_SIMILARITY)
    pairs = sorted(((graph.titles[a], graph.titles[b], float(similarity[a, b])) for a, b in zip(left, right)
                    if numbers(graph.titles[a]) == numbers(graph.titles[b])), key=lambda pair: -pair[2])
    return {'pronoun_entities': sum(map(is_pronoun, graph.titles)),
            'synonym_pairs': len(pairs), 'entities_in_synonym_pairs': len({title for pair in pairs for title in pair[:2]}),
            'synonym_examples': [[a, b, round(score, 3)] for a, b, score in pairs[:EXAMPLES]]}


def squash(text):
    return re.sub(r'[\s_^{}$\\]', '', unicodedata.normalize('NFKC', text)).casefold()


def within_window(graph, nodes):
    return any(nodes <= graph.breadth_first(center, depth=2).keys() for center in nodes)


def integrity(graph, units):
    to_text = LatexNodes2Text()
    by_unit = defaultdict(list)
    for title, unit_ids in zip(graph.entities.title, graph.entities.text_unit_ids):
        if graph.types[title] != 'ЭЛЕМЕНТ' and not WORD.search(title):
            for unit in unit_ids:
                by_unit[unit].append(title)
    formulas, formulas_in_parts, split_formulas = set(), set(), {}
    for unit, text in units.items():
        for formula in DISPLAY_FORMULA.findall(text):
            if 'begin{array}' in formula:
                continue
            key = squash(formula)
            formulas.add(key)
            plain = squash(to_text.latex_to_text(formula))
            parts = {title for title in by_unit[unit] if squash(title) and squash(title) in plain}
            if len(parts) > 1:
                formulas_in_parts.add(key)
                if not within_window(graph, parts):
                    split_formulas[key] = sorted(parts)
    unit_titles = defaultdict(dict)
    for title, unit_ids in zip(graph.entities.title, graph.entities.text_unit_ids):
        for unit in unit_ids:
            unit_titles[unit][squash(title)] = title
    tables, tables_in_parts, split_tables = set(), set(), {}
    for unit, text in units.items():
        for table in TABLE.findall(text):
            key = squash(table)[:200]
            tables.add(key)
            cells = {squash(first or second) for first, second in TABLE_CELL.findall(table)}
            nodes = {title for key_, title in unit_titles[unit].items() if key_ in cells}
            if len(nodes) > 1:
                tables_in_parts.add(key)
                if not within_window(graph, nodes):
                    split_tables[key] = sorted(nodes)
    return {'formulas': {'in_text': len(formulas), 'in_several_entities': len(formulas_in_parts), 'split': len(split_formulas),
                         'examples': list(split_formulas.values())[:EXAMPLES]},
            'tables': {'in_text': len(tables), 'in_several_entities': len(tables_in_parts), 'split': len(split_tables),
                       'examples': list(split_tables.values())[:EXAMPLES]}}


def halo(graph, center, direct_relations=DIRECT_RELATIONS):
    def strongest(node, excluded):
        return sorted(graph.adjacency[node] - excluded, key=lambda other: -graph.edges[frozenset((node, other))][1])

    def statement(node, other):
        return f'{node} — {other}: {graph.edges[frozenset((node, other))][0]}'
    direct = strongest(center, set())[:direct_relations]
    seen = {center, *direct}
    statements = [statement(center, other) for other in direct]
    for node in direct[:HALO_WIDTH]:
        for other in strongest(node, seen)[:HALO_WIDTH]:
            statements.append(statement(node, other))
            seen.add(other)
    return statements


def is_counts(answer):
    return (isinstance(answer, list) and len(answer) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in answer)
            and answer[2] <= answer[1])


def ask_judge(path, task, identity, items, is_valid):
    memo = Memo(path)
    with openrouter.session() as client, ThreadPoolExecutor(PARALLEL_REQUESTS) as pool:
        def ask(batch):
            try:
                return [openrouter.ask_list(client, JUDGE, batch[0], is_valid)]
            except openrouter.NoValidAnswer:
                return [None]
        answers = memoized_answers(memo, task, identity, items, ask, 1, pool)
    memo.save()
    return answers


def judge_against_text(graph, run, units):
    entities = graph.entities.set_index('title')
    sample = random.Random(SEED).sample(sorted(graph.titles), TEXT_JUDGE_SAMPLE)
    halos = [halo(graph, title) for title in sample]
    items = [TEXT_JUDGE_PROMPT.format(term=title, description=entities.description[title], statements=len(statements),
                                      halo='\n'.join(statements) or '—',
                                      fragments='\n\n'.join(units[unit] for unit in list(entities.text_unit_ids[title])[:MAX_FRAGMENTS]))
             for title, statements in zip(sample, halos)]
    answers = ask_judge(run / 'text_judge.json', 'text_judge', (JUDGE, TEXT_JUDGE_PROMPT), items, is_counts)
    judged = [(len(statements), answer) for statements, answer in zip(halos, answers) if answer]
    statements = sum(count for count, _ in judged)
    contradictions = sum(min(answer[0], count) for count, answer in judged)
    facts, present = sum(answer[1] for _, answer in judged), sum(answer[2] for _, answer in judged)
    completeness = {'terms': len(judged), 'facts': facts, 'facts_present': share(present, facts),
                    'fully_covered_terms': share(sum(answer[2] == answer[1] for _, answer in judged), len(judged))}
    consistency = {'terms': len(judged), 'statements': statements, 'contradicting_statements': share(contradictions, statements)}
    return completeness, consistency


def embeddings(texts):
    with openrouter.session() as client:
        response = client.post('https://openrouter.ai/api/v1/embeddings', json={'model': EMBEDDING_MODEL, 'input': texts})
    vectors = np.array([item['embedding'] for item in sorted(response.json()['data'], key=lambda item: item['index'])], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def domain_concepts():
    path = STAGE / 'domain_concepts.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    is_list = lambda answer: (isinstance(answer, list) and len(answer) == DOMAIN_CONCEPTS
                              and all(isinstance(item, str) and item.strip() for item in answer) and len(set(answer)) == len(answer))
    with openrouter.session() as client:
        concepts = openrouter.ask_list(client, JUDGE, CONCEPTS_PROMPT.format(count=DOMAIN_CONCEPTS, domain=DOMAIN), is_list)
    write_json(path, concepts)
    return concepts


def coverage(graph, run, concepts):
    descriptions = dict(zip(graph.entities.title, graph.entities.description))
    similarity = embeddings(concepts) @ graph.vectors.T
    items = []
    for concept, row in zip(concepts, similarity):
        nearest = [graph.titles[index] for index in np.argsort(-row)[:COVERAGE_ENTITIES]]
        blocks = [f'{title}: {descriptions[title]}\n' + ('\n'.join(halo(graph, title, COVERAGE_RELATIONS)) or '—') for title in nearest]
        items.append(COVERAGE_PROMPT.format(domain=DOMAIN, concept=concept, entities='\n\n'.join(blocks)))
    is_score = lambda answer: isinstance(answer, list) and len(answer) == 1 and answer[0] in (0, 1, 2) and not isinstance(answer[0], bool)
    answers = ask_judge(run / 'coverage_judge.json', 'coverage', (JUDGE, COVERAGE_PROMPT, DOMAIN), items, is_score)
    scores = {concept: answer[0] for concept, answer in zip(concepts, answers) if answer}
    return {'concepts': len(scores), 'covered': share(sum(score == 2 for score in scores.values()), len(scores)),
            'partially_covered': share(sum(score == 1 for score in scores.values()), len(scores)),
            'score': share(sum(scores.values()), 2 * len(scores)), 'by_concept': scores}


def main():
    parser = argparse.ArgumentParser(description='Criteria of one GraphRAG graph computed by traversing the graph.')
    parser.add_argument('run', choices=('raw', 'clean-default-chunks', 'clean'))
    run = STAGE / parser.parse_args().run
    graph = Graph(run)
    components = graph.components()
    units = dict(zip(*(pd.read_parquet(run / 'output' / 'text_units.parquet')[column] for column in ('id', 'text'))))
    completeness, consistency = judge_against_text(graph, run, units)
    metrics = {
        'completeness': completeness,
        'consistency': consistency,
        'coreferences': coreferences(graph),
        'noise': noise(graph),
        'integrity': integrity(graph, units),
        'indicators': indicators(graph, components),
        'domain_coverage': coverage(graph, run, domain_concepts()),
        'by_type': by_type(graph),
    }
    print(json.dumps({key: value for key, value in metrics.items() if key != 'by_type'}, ensure_ascii=False, indent=1)[:6000])
    write_json(run / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
