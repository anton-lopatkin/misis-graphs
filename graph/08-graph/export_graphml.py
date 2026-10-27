#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["networkx==3.7", "pandas==2.2.3", "pyarrow==25.0.1"]
# ///
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

STAGE = Path(__file__).resolve().parent
MAX_ENTITY_DESCRIPTION_CHARS = 2000
MAX_RELATIONSHIP_DESCRIPTION_CHARS = 1000


def entity_communities_by_level(communities):
    result = {}
    for _, community in communities.iterrows():
        for entity_id in community['entity_ids']:
            result.setdefault(entity_id, {})[int(community['level'])] = int(community['community'])
    return result


def build_graph(entities, relationships, levels, entity_communities):
    graph = nx.Graph()
    for _, entity in entities.iterrows():
        communities = entity_communities.get(entity['id'], {})
        graph.add_node(entity['title'],
                       title=entity['title'],
                       type=entity['type'] or '',
                       description=(entity['description'] or '')[:MAX_ENTITY_DESCRIPTION_CHARS],
                       frequency=int(entity['frequency']),
                       degree=int(entity['degree']),
                       **{f'community_l{level}': communities.get(level, -1) for level in levels})
    for _, relationship in relationships.iterrows():
        graph.add_edge(relationship['source'], relationship['target'],
                       weight=float(relationship['weight']),
                       description=(relationship['description'] or '')[:MAX_RELATIONSHIP_DESCRIPTION_CHARS],
                       combined_degree=int(relationship['combined_degree']))
    return graph


def main():
    parser = argparse.ArgumentParser(description='Export GraphRAG output to GraphML with titles, types, descriptions and communities.')
    parser.add_argument('run', choices=('raw', 'clean-default-chunks', 'clean'))
    output = STAGE / parser.parse_args().run / 'output'
    entities = pd.read_parquet(output / 'entities.parquet')
    relationships = pd.read_parquet(output / 'relationships.parquet')
    communities = pd.read_parquet(output / 'communities.parquet')
    levels = sorted(int(level) for level in communities['level'].unique())
    graph = build_graph(entities, relationships, levels, entity_communities_by_level(communities))
    nx.write_graphml(graph, output / 'graph_full.graphml')
    print(f'graph_full.graphml: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships')


if __name__ == '__main__':
    main()
