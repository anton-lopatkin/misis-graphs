#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["graphrag==3.1.2", "langchain-text-splitters==0.3.8"]
# ///
import argparse
import asyncio
import importlib.util
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from graphrag.api import build_index
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.load_config import load_config
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag_chunking.chunker import Chunker
from graphrag_chunking.chunker_factory import register_chunker
from graphrag_chunking.text_chunk import TextChunk

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
CHUNKING_STAGE = ROOT / '05-chunking' / 'run.py'
WORKFLOWS = ['load_input_documents', 'create_base_text_units', 'create_final_documents', 'extract_graph',
             'finalize_graph', 'create_communities', 'create_final_text_units', 'generate_text_embeddings']
ENTITY_EXTRACTION_ERROR = 'Entity Extraction Error'
RETRY_DELAY_SECONDS = 15
RECORD_SEPARATOR = re.compile(r'\)\s*\|*\s*(?:##|\|\|)\s*\|*\s*(?=\()')


def accept_stray_separators(process_result):
    def process(self, result, *args, **kwargs):
        return process_result(self, RECORD_SEPARATOR.sub(')\n##\n', result), *args, **kwargs)
    return process


GraphExtractor._process_result = accept_stray_separators(GraphExtractor._process_result)


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def load_chunking_stage():
    spec = importlib.util.spec_from_file_location('chunking_stage', CHUNKING_STAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage05Chunker(Chunker):
    def __init__(self, encode, **kwargs):
        self._encode = encode
        self._stage = load_chunking_stage()

    def chunk(self, text, transform=None):
        pieces = self._stage.chunk(text)
        results = []
        for index, piece in enumerate(pieces):
            final = transform(piece['text']) if transform else piece['text']
            results.append(TextChunk(original=piece['text'], text=final, index=index, start_char=piece['start'],
                                     end_char=piece['end'] - 1, token_count=len(self._encode(final))))
        return results


class LostTextUnitCounter(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.count = 0

    def emit(self, record):
        self.count += record.getMessage().startswith(ENTITY_EXTRACTION_ERROR)


class WorkflowLog(NoopWorkflowCallbacks):
    def workflow_start(self, name, instance):
        log(f'workflow {name} started')


async def run(root, tries):
    start = time.monotonic()
    index_logger = logging.getLogger('graphrag.index')
    for attempt in range(1, tries + 1):
        log(f'attempt {attempt}')
        config = load_config(root)
        config.workflows = WORKFLOWS
        lost = LostTextUnitCounter()
        index_logger.addHandler(lost)
        try:
            results = await build_index(config, callbacks=[WorkflowLog()])
            errors = [result.error for result in results if result.error]
        except Exception as error:
            errors = [error]
        finally:
            index_logger.removeHandler(lost)
        if not errors and not lost.count:
            log(f'done in {timedelta(seconds=round(time.monotonic() - start))}')
            return 0
        log(f'attempt {attempt} failed: {errors[0]}' if errors else f'attempt {attempt}: {lost.count} text units lost')
        await asyncio.sleep(RETRY_DELAY_SECONDS)
    log(f'failed after {tries} attempts')
    return 1


def main():
    parser = argparse.ArgumentParser(description='Run GraphRAG indexing, retrying lost text units from cache.')
    parser.add_argument('run', choices=('raw', 'clean-default-chunks', 'clean'))
    parser.add_argument('--tries', type=int, default=10, help='maximum attempts')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    register_chunker('stage05', Stage05Chunker)
    sys.exit(asyncio.run(run(STAGE / args.run, args.tries)))


if __name__ == '__main__':
    main()
