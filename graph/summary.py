#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = [(str(path.parent.relative_to(ROOT)), result)
            for path in sorted(ROOT.glob('0*/metrics.json')) for result in json.loads(path.read_text(encoding='utf-8')).get('checks', [])]
    node_width = max(len(node) for node, _ in rows)
    metric_width = max(len(result['metric']) for _, result in rows)
    for node, result in rows:
        value = result['value']
        shown = f'{value:.3f}' if isinstance(value, float) else str(value)
        print(f"{node:<{node_width}}  {result['metric']:<{metric_width}}  {shown:>8}  {result['threshold']:<8}  "
              f"{'passed' if result['passed'] else 'FAILED'}")
    failed = sum(not result['passed'] for _, result in rows)
    print(f'\n{len(rows) - failed} of {len(rows)} checks passed')


if __name__ == '__main__':
    main()
