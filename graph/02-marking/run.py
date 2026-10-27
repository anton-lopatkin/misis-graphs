#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '01-extraction' / 'out'
OUT = STAGE / 'out'

TABLE = re.compile(r'<table.*?</table>|^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*$', re.S | re.M)
BLOCK_FORMULA = re.compile(r'^\$\$\n.*?\n\$\$$', re.S | re.M)
INLINE_FORMULA = re.compile(r'\$[^$\n]+\$')
LEFTOVER = re.compile(r'\$|</?table|^\|', re.M)
UNIT_EXPONENT = re.compile(r'(Н/мм|H/mm|Дж/см|J/cm|кгс/мм|мм|mm|см|cm) ?(⟦MATH_\d+⟧)')
EXPONENT = re.compile(r'\$\s*\^\s*\{?\s*(-?\d+(?:/\d+)?)\s*\}?\s*\$')
SUPERSCRIPTS = str.maketrans('0123456789-/', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᐟ')
TABLE_UNIT_EXPONENT = re.compile(r'(Н/мм|H/mm|Дж/см|J/cm|кгс/мм|мм|mm|см|cm) ?\$\s*\^\s*\{?\s*(-?\d+(?:/\d+)?)\s*\}?\s*\$')


def mark(text):
    markers = {}

    def hide(kind):
        def replace(match):
            marker = f'⟦{kind}_{len(markers)}⟧'
            markers[marker] = match.group(0)
            return marker
        return replace

    for kind, pattern in (('TABLE', TABLE), ('MATH', BLOCK_FORMULA), ('MATH', INLINE_FORMULA)):
        text = pattern.sub(hide(kind), text)
    for marker in markers:
        if marker.startswith('⟦TABLE'):
            markers[marker] = TABLE_UNIT_EXPONENT.sub(lambda match: match.group(1) + match.group(2).translate(SUPERSCRIPTS), markers[marker])
    return return_unit_exponents(text, markers), markers


def return_unit_exponents(text, markers):
    def restore(match):
        exponent = EXPONENT.fullmatch(markers[match.group(2)])
        if not exponent:
            return match.group(0)
        del markers[match.group(2)]
        return match.group(1) + exponent.group(1).translate(SUPERSCRIPTS)
    return UNIT_EXPONENT.sub(restore, text)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    for source in sorted(SOURCE.glob('*.txt')):
        marked, markers = mark(source.read_text(encoding='utf-8'))
        if leftovers := [marked[max(0, match.start() - 40):match.end() + 40] for match in LEFTOVER.finditer(marked)]:
            raise SystemExit(f'{source.stem}: {len(leftovers)} delimiters left outside markers, e.g. {leftovers[:3]}')
        (OUT / source.name).write_text(marked, encoding='utf-8')
        write_json(OUT / f'{source.stem}.markers.json', markers, indent=1)
        kinds = Counter(marker[1:].split('_')[0] for marker in markers)
        print(f"{source.stem:<9} tables {kinds['TABLE']}, formulas {kinds['MATH']}")


if __name__ == '__main__':
    main()
