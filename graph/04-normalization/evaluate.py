#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pint==0.24.4", "pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142", "pylatexenc==2.10", "pandas==2.2.3", "lxml==5.3.0"]
# ///
import json
import re
from collections import Counter

import pandas

from run import ABBREVIATIONS, CELL_FORMULA, LEMMAS, OUT, SOURCE, STAGE
from stage import check, write_json

CONCEPTS = {
    'stress unit': r'МПа|MPa|МРа|[НH]\s?/\s?(?:мм|mm|MM)(?:\s?\^\s?\{?2\}?|²|2)?',
    'degree Celsius': r'\^\s*\{?\\circ\}?\s*(?:\\mathrm\s*\{\s*C\s*\}|C)|°\s?[CС](?![a-zа-я])',
    'minus sign': r'(?:(?<=^)|(?<=[\s(;÷/=]))[-–−](?=\d)',
    'range with unit': r'\d(?:\s?[-–]\s?|—)\d+(?:,\d+)?(?=\s?(?:°|%|МПа|мм|мкм|мин|Дж))',
    'percent': r'\d ?%',
    'yield strength σт': r'\\sigma\s*_\s*\{?\s*(?:\\(?:mathrm|text)\s*\{\s*)?[TтT]\s*\}?\s*\}?|σ_[TтT](?!\w)',
    'tensile strength σв': r'\\sigma\s*_\s*\{?\s*(?:\\(?:mathrm|text)\s*\{\s*)?[BвB]\s*\}?\s*\}?|σ_[BвB](?!\w)',
    'γ→α transformation': r'(?:\\gamma|γ)\s*(?:-|–|—|\\to|\\rightarrow|→)\s*(?:\\alpha|α)',
}
INLINE = re.compile(r'⟦MATH_\d+⟧')
NUMERIC_CELL = re.compile(r'[−+]?\d+(?:,\d+)?')
ABBREVIATION = re.compile(r'(?<![\w-])(' + '|'.join(map(re.escape, sorted(ABBREVIATIONS, key=len, reverse=True))) + r')(?![\w-])')
ABBREVIATION_IN_COMPOUND = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, sorted(ABBREVIATIONS, key=len, reverse=True))) + r')-\w')


def with_markers(path):
    markers = json.loads(path.with_name(path.name.replace('.txt', '.markers.json')).read_text(encoding='utf-8'))
    return path.read_text(encoding='utf-8') + '\n' + '\n'.join(value for value in markers.values() if not value.startswith('$$'))


def variants(text, pattern):
    return Counter(re.sub(r'\s+', ' ', re.sub(r'\d+(?:,\d+)?', 'N', found)) for found in re.findall(pattern, text, re.M))


def uniformity(before, after):
    result = {}
    for concept, pattern in CONCEPTS.items():
        old, new = variants(before, pattern), variants(after, pattern)
        canonical = new.most_common(1)[0][0] if new else None
        result[concept] = {'before': dict(old.most_common()), 'after': dict(new.most_common()),
                           'canonical_share': round(new[canonical] / sum(new.values()), 4) if new else None}
    return result


def main():
    books = [source.stem for source in sorted(SOURCE.glob('*.txt'))]
    before = '\n'.join(with_markers(SOURCE / f'{book}.txt') for book in books)
    after = '\n'.join(with_markers(OUT / f'{book}.txt') for book in books)
    lemmas = '\n'.join((LEMMAS / f'{book}.txt').read_text(encoding='utf-8') for book in books)
    metrics = {'uniformity': uniformity(before, after),
               'abbreviations': {'before': len(ABBREVIATION.findall(before)), 'after_in_lemmas': len(ABBREVIATION.findall(lemmas)),
                                 'left_in_compounds': len(ABBREVIATION_IN_COMPOUND.findall(lemmas))}}
    for book in books:
        text = (SOURCE / f'{book}.txt').read_text(encoding='utf-8')
        markers = json.loads((SOURCE / f'{book}.markers.json').read_text(encoding='utf-8'))
        normalized = (OUT / f'{book}.txt').read_text(encoding='utf-8')
        normalized_markers = json.loads((OUT / f'{book}.markers.json').read_text(encoding='utf-8'))
        inline = lambda source: sum(not markers[marker].startswith('$$') for marker in INLINE.findall(source))
        tables = [value for marker, value in normalized_markers.items() if marker.startswith('⟦TABLE')]
        cells = [cell.strip() for table in tables for row in table.split('\n')[2:] for cell in row.strip('|').split('|')]
        numeric = [cell for cell in cells if NUMERIC_CELL.fullmatch(cell)]
        metrics[book] = {
            'formulas': {'inline_before': inline(text), 'inline_left_as_latex': inline(normalized),
                         'block_kept': sum(value.startswith('$$') for value in normalized_markers.values())},
            'tables': {'to_markdown': len(tables), 'cell_formulas_left_as_latex': sum(len(CELL_FORMULA.findall(table)) for table in tables),
                       'cells_typed_as_numbers': len(numeric),
                       'numeric_cells_unparsed': int(pandas.to_numeric(pandas.Series(numeric, dtype=str).str.replace('−', '-').str.replace(',', '.'), errors='coerce').isna().sum())},
        }
    for concept, values in metrics['uniformity'].items():
        print(f"{concept:<22} {len(values['before'])} → {len(values['after'])} forms, canonical {values['canonical_share']:.0%}  {values['after']}")
    abbreviations = metrics['abbreviations']
    print(f"abbreviations          {abbreviations['before']} → {abbreviations['after_in_lemmas']} in lemmatized text, "
          f"{abbreviations['left_in_compounds']} left in compounds like ОЦК-решетка")
    for book in books:
        print(f"{book:<9} " + '; '.join(f"{group}: " + ', '.join(f"{name} {value}" for name, value in values.items())
                                        for group, values in metrics[book].items()))
    checks = [check(f'uniformity.{concept}.forms_after', len(values['after']), '==', 1) for concept, values in metrics['uniformity'].items()]
    checks.append(check('abbreviations.after_in_lemmas', metrics['abbreviations']['after_in_lemmas'], '==', 0))
    write_json(STAGE / 'metrics.json', metrics | {'checks': checks})


if __name__ == '__main__':
    main()
