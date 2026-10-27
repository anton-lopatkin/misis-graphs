#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pint==0.24.4", "pymorphy3==2.0.6", "pymorphy3-dicts-ru==2.4.417150.4580142", "pylatexenc==2.10", "pandas==2.2.3", "lxml==5.3.0"]
# ///
import io
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas
import pint
import pymorphy3
from pylatexenc.latexwalker import LatexCharsNode, LatexGroupNode, LatexMacroNode, LatexSpecialsNode, LatexWalker

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE.parent / 'shared'))
from stage import write_json

SOURCE = STAGE.parent / '03-cleaning' / 'out'
OUT = STAGE / 'out'
LEMMAS = OUT / 'lemmas'

MARKER = re.compile(r'⟦(?:TABLE|MATH)_\d+⟧')
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
ABBREVIATIONS = json.loads((STAGE / 'abbreviations.json').read_text(encoding='utf-8'))

SPELLINGS = {
    '°С': '°C', '° C': '°C', '° С': '°C', '°C/c': '°C/с', '°С/с': '°C/с', '°C/s': '°C/с',
    'MPa': 'МПа', 'МРа': 'МПа', 'Н/мм²': 'МПа', 'H/mm²': 'МПа', 'Н/мм2': 'МПа',
    'ррт': 'ppm', 'ртм': 'ppm', 'НОР': 'HOP', 'ВН': 'BH',
}
GRADE_LETTERS = str.maketrans('CBFHJOATPMKXE', 'СБФНЮОАТРМКХЕ')
NOTATION = [
    (re.compile(r'σ_[Tr](?!\w)'), 'σ_т'),
    (re.compile(r'σ_[Bv](?!\w)'), 'σ_в'),
    (re.compile(r'[НH]\s?/\s?(?:мм|mm|MM)(?=[²2](?!\w))'), 'Н/мм'),
    (re.compile(r'([γα])\s*[-–—→]\s*([γα])'), r'\1→\2'),
    (re.compile(r'\b\d{2}(?=[A-ZА-Я0-9]*[A-Z])(?=[A-ZА-Я0-9]*[А-Я])[A-ZА-Я0-9]+\b'), lambda match: match.group(0).translate(GRADE_LETTERS)),
    (re.compile(r'(?<![A-Za-zА-Яа-яЁё°])(' + '|'.join(map(re.escape, sorted(SPELLINGS, key=len, reverse=True))) + r')(?![\w²³])'),
     lambda match: SPELLINGS[match.group(0)]),
]
CONVERSIONS = {
    'кгс/мм²': ('МПа', 'kgf/mm**2', 'MPa'), 'атм': ('МПа', 'atm', 'MPa'), 'ppm': ('%', 'ppm', 'percent'),
    'мм рт. ст.': ('Па', 'mmHg', 'Pa'), 'мм рт.ст.': ('Па', 'mmHg', 'Pa'),
    'Н/мм³ᐟ²': ('МПа·м¹ᐟ²', 'N/mm**1.5', 'MPa*m**0.5'), 'кгс/мм³ᐟ²': ('МПа·м¹ᐟ²', 'kgf/mm**1.5', 'MPa*m**0.5'),
}
VALUE = r'-?\d+(?:,\d+)?'
QUANTITY = re.compile(rf'(?<![\w,.])({VALUE})(?:(\s?[–-]\s?)({VALUE}))?(\s?)('
                      + '|'.join(map(re.escape, sorted(CONVERSIONS, key=len, reverse=True))) + r')(?![\w²³])')
STANDARD = re.compile(r'(?:ГОСТ|ОСТ|ТУ|СТО|СНиП|API|ISO|EN|ASTM|DIN|TM|№)[\sР]*[\d.\s-]*$')
NUMBERS = [
    (re.compile(r'(?<![\w,.])(\d+(?:,\d+)?)(?:[-—]| - (?=\d+(?:,\d+)?\s?(?:°|%|МПа|мм|мкм|мин|Дж)))(\d+(?:,\d+)?)(?![\w-])'),
     lambda match: match.group(0) if STANDARD.search(match.string[max(0, match.start() - 30):match.start()]) else f'{match.group(1)}–{match.group(2)}'),
    (re.compile(r'(?:(?<=^)|(?<=[\s(;÷/]))[-–](?=\d)', re.M), '−'),
    (re.compile(r'(?<=\d)(?=%|°C)'), ' '),
]

FONT_COMMANDS = {'mathrm', 'mathbf', 'mathit', 'mathsf', 'text', 'textrm', 'textbf', 'operatorname', 'boldsymbol'}
SYMBOLS = {
    'circ': '°', '%': '%', 'times': '×', 'cdot': '·', 'pm': '±', 'approx': '≈', 'sim': '~', 'leq': '≤', 'le': '≤',
    'geq': '≥', 'ge': '≥', 'to': '→', 'rightarrow': '→', 'infty': '∞', 'prime': '′', ',': ' ', ';': ' ', ':': ' ', 'quad': ' ', ' ': ' ', '!': '', '{': '{', '}': '}',
    'max': 'max', 'min': 'min', 'ln': 'ln', 'lg': 'lg', 'exp': 'exp', 'partial': '∂', 'pi': 'π', 'Phi': 'Ф', 'Gamma': 'Г', 'Pi': 'П',
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'Delta': 'Δ', 'varepsilon': 'ε', 'epsilon': 'ε', 'eta': 'η',
    'theta': 'θ', 'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'phi': 'φ',
    'varphi': 'φ', 'psi': 'ψ', 'Psi': 'Ψ', 'omega': 'ω', 'Omega': 'Ω', 'Sigma': 'Σ', 'chi': 'χ', 'xi': 'ξ',
}
SUBSCRIPTS = str.maketrans('0123456789+-−', '₀₁₂₃₄₅₆₇₈₉₊₋₋')
SUPERSCRIPTS = str.maketrans('0123456789+-−', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻')
SCRIPT_DIGITS = str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹', '01234567890123456789')
RELATION = re.compile(r'\s*([=≈<>≤≥+])\s*')
CELL_FORMULA = re.compile(r'\$[^$]+\$')
TABLE_SEPARATOR = re.compile(r'[\s|:-]+')
UNIT_EXPONENT = re.compile(r'(?<=[A-Za-zА-Яа-я}$])(?:\^\{?[23]\}?|[²³])(?![\d⁰¹²³⁴⁵⁶⁷⁸⁹])')

TOKEN = re.compile(r'⟦(?:TABLE|MATH)_\d+⟧|[\wё]+(?:-[\wё]+)*')
RUSSIAN_WORD = re.compile(r'[а-яё]+(?:-[а-яё]+)*', re.I)
UNIT_WORDS = {'мм', 'мкм', 'нм', 'см', 'км', 'м', 'мин', 'ч', 'с', 'т', 'кг', 'г', 'мг', 'л'}
UNCHANGED_WORDS = {'его', 'ее', 'её', 'их'}
PROPER_NAMES = {'Name', 'Surn', 'Patr', 'Geox', 'Orgn', 'Trad'}
SENTENCE_START = set('.!?:\n#—')

units = pint.UnitRegistry()
morph = pymorphy3.MorphAnalyzer()


def script(kind, content):
    content = content.strip()
    if content in ('°', '′') and kind == '^':
        return content
    if re.fullmatch(r'[0-9+\-−]+', content):
        return content.translate(SUBSCRIPTS if kind == '_' else SUPERSCRIPTS)
    return f'{kind}{content}' if re.fullmatch(r'\w+', content) else f'{kind}({content})'


def render(nodes):
    out, index = '', 0
    while index < len(nodes):
        node = nodes[index]
        following = nodes[index + 1] if index + 1 < len(nodes) else None
        if isinstance(node, LatexCharsNode) and node.chars[-1:] in '_^' and node.chars and isinstance(following, LatexGroupNode):
            out += node.chars[:-1] + script(node.chars[-1], render(following.nodelist))
            index += 1
        elif isinstance(node, LatexCharsNode):
            out += re.sub(r'([_^])(\w)', lambda match: script(*match.groups()), node.chars)
        elif isinstance(node, LatexGroupNode):
            out += render(node.nodelist)
        elif isinstance(node, LatexSpecialsNode) and node.specials_chars == '~':
            out += ' '
        elif isinstance(node, LatexMacroNode) and node.macroname in FONT_COMMANDS and node.nodeargd and node.nodeargd.argnlist:
            out += render([argument for argument in node.nodeargd.argnlist if argument is not None])
        elif isinstance(node, LatexMacroNode) and node.macroname in ('sqrt', 'frac') and node.nodeargd:
            parts = [render([argument]) for argument in node.nodeargd.argnlist if argument is not None]
            out += f'√({parts[-1]})' if node.macroname == 'sqrt' else '/'.join(part if re.fullmatch(r'[\w′]+', part) else f'({part})' for part in parts)
        elif isinstance(node, LatexMacroNode) and node.macroname in SYMBOLS:
            out += SYMBOLS[node.macroname]
        else:
            raise ValueError(node.latex_verbatim())
        index += 1
    return out


def latex_text(latex):
    try:
        text = render(LatexWalker(latex.strip('$'), tolerant_parsing=False).get_latex_nodes(pos=0)[0])
    except Exception:
        return None
    text = re.sub(r'\s+', ' ', RELATION.sub(r' \1 ', text)).strip()
    return text if re.sub(r'\D', '', text.translate(SCRIPT_DIGITS)) == re.sub(r'\D', '', latex) else None


def replace(text, rules, changed):
    def record(match, replacement):
        result = match.expand(replacement) if isinstance(replacement, str) else replacement(match)
        changed.append((match.group(0), result))
        return result
    for pattern, replacement in rules:
        text = pattern.sub(lambda match: record(match, replacement), text)
    return text


def convert_quantity(match):
    first, dash, second, space, unit = match.groups()
    target, source_unit, si_unit = CONVERSIONS[unit]
    factor = units.Quantity(1, source_unit).to(si_unit).magnitude
    convert = lambda value: f"{float(value.replace(',', '.')) * factor:.4g}".replace('.', ',')
    return convert(first) + (dash + convert(second) if second else '') + space + target


def unify(text, changed):
    text = replace(text, NOTATION, changed)
    text = replace(text, [(QUANTITY, convert_quantity)], changed)
    return replace(text, NUMBERS, changed)


def best_parse(word, following):
    parses = morph.parse(word.lower())
    if word.lower() == 'стали' and not (following and following.tag.POS == 'INFN'):
        return next(parse for parse in parses if parse.tag.POS == 'NOUN')
    return next((parse for parse in parses if not PROPER_NAMES & parse.tag.grammemes), parses[0])


def is_protected(token, text):
    word, before = token.group(0), text[max(0, token.start() - 20):token.start()].rstrip(' \t')
    heading = text.startswith('#', text.rfind('\n', 0, token.start()) + 1)
    sentence_start = heading or token.start() < 20 and not before or bool(before) and before[-1] in SENTENCE_START
    return (not RUSSIAN_WORD.fullmatch(word) or any(letter.isupper() for letter in word[1:]) or word.lower() in UNIT_WORDS
            or (text.startswith('.', token.end()) and len(word) <= 3)
            or (word[0].isupper() and not sentence_start))


def agreeing_noun(index, tokens, parses, text):
    for next_index in range(index + 1, len(tokens)):
        if text[tokens[next_index - 1].end():tokens[next_index].start()].strip() or parses[next_index] is None:
            return None
        if parses[next_index].tag.POS == 'NOUN':
            agree = any(left.tag.POS in ('ADJF', 'PRTF') and right.tag.POS == 'NOUN' and left.tag.case == right.tag.case
                        and left.tag.number == right.tag.number and (left.tag.number == 'plur' or left.tag.gender == right.tag.gender)
                        for left in morph.parse(tokens[index].group(0).lower()) for right in morph.parse(tokens[next_index].group(0).lower()))
            return parses[next_index] if agree else None
        if parses[next_index].tag.POS not in ('ADJF', 'PRTF'):
            return None
    return None


def word_lemma(index, tokens, parses, text):
    word, parse = tokens[index].group(0), parses[index]
    if word in ABBREVIATIONS:
        return lemmatize(ABBREVIATIONS[word])
    if parse is None:
        prefix, _, tail = word.rpartition('-')
        lemmatized_tail = prefix and re.search(r'[^\W\d_]', prefix) and len(tail) > 2 and RUSSIAN_WORD.fullmatch(tail) and tail.islower()
        return f'{prefix}-{lemmatize(tail)}' if lemmatized_tail else word
    if word.lower() in UNCHANGED_WORDS or parse.tag.POS == 'COMP':
        return word.lower()
    if parse.tag.POS in ('ADJF', 'PRTF'):
        if noun := agreeing_noun(index, tokens, parses, text):
            number = 'plur' if 'Pltm' in noun.tag.grammemes else 'sing'
            gender = {'ms-f': 'masc'}.get(noun.normalized.tag.gender, noun.normalized.tag.gender)
            agreed = next((left.inflect({'nomn', number} | ({gender} if number == 'sing' and gender else set()))
                           for left in morph.parse(word.lower()) if left.tag.POS == parse.tag.POS), None)
            if agreed:
                return agreed.word.replace('ё', 'е')
        if parse.tag.POS == 'PRTF' and (participle := parse.inflect({'masc', 'sing', 'nomn'})):
            return participle.word.replace('ё', 'е')
    return parse.normal_form.replace('ё', 'е')


def lemmatize(text):
    tokens = list(TOKEN.finditer(text))
    parses = [None] * len(tokens)
    for index in reversed(range(len(tokens))):
        if not is_protected(tokens[index], text):
            parses[index] = best_parse(tokens[index].group(0), parses[index + 1] if index + 1 < len(tokens) else None)
    pieces, position = [], 0
    for index, token in enumerate(tokens):
        pieces += [text[position:token.start()], word_lemma(index, tokens, parses, text)]
        position = token.end()
    return ''.join(pieces + [text[position:]])


def table_frame(table):
    if table.startswith('<'):
        width = pandas.read_html(io.StringIO(table), header=None, thousands=None)[0].shape[1]
        return pandas.read_html(io.StringIO(table), header=None, keep_default_na=False, thousands=None,
                                converters={column: str for column in range(width)})[0]
    rows = [line.strip().strip('|').split('|') for line in table.splitlines() if line.strip() and not TABLE_SEPARATOR.fullmatch(line)]
    width = max(map(len, rows))
    return pandas.DataFrame([[cell.strip() for cell in row] + [''] * (width - len(row)) for row in rows])


def markdown(frame):
    rows = [[str(cell).replace('|', '/') for cell in row] for row in frame.values.tolist()]
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '|' + '---|' * len(rows[0])] + ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])


def cell_digits(cell):
    return re.sub(r'\D', '', UNIT_EXPONENT.sub('', cell).translate(SCRIPT_DIGITS))


def normalize_table(table):
    source = table_frame(table).map(str)
    frame = source.map(lambda cell: unify(' '.join(CELL_FORMULA.sub(lambda match: latex_text(match.group(0)) or match.group(0), cell).split()), []))
    if mismatch := [(before, after) for before, after in zip(source.values.flat, frame.values.flat) if cell_digits(before) != cell_digits(after)]:
        raise SystemExit(f'table cells changed digits: {mismatch[:5]}')
    return markdown(frame), markdown(frame.map(lemmatize))


def formula(match, markers):
    latex = markers[match.group(0)]
    return match.group(0) if latex.startswith('$$') else latex_text(latex) or match.group(0)


def normalize(text, markers):
    changed = []
    text = replace(text, [(re.compile(r'⟦MATH_\d+⟧'), lambda match: formula(match, markers))], changed)
    text = unify(text, changed)
    referenced = MARKER.findall(text)
    tables = {marker: normalize_table(markers[marker]) for marker in referenced if marker.startswith('⟦TABLE')}
    normalized_markers = {marker: tables[marker][0] if marker in tables else markers[marker] for marker in referenced}
    lemma_markers = {marker: tables[marker][1] if marker in tables else markers[marker] for marker in referenced}
    return text, lemmatize(text), normalized_markers, lemma_markers, changed


def protected(text):
    return Counter(MARKER.findall(text)) + Counter(NUMBER.findall(MARKER.sub(' ', text)))


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    LEMMAS.mkdir(parents=True)
    for source in sorted(SOURCE.glob('*.txt')):
        text = source.read_text(encoding='utf-8')
        markers = json.loads((SOURCE / f'{source.stem}.markers.json').read_text(encoding='utf-8'))
        normalized, lemmas, normalized_markers, lemma_markers, changed = normalize(text, markers)
        removed = sum((protected(before) - protected(after) for before, after in changed), Counter())
        added = sum((protected(after) - protected(before) for before, after in changed), Counter())
        if unexplained := protected(text) - removed - (protected(normalized) - added):
            raise SystemExit(f'{source.stem}: numbers or markers changed outside recorded replacements: {dict(unexplained.most_common(10))}')
        if protected(lemmas) != protected(normalized):
            raise SystemExit(f'{source.stem}: lemmatization changed numbers or markers')
        (OUT / source.name).write_text(normalized, encoding='utf-8')
        (LEMMAS / source.name).write_text(lemmas, encoding='utf-8')
        write_json(OUT / f'{source.stem}.markers.json', normalized_markers, indent=1)
        write_json(LEMMAS / f'{source.stem}.markers.json', lemma_markers, indent=1)
        print(f'{source.stem:<9} {len(MARKER.findall(text))} → {len(normalized_markers)} markers, {len(changed)} replacements')


if __name__ == '__main__':
    main()
