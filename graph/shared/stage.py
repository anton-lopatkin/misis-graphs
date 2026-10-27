import hashlib
import json
import operator

COMPARATORS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}


def write_json(path, data, indent=2):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent, default=str), encoding='utf-8')


def share(count, total, digits=4):
    return {'count': count, 'total': total, 'share': round(count / total, digits) if total else None}


def check(metric, value, comparator, bound):
    passed = value is not None and COMPARATORS[comparator](value, bound)
    return {'metric': metric, 'value': value, 'threshold': f'{comparator} {bound}', 'passed': passed}


def fingerprint(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Memo:
    def __init__(self, path):
        self.path = path
        self.saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        self.used = {}

    def get(self, key):
        if key in self.saved:
            self.used[key] = self.saved[key]
            return self.saved[key]['answer']
        return None

    def put(self, key, task, item, answer):
        if answer is not None:
            self.used[key] = {'task': task, 'item': item, 'answer': answer}

    def save(self):
        write_json(self.path, self.used, indent=1)


def memoized_answers(memo, tasks, identity, items, ask_batch, batch_size, pool):
    tasks = [tasks] * len(items) if isinstance(tasks, str) else tasks
    keys = [fingerprint(identity, item) for item in items]
    answers = [memo.get(key) for key in keys]
    missing = [index for index, answer in enumerate(answers) if answer is None]
    if not missing:
        return answers
    batches = [missing[start:start + batch_size] for start in range(0, len(missing), batch_size)]
    for batch, result in zip(batches, pool.map(lambda batch: ask_batch([items[index] for index in batch]), batches)):
        for index, answer in zip(batch, result):
            answers[index] = answer
            memo.put(keys[index], tasks[index], items[index], answer)
        memo.save()
    return answers
