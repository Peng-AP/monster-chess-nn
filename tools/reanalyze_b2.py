"""Stateful reanalysis with complete state retained in published teacher rows."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalyze_stateful
import reanalyze

original_one = reanalyze_stateful.reanalyze_one
original_write = reanalyze._write_teacher


def one(item):
    result = original_one(item)
    result['state'] = item['record']['state']
    return result


def write(output_dir, rows, model_path, simulations):
    if any('state' not in row for row in rows):
        raise ValueError('B2 teacher missing complete state')
    original_write(output_dir, rows, model_path, simulations)
    for index, row in enumerate(rows):
        path = Path(output_dir) / f'teacher_{index:05d}.jsonl'
        record = json.loads(path.read_text())
        record['state'] = row['state']
        path.write_text(json.dumps(record) + '\n', encoding='utf-8')


if __name__ == '__main__':
    reanalyze._reanalyze_one = one
    reanalyze._write_teacher = write
    reanalyze.__file__ = __file__
    reanalyze.main()
