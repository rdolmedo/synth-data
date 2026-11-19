"""aa_evals.common.jsonl: '\n'-only records, torn tails, no silent drops (#348)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aa_evals.common import jsonl

SEPARATORS = '\x0b\x0c\x1c\x1d\x1e\x85  '


class Loads(unittest.TestCase):
    def test_unicode_separators_are_data(self):
        rows = [{'t': f'a{c}b'} for c in SEPARATORS] + [{'t': 'plain'}]
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        self.assertGreater(len(text.splitlines()), len(rows))
        self.assertEqual(jsonl.loads(text), rows)

    def test_crlf_and_blank_lines(self):
        self.assertEqual(jsonl.loads('{"a":1}\r\n\n{"a":2}\r\n'), [{'a': 1}, {'a': 2}])

    def test_tail(self):
        self.assertEqual(jsonl.loads('{"a":1}\n{"a":2'), [{'a': 1}])          # being appended: skipped
        self.assertEqual(jsonl.loads('{"a":1}\n{"a":2}'), [{'a': 1}, {'a': 2}])  # no final newline: kept

    def test_bad_line_raises_or_is_collected(self):
        text = '{"a":1}\n{"a":\n{"a":3}\n'   # a torn line followed by later appends
        with self.assertRaises(jsonl.JSONLError) as e:
            jsonl.loads(text, where='f.jsonl')
        self.assertIn('f.jsonl:2', str(e.exception))
        bad = []
        self.assertEqual(jsonl.loads(text, where='f.jsonl', bad=bad), [{'a': 1}, {'a': 3}])
        self.assertEqual([(w, i) for w, i, _ in bad], [('f.jsonl', 2)])


class Read(unittest.TestCase):
    def test_missing(self):
        d = Path(tempfile.mkdtemp())
        self.assertEqual(jsonl.read(d / 'x.jsonl'), [])
        with self.assertRaises(FileNotFoundError):
            jsonl.read(d / 'x.jsonl', required=True)

    def test_tolerant_counts_once_per_line(self):
        d = Path(tempfile.mkdtemp())
        (d / 'run').mkdir()
        f = d / 'run' / 'items.jsonl'
        f.write_text('{"a":1}\n{"a\n{"a":3}\n', encoding='utf-8')
        t = jsonl.Tolerant()
        self.assertEqual(t.read(f), [{'a': 1}, {'a': 3}])
        t.read(f)
        self.assertEqual(t.count(d / 'run'), 1)
        self.assertEqual(t.count(d / 'other'), 0)


if __name__ == '__main__':
    unittest.main()
