import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hf_import import parse_request

BASE = {'repo_id': 'cornell-movie-review-data/rotten_tomatoes', 'filename': 'test.parquet', 'revision': 'main', 'max_bytes': 10485760}

class ParseRequestTest(unittest.TestCase):
    def request(self, **overrides):
        return {'HF_REQUEST_JSON': json.dumps({**BASE, **overrides})}

    def test_valid(self):
        self.assertEqual(parse_request(self.request())[:3],
            ('cornell-movie-review-data/rotten_tomatoes', 'test.parquet', 'main'))

    def test_traversal_rejected(self):
        for name in ['../test.parquet', '/etc/passwd', 'sub/../../token', 'foo//bar']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_request(self.request(filename=name))

    def test_oversized_rejected(self):
        with self.assertRaises(ValueError):
            parse_request(self.request(max_bytes=300 * 1024 * 1024))

    def test_unexpected_fields_rejected(self):
        with self.assertRaises(ValueError):
            parse_request(self.request(url='https://example.com'))

if __name__ == '__main__':
    unittest.main()
