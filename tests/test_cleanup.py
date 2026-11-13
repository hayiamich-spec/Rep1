import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hf_cleanup import parse_request, validate_artifact

class CleanupRequestTests(unittest.TestCase):
    def test_parses_issue_json_artifact_id(self):
        self.assertEqual(parse_request({"HF_CLEAN_JSON": '{"artifact_id":14014533446}'}), 14014533446)

    def test_rejects_non_numeric_values(self):
        for text in ('{"artifact_id":true}', '{"artifact_id":0}', '{"artifact_id":-3}',
                     '{"artifact_id":"12x"}', '{"artifact_id":140,"other":"bad"}', "{}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_request({"HF_CLEAN_JSON": text})

    def test_requires_importer_artifact(self):
        sample={"id":14014533446, "name":"hf-dataset-43377088791",
                "workflow_run":{"id":43377088791,"repository_id":1427873421}}
        validate_artifact(sample,14014533446)
        for bad in (
            {**sample, "id":123},
            {**sample, "name":"db-backup-43377088791"},
            {**sample, "workflow_run":{"id":22,"repository_id":1427873421}},
            {**sample, "workflow_run":None},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_artifact(bad,14014533446)

if __name__ == "__main__":
    unittest.main()
