"""Test that final artifacts cannot silently drop or alter comparison groups."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from protocol import digest
from summary_guard import required_groups,verify_summary


class SummaryGuardTests(unittest.TestCase):
    def test_complete_summary_and_reject_missing_or_changed_group(self):
        fields=['head','kd','criterion','budget','allowed_drop','consensus']
        groups=[dict(zip(fields,k))|dict(accuracy_mean=.75,seeds=[17,42,73]) for k in required_groups()]
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/'manifest.json';manifest.write_text('{}')
            target=Path(folder)/'summary.json'
            payload=dict(exploratory=False,manifest_sha256=digest(manifest),groups=groups,provenance=[{'run':'verified'}])
            with patch('summary_guard.collect',return_value=([],payload['provenance'])),patch('summary_guard.aggregate',return_value=groups):
                target.write_text(json.dumps(payload));verify_summary(target,manifest)
                damaged=json.loads(json.dumps(payload));damaged['groups'].pop()
                target.write_text(json.dumps(damaged))
                with self.assertRaises(RuntimeError):verify_summary(target,manifest)
                damaged=json.loads(json.dumps(payload));damaged['groups'][0]['accuracy_mean']=.99
                target.write_text(json.dumps(damaged))
                with self.assertRaises(RuntimeError):verify_summary(target,manifest)


if __name__=='__main__':unittest.main()
