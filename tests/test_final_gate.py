"""The final-test campaign must use the queue that actually completed."""
import json
from pathlib import Path
import tempfile
import unittest
from protocol import digest
from run_final_tests import queue_complete


class FinalGateTests(unittest.TestCase):
    def test_requires_matching_completed_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/'manifest.json';manifest.write_text('{}')
            state=Path(folder)/'state.json'
            for status,sha in [('training',digest(manifest)),('queue_complete','wrong')]:
                state.write_text(json.dumps(dict(status=status,manifest_sha256=sha)))
                with self.assertRaises(RuntimeError):queue_complete(manifest,state)
            state.write_text(json.dumps(dict(status='queue_complete',manifest_sha256=digest(manifest))))
            queue_complete(manifest,state)


if __name__=='__main__':unittest.main()
