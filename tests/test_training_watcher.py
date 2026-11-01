"""Verify that only a successfully completed matching queue starts final tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from protocol import digest
import watch_training


class FakeProcess:
    def __init__(self,pid,cwd):self.pid=pid;self.root=cwd
    def cmdline(self):return ['python','run_queue.py','configs/main_queue.json']
    def cwd(self):return str(self.root)
    def create_time(self):return 1234.0+self.pid
    def is_running(self):return False


class WatcherTests(unittest.TestCase):
    def run_case(self,status,pause=False):
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'runs').mkdir();(root/'configs').mkdir()
            manifest=root/'configs/main_queue.json';manifest.write_text('{}')
            state={'status':status,'manifest_sha256':digest(manifest)}
            (root/'runs/queue_state.json').write_text(json.dumps(state))
            if pause:(root/'runs/QUEUE_PAUSE').touch()
            queue=FakeProcess(73,root);self_process=FakeProcess(99,root)
            def process(pid=None):return self_process if pid is None else queue
            try:
                os.chdir(root)
                with patch.object(watch_training.psutil,'Process',side_effect=process),\
                     patch.object(watch_training.sys,'argv',['watch_training.py','--queue-pid','73']),\
                     patch.object(watch_training.subprocess,'run') as call:
                    watch_training.main()
                    self.assertFalse((root/'runs/post_training.lock').exists())
                    return call.call_count
            finally:os.chdir(original)

    def test_failed_queue_does_not_start_tests(self):
        self.assertEqual(self.run_case('training_failed'),0)

    def test_pause_marker_prevents_tests_even_when_queue_complete(self):
        self.assertEqual(self.run_case('queue_complete',pause=True),0)

    def test_completed_matching_queue_starts_gated_entry(self):
        self.assertEqual(self.run_case('queue_complete'),1)


if __name__=='__main__':unittest.main()
