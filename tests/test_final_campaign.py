"""Exercise the final campaign sequence using only isolated stub artifacts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from protocol import digest
import run_final_tests


class FinalCampaignTests(unittest.TestCase):
    def test_interrupted_campaign_resumes_with_frozen_lock_and_audits(self):
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for p in ['runs/a','runs/b','runs/c','docs','configs']:(root/p).mkdir(parents=True,exist_ok=True)
            manifest=root/'configs/main_queue.json';manifest.write_text(json.dumps({'runs':[{'name':x} for x in 'abc']}))
            (root/'runs/queue_state.json').write_text(json.dumps({'status':'queue_complete','manifest_sha256':digest(manifest)}))
            for x in 'abc':(root/'runs'/x/'config.json').write_text('{}')
            calls=[];fail_once=[True]
            def fake_run(command,**kwargs):
                action=command[1];calls.append(action)
                if action=='freeze_protocol.py':
                    (root/'docs/protocol_lock.json').write_text(json.dumps({'manifest_sha256':digest(manifest),'runs':{x:{} for x in 'abc'}}))
                elif action=='evaluate.py':
                    name=Path(command[2]).name
                    if name=='b' and fail_once[0]:
                        fail_once[0]=False;raise subprocess.CalledProcessError(1,command)
                    (root/'runs'/name/'test_metrics.json').write_text('{}')
            def fake_audit(run,lock):
                calls.append('audit:'+Path(run).name)
                self.assertTrue((root/lock).exists())
                return {'policies_checked':1}
            try:
                os.chdir(root)
                with patch.object(run_final_tests.sys,'argv',['run_final_tests.py']),\
                     patch.object(run_final_tests.subprocess,'run',side_effect=fake_run),\
                     patch.object(run_final_tests,'validate_frozen_run'),\
                     patch.object(run_final_tests,'audit_run',side_effect=fake_audit):
                    with self.assertRaises(subprocess.CalledProcessError):run_final_tests.main()
                    self.assertTrue((root/'docs/protocol_lock.json').exists())
                    self.assertTrue((root/'runs/a/test_metrics.json').exists())
                    self.assertNotIn('summarise.py',calls)
                    calls.clear();run_final_tests.main()
                    self.assertNotIn('freeze_protocol.py',calls)
                    self.assertEqual(calls.count('evaluate.py'),2)
                    self.assertEqual([x for x in calls if x.startswith('audit:')],['audit:a','audit:b','audit:c'])
                    self.assertEqual(calls[-2:],['summarise.py','paired_effects.py'])
            finally:os.chdir(original)


if __name__=='__main__':unittest.main()
