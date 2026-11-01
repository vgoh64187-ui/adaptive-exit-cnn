"""Prove that test evaluation cannot silently drift from pre-test artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from freeze_protocol import build_lock
from protocol import digest, validate_frozen_run
from train import source_hash


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.jobs=[]
        for seed in [17,42,73]:
            job=dict(name=f'baseline_seed{seed}',head='baseline',kd=0.,seed=seed)
            self.jobs.append(job)
            run=self.root/job['name'];run.mkdir()
            config=dict(head='baseline',kd=0.,seed=seed,epochs=2,batch_size=128,workers=2,
                        width=64,lr=.1,weight_decay=.0005,temperature=3.,limit=0,
                        smoke_only=False,source_sha256=source_hash(),split_sha256='fixture_split')
            (run/'config.json').write_text(json.dumps(config))
            (run/'best.pt').write_bytes(b'unit-test artifact; not a trained model')
            (run/'completed.json').write_text(json.dumps(dict(epochs=2,best_validation_score=.75)))
            (run/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i)) for i in [1,2]))
            (run/'validation_predictions.npz').write_bytes(b'unit-test cache fixture')
            (run/'policies.json').write_text(json.dumps(dict(weights_sha256=digest(run/'best.pt'),
                split_sha256='fixture_split',smoke_only=False,policies=[dict(criterion='full_depth')])))
            (run/'validation_predictions.metadata.json').write_text(json.dumps(dict(
                weights_sha256=digest(run/'best.pt'),prediction_sha256=digest(run/'validation_predictions.npz'),
                split_sha256='fixture_split',n=5000,split='validation',smoke_only=False)))
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(runs=self.jobs,epochs=2,batch_size=128,
                                                workers=2,minimum_baseline_validation=.70)))

    def test_freeze_valid_artifacts_then_reject_changed_policy(self):
        lock=build_lock(self.manifest,self.root)
        run=self.root/'baseline_seed42'
        config=json.loads((run/'config.json').read_text())
        validate_frozen_run(run,config,lock)
        (run/'policies.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):
            validate_frozen_run(run,config,lock)

    def test_reject_changed_evaluator(self):
        lock=build_lock(self.manifest,self.root)
        lock['evaluation_sources']['evaluate.py']='changed'
        run=self.root/'baseline_seed42'
        with self.assertRaisesRegex(RuntimeError,'evaluation source changed'):
            validate_frozen_run(run,json.loads((run/'config.json').read_text()),lock)

    def test_reject_incomplete_history_and_prior_test_access(self):
        run=self.root/'baseline_seed42'
        (run/'history.jsonl').write_text('{"epoch": 1}\n')
        with self.assertRaisesRegex(RuntimeError,'epoch history'):
            build_lock(self.manifest,self.root)
        (run/'test_predictions.npz').write_bytes(b'previous test')
        with self.assertRaisesRegex(RuntimeError,'already exist'):
            build_lock(self.manifest,self.root)

    def test_reject_corrupt_validation_cache(self):
        (self.root/'baseline_seed73'/'validation_predictions.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'prediction cache'):
            build_lock(self.manifest,self.root)


if __name__=='__main__':unittest.main()
