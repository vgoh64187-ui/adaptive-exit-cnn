import unittest
from paired_effects import compare, CONTRASTS


class PairedEffectsTests(unittest.TestCase):
    def rows(self):
        rows=[]
        methods=sorted({m for _,a,b in CONTRASTS for m in [a,b]})
        for head,kd,consensus in methods:
            for budget in [.55,.65,.75,.85,.95,1.01]:
                for seed in [17,42,73]:
                    # Deliberately different actual costs: preserve, do not match away.
                    rows.append(dict(run=f'{head}_{kd}_{seed}',head=head,kd=kd,consensus=consensus,
                        criterion='budget',budget=budget,seed=seed,accuracy=.7+kd*.02,
                        cost_ratio=budget-.01+kd*.01,harmful_early_exit_fraction=.04,
                        beneficial_early_exit_fraction=.01))
        return rows

    def test_pairs_preserve_actual_cost_difference(self):
        rows=self.rows();result=compare(list(reversed(rows)))
        effect=next(r for r in result if r['contrast']=='distillation')
        self.assertAlmostEqual(effect['summary']['accuracy_delta_pp_mean'],1.)
        self.assertAlmostEqual(effect['summary']['cost_ratio_delta_pp_mean'],.5)
        self.assertEqual([p['seed'] for p in effect['pairs']],[17,42,73])

    def test_missing_seed_is_not_dropped_silently(self):
        with self.assertRaisesRegex(RuntimeError,'Missing paired'):
            compare(self.rows()[1:])


if __name__=='__main__':unittest.main()
