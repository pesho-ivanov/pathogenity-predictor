"""Scientific contracts for magnitude features and matched Q12 continuation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from notebooks.src import q12, refresh_q12
try:
    import torch
    from notebooks.src import q12_backend as backend
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_scaler_is_training_only_and_constant_columns_remain_finite(self):
        training=np.array([[1.,2.,4.],[3.,2.,8.]],dtype=np.float32)
        mean,scale=q12.fit_scaler(training)
        np.testing.assert_allclose(mean,[2.,2.,6.])
        np.testing.assert_allclose(scale,[1.,1e-6,2.])
        validation=np.array([[1e9,-1e9,1e9]])
        _=(validation-mean)/scale
        np.testing.assert_array_equal(q12.fit_scaler(training)[0],mean)
        with self.assertRaisesRegex(ValueError,'Invalid'):
            q12.fit_scaler(np.array([[np.nan,1.]]))

    def test_selection_and_promotion_require_both_metrics(self):
        scores={f:{'auroc':.8,'average_precision':.7} for f in q12.CONFIG['forms']}
        self.assertEqual(q12.select_form(scores),'difference_magnitude')
        self.assertFalse(q12.promote({'auroc':.804,'average_precision':.8},scores['original']))
        self.assertFalse(q12.promote({'auroc':.81,'average_precision':.69},scores['original']))
        self.assertTrue(q12.promote({'auroc':.805,'average_precision':.7},scores['original']))

    def test_patience_uses_material_improvement(self):
        p={'monitor_auroc':.8,'bad_checks':0}
        self.assertFalse(q12.monitor(p,.801))
        self.assertFalse(q12.monitor(p,.803))
        self.assertEqual(p['bad_checks'],0)
        self.assertFalse(q12.monitor(p,.802))
        self.assertTrue(q12.monitor(p,.802))

    def test_budget_reserves_validation_and_keeps_original_deadline(self):
        b=q12.Budget(started=100.,seconds_per_variant=.05)
        self.assertAlmostEqual(b.reserve(),317.28)
        with patch.object(q12.time,'time',return_value=200.):self.assertTrue(b.can_train())
        with patch.object(q12.time,'time',return_value=3500.):self.assertFalse(b.can_train())
        with patch.object(q12.time,'time',return_value=4000.):self.assertFalse(q12.Budget(100.).can_train())

    def test_identical_paired_scores_have_zero_bootstrap_difference(self):
        y=np.array([0,1,0,1,0,1,0,1]);p=np.array([.1,.9,.2,.8,.3,.7,.4,.6])
        metrics,paired=q12.intervals(y,{'lora':p,'matched_control':p},np.array(['a','a','b','b','c','c','d','d']),repetitions=25)
        self.assertEqual(paired['auroc'],{'value':0.,'ci95':[0.,0.]})
        self.assertEqual(paired['average_precision'],{'value':0.,'ci95':[0.,0.]})
        self.assertEqual(metrics['lora']['auroc']['value'],1.)

    def test_sample_export_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError,'Sampled'):
            q12.export_full_comparison({'scope':'sampled_validation','status':'complete','validation_variants':2048})

    def test_notebooks_use_short_calls_and_separate_full_validation(self):
        for full in [False,True]:
            notebook=refresh_q12.build_notebook(full)
            cells=[c for c in notebook.cells if c.cell_type=='code']
            self.assertTrue(all(c.source.strip() and len(c.source.splitlines())<=2 for c in cells))
            self.assertTrue(all(c.execution_count is None and not c.outputs for c in cells))
            self.assertEqual(any('run_full_validation' in c.source for c in cells),full)


@unittest.skipIf(torch is None,'PyTorch is required for feature/checkpoint tests')
class FeatureAndResumeTests(unittest.TestCase):
    def test_magnitude_preserves_change_discarded_by_unit_normalization(self):
        reference=torch.tensor([[2.,-2.]])
        delta=torch.tensor([[.2,-.4]])
        a=backend.features_from_pooled(reference,delta)
        b=backend.features_from_pooled(reference,delta*10)
        torch.testing.assert_close(a[:,:4],b[:,:4])
        torch.testing.assert_close(b[:,4:]-a[:,4:],torch.full((1,2),np.log(10.),dtype=torch.float32))
        c=backend.features_from_pooled(reference*10,delta*10)
        torch.testing.assert_close(c[:,-1],a[:,-1])
        self.assertEqual(q12.columns('difference_magnitude',hidden=2).tolist(),[2,3,4,5])

    def test_zero_difference_and_zero_reference_have_finite_features_and_gradients(self):
        for ref in [[[2.,-2.]],[[0.,0.]]]:
            reference=torch.tensor(ref,requires_grad=True)
            delta=torch.zeros_like(reference,requires_grad=True)
            out=backend.features_from_pooled(reference,delta)
            self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()
            self.assertTrue(torch.isfinite(reference.grad).all())
            self.assertTrue(torch.isfinite(delta.grad).all())

    def test_matched_cursor_rejects_different_examples_or_update_counts(self):
        p={'steps':1,'control_steps':1,'offset':32,'order':np.random.default_rng(42).permutation(64).tolist()}
        backend.validate_cursor(p,64)
        for changed in [{'control_steps':0},{'offset':0},{'order':list(range(64))}]:
            with self.assertRaises(ValueError):backend.validate_cursor(p|changed,64)

    def test_checkpoint_roundtrip_rejects_corruption_identity_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            state={'format':'q12-magnitude-v1','identity':'test','master':torch.tensor([.1,.2])}
            backend.save_state(path,state)
            torch.testing.assert_close(backend.load_state(path,'test')['master'],state['master'])
            with self.assertRaisesRegex(ValueError,'identity'):backend.load_state(path,'other')
            data=path.read_bytes();path.write_bytes(data[:-1]+bytes([data[-1]^1]))
            with self.assertRaisesRegex(ValueError,'checksum'):backend.load_state(path,'test')
            backend.save_state(path,state|{'master':torch.tensor([float('nan')])})
            with self.assertRaisesRegex(ValueError,'Non-finite'):backend.load_state(path,'test')

    def test_restored_optimizer_matches_uninterrupted_training(self):
        def setup():
            torch.manual_seed(9)
            adapters=torch.nn.ModuleDict({'test':torch.nn.Linear(3,3,bias=False)})
            head,control=torch.nn.Linear(3,1),torch.nn.Linear(3,1)
            optim=backend.base.MasterAdamW(adapters,head,q12.CONFIG)
            control_optim=torch.optim.AdamW(control.parameters(),lr=q12.CONFIG['head_lr'])
            return adapters,head,control,optim,control_optim
        def step(state):
            adapters,head,control,optim,copt=state
            x=torch.tensor([[.2,.4,.6],[.7,.3,-.2]]);y=torch.tensor([1.,0.])
            torch.nn.functional.binary_cross_entropy_with_logits(head(adapters['test'](x)).flatten(),y,reduction='sum').backward()
            optim.accumulate(2);optim.step()
            torch.nn.functional.binary_cross_entropy_with_logits(control(x).flatten(),y).backward()
            copt.step();copt.zero_grad(set_to_none=True)
        original=setup();step(original)
        with patch.object(backend.base,'rng_state',return_value={'fixture':True}):
            saved=backend.make_checkpoint('id','original',torch.zeros(3),torch.ones(3),*original[:3],
                {'steps':1},original[3],original[4])
        restored=setup()
        with patch.object(backend.base,'restore_rng'):
            backend.restore_training(saved,*restored)
        step(original);step(restored)
        for old,new in zip(original[:3],restored[:3]):
            for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
