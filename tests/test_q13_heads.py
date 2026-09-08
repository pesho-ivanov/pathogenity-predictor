"""CPU contracts for the Q13 converged linear control."""

import unittest
import numpy as np

try:
    import torch
    from notebooks.src import q13_heads
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for linear-head fitting')
class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 4)).astype(np.float32)
        x[:, 3] = 3.
        y = (x[:, 0] - .4 * x[:, 1] + rng.normal(size=80) > 0).astype(int)
        return x[:60], y[:60], x[60:], y[60:]

    def test_convergence_determinism_and_training_only_scaling(self):
        x, y, v, vy = self.data()
        first = q13_heads.fit_heads(x, y, v, vy, strengths=(.01, .1))
        second = q13_heads.fit_heads(x, y, v, vy, strengths=(.01, .1))
        for key in ['weight', 'bias', 'mean', 'scale', 'class_weights']:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertEqual(first[key].device.type, 'cpu')
            self.assertEqual(first[key].dtype, torch.float32)
        self.assertEqual(first['candidates'], second['candidates'])
        self.assertEqual(first['weight'].shape, (1, x.shape[1]))
        self.assertEqual(first['bias'].shape, (1,))
        self.assertTrue(all(c['converged'] for c in first['candidates']))
        self.assertTrue(all(c['gradient_norm'] <= 1e-6 for c in first['candidates']))
        self.assertAlmostEqual(float(first['scale'][-1]), 1e-6, places=12)
        changed = q13_heads.fit_heads(x, y, v + 100., vy, strengths=(.01, .1))
        for key in ['mean', 'scale', 'class_weights']:
            torch.testing.assert_close(first[key], changed[key], atol=0, rtol=0)
        for a, b in zip(first['candidates'], changed['candidates']):
            self.assertEqual(a['weighted_bce'], b['weighted_bce'])
            self.assertEqual(a['gradient_norm'], b['gradient_norm'])
        torch.testing.assert_close(first['class_weights'],
            torch.tensor(len(y) / (2 * np.bincount(y)), dtype=torch.float32))

    def test_fp64_convergence_and_fp32_deployment_are_reported_separately(self):
        x, y, v, vy = self.data()
        fitted = q13_heads.fit_heads(x, y, v, vy, strengths=(.1,), tolerance=1e-9)
        candidate = fitted['candidates'][0]
        self.assertTrue(candidate['converged'])
        self.assertLessEqual(candidate['fitted_gradient_norm'], 1e-9)
        self.assertGreater(candidate['deployed_gradient_norm'], 1e-9)
        self.assertTrue(candidate['deployment_logits_verified'])
        self.assertLess(candidate['max_training_logit_cast_difference'], 2e-4)
        head = torch.nn.Linear(x.shape[1], 1)
        head.load_state_dict({key: fitted[key] for key in ['weight', 'bias']})
        standardized = (torch.from_numpy(x) - fitted['mean']) / fitted['scale']
        loss = q13_heads.objective(head.weight.flatten(), head.bias, standardized,
            torch.tensor(y, dtype=torch.float32), fitted['class_weights'][y], .1)
        grads = torch.autograd.grad(loss, tuple(head.parameters()))
        norm = float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads])))
        self.assertAlmostEqual(norm, candidate['deployed_gradient_norm'], places=12)
        with torch.no_grad():
            scores = head((torch.from_numpy(v) - fitted['mean']) / fitted['scale']).flatten().numpy()
        self.assertEqual(candidate['auroc'], q13_heads.roc_auc_score(vy, scores))
        self.assertEqual(candidate['average_precision'], q13_heads.average_precision_score(vy, scores))

    def test_objective_penalizes_weights_but_not_bias(self):
        x = torch.zeros(4, 2)
        y = torch.tensor([0., 1., 0., 1.])
        weights = torch.tensor([2., -3.], requires_grad=True)
        bias = torch.tensor([1.], requires_grad=True)
        sample_weights = torch.tensor([.5, 1.5, .5, 1.5])
        unpenalized = q13_heads.objective(weights, bias, x, y, sample_weights, 0.)
        penalized = q13_heads.objective(weights, bias, x, y, sample_weights, .1)
        self.assertAlmostEqual(float((penalized - unpenalized).detach()), .65, places=6)
        first_bias_gradient = torch.autograd.grad(unpenalized, bias, retain_graph=True)[0]
        weight_gradient, bias_gradient = torch.autograd.grad(penalized, (weights, bias))
        torch.testing.assert_close(weight_gradient, .1 * weights)
        torch.testing.assert_close(bias_gradient, first_bias_gradient)

    def test_selection_excludes_unconverged_fits_and_prefers_stronger_ties(self):
        candidate = {'auroc': .8, 'average_precision': .7, 'strength': .01,
                     'converged': True, 'gradient_norm': 1e-7}
        candidates = [candidate, candidate | {'strength': .1},
                      candidate | {'auroc': 1., 'converged': False}]
        self.assertEqual(q13_heads.select_candidate(candidates)['strength'], .1)
        with self.assertRaisesRegex(RuntimeError, 'No linear head converged'):
            q13_heads.select_candidate([candidate | {'converged': False}])

    def test_iteration_limit_does_not_count_as_convergence(self):
        with self.assertRaisesRegex(RuntimeError, 'No linear head converged'):
            q13_heads.fit_heads(*self.data(), strengths=(.01,), max_iter=1, tolerance=1e-12)

    def test_bad_inputs_fail_before_fitting(self):
        x, y, v, vy = self.data()
        bad = x.copy(); bad[0, 0] = np.nan
        for args in [(bad, y, v, vy), (x, y[:, None], v, vy),
                     (x, y, v[:, :-1], vy), (x, y * 0, v, vy),
                     (x[:0], y[:0], v, vy)]:
            with self.subTest(shapes=[a.shape for a in args]), self.assertRaises(ValueError):
                q13_heads.fit_heads(*args)
        for kwargs in [{'strengths': (0.,)}, {'strengths': (.1, .1)},
                       {'strengths': (float('inf'),)}, {'max_iter': 0},
                       {'tolerance': float('nan')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                q13_heads.fit_heads(x, y, v, vy, **kwargs)


if __name__ == '__main__':
    unittest.main()
