import unittest
import torch
from quality_gate import image_quality_features, ReliabilityGate


class QualityGateTests(unittest.TestCase):
    def test_features_finite_and_bounded_shape(self):
        x=torch.randn(5,3,32,32)
        f=image_quality_features(x)
        self.assertEqual(tuple(f.shape),(5,6))
        self.assertTrue(torch.isfinite(f).all())

    def test_zero_init_is_identity(self):
        g=ReliabilityGate()
        x=torch.randn(5,3,32,32)
        self.assertTrue(torch.equal(g(x),torch.ones(5)))

    def test_gate_has_gradient(self):
        g=ReliabilityGate()
        x=torch.randn(5,3,32,32)
        loss=g(x).sum()
        loss.backward()
        self.assertIsNotNone(g.proj.weight.grad)


if __name__=='__main__': unittest.main(verbosity=2)
