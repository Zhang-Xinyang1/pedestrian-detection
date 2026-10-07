"""Check that only the intended loss coefficient changes the official recipe."""
import argparse
from pathlib import Path
import unittest
import torch
import yaml
from run_official import config
from loss.make_loss import make_loss
from loss.softmax_loss import CrossEntropyLabelSmooth


def recipe(weight):
    return config(argparse.Namespace(data_root=Path('/data'),output=Path('/out'),
                                     local_smoke=False,i2t_weight=weight))


class I2TTests(unittest.TestCase):
    def test_only_loss_coefficient_changes(self):
        base=yaml.safe_load(recipe(1.).dump())
        for weight in (.25,.5,1.):
            actual=yaml.safe_load(recipe(weight).dump())
            self.assertEqual(actual['MODEL'].pop('I2T_LOSS_WEIGHT'),weight)
            expected=yaml.safe_load(recipe(1.).dump())
            expected['MODEL'].pop('I2T_LOSS_WEIGHT')
            self.assertEqual(actual,expected)

    def test_invalid_weights_rejected(self):
        for weight in (-1.,0.,float('nan'),float('inf'),2.):
            with self.assertRaises(ValueError): recipe(weight)

    @unittest.skipUnless(torch.cuda.is_available(),'Official loss constructor requires CUDA')
    def test_actual_loss_value_and_logit_gradient(self):
        torch.manual_seed(12)
        labels=torch.tensor([0,0,1,1,2,2,3,3],device='cuda')
        scores=[torch.randn(8,500,device='cuda') for _ in range(2)]
        features=[torch.randn(8,d,device='cuda') for d in (768,512)]
        logits=torch.randn(8,500,device='cuda',requires_grad=True)
        ce=CrossEntropyLabelSmooth(num_classes=500)(logits,labels)
        ce_grad=torch.autograd.grad(ce,logits,retain_graph=True)[0]
        base_loss=None
        for weight in (.25,.5,1.):
            loss_fn,_=make_loss(recipe(weight),500)
            visual=loss_fn(scores,features,labels,None)
            if base_loss is not None: torch.testing.assert_close(visual,base_loss)
            base_loss=visual
            total=loss_fn(scores,features,labels,None,logits)
            torch.testing.assert_close(total,visual+weight*ce)
            grad=torch.autograd.grad(total,logits,retain_graph=True)[0]
            torch.testing.assert_close(grad,weight*ce_grad)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
