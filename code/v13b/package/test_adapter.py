"""Verify the WHU protocol and official recipe's intended adaptations."""
import argparse
from pathlib import Path
import unittest
import torch
from whu_metrics import retrieval_metrics,diagnostic_matrices


class AdapterTests(unittest.TestCase):
    def test_same_camera_negative_is_excluded(self):
        q=torch.tensor([[1.,0.]])
        g=torch.tensor([[1.,0.],[.8,.2],[0.,1.]])
        query=[('q',10,0,0)]
        gallery=[('same_camera_negative',20,0,0),('positive',10,1,0),('negative',30,2,0)]
        result=retrieval_metrics(q,g,query,gallery)
        self.assertEqual(result['r1'],1.)
        self.assertEqual(result['ap'],1.)

    def test_no_valid_positive_and_null_cells(self):
        q=torch.tensor([[1.,0.],[0.,1.]])
        g=torch.tensor([[1.,0.],[0.,1.]])
        query=[('ground',10,0,0),('aerial',10,5,0)]
        gallery=[('ground',10,1,0),('aerial',10,5,0)]
        matrices=diagnostic_matrices(q,g,query,gallery)
        self.assertIsNone(matrices['scene'][3][3])
        self.assertIsNotNone(matrices['scene'][3][0])
        with self.assertRaises(ValueError):
            retrieval_metrics(q[:1],g[:1],[('q',10,0,0)],[('g',11,1,0)])

    def test_skipped_queries_are_counted(self):
        q=torch.eye(2); g=torch.eye(2)
        r=retrieval_metrics(q,g,[('a',1,0,0),('b',2,0,1)],[('a',1,1,0),('c',3,1,1)])
        self.assertEqual((r['queries'],r['valid_queries'],r['skipped_queries']),(2,1,1))

    def test_config_official_and_budget_overrides(self):
        from run_official import config
        cfg=config(argparse.Namespace(data_root=Path('/data'),output=Path('/out'),local_smoke=False,i2t_weight=1.0))
        self.assertEqual(cfg.SOLVER.STAGE1.MAX_EPOCHS,120)
        self.assertEqual(cfg.SOLVER.STAGE2.MAX_EPOCHS,120)
        self.assertEqual(cfg.SOLVER.STAGE2.STEPS,[60,100])
        self.assertEqual(cfg.SOLVER.STAGE1.BASE_LR,.00035)
        self.assertEqual(cfg.SOLVER.STAGE2.BASE_LR,.000005)
        self.assertEqual(cfg.SOLVER.STAGE2.BIAS_LR_FACTOR,2)
        self.assertEqual(cfg.MODEL.ID_LOSS_WEIGHT,.25)
        self.assertFalse(cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW)
        self.assertEqual(cfg.MODEL.STRIDE_SIZE,[16,16])
        self.assertEqual(cfg.INPUT.PIXEL_MEAN,[.5,.5,.5])
        self.assertEqual(cfg.TEST.NECK_FEAT,'before')


if __name__=='__main__':
    torch.set_num_threads(2); unittest.main(verbosity=2)
