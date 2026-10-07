"""Prompt token placement, sharing, routing and gradient correctness."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent/'upstream'))
from scene_prompt import templates,ScenePromptLearner,matched_scene_logits,clip


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(6)
        cls.embedding=torch.randn(49408,8)
        ids=clip.tokenize('A photo of a X X X X person.')
        e=nn.functional.embedding(ids,cls.embedding)
        cls.original=SimpleNamespace(cls_ctx=nn.Parameter(torch.randn(5,4,8)),num_class=5,n_cls_ctx=4,
            token_prefix=e[:,:5],token_suffix=e[:,9:])

    def test_neutral_is_official_embedding_and_shared_parameter(self):
        learner=ScenePromptLearner(self.original,self.embedding,'neutral')
        labels=torch.tensor([0,2,2]);scenes=torch.tensor([0,3,5])
        actual,_=learner(labels,scenes)
        expected=torch.cat((self.original.token_prefix.expand(3,-1,-1),self.original.cls_ctx[labels],self.original.token_suffix.expand(3,-1,-1)),dim=1)
        self.assertTrue(torch.equal(actual,expected))
        self.assertIs(learner.cls_ctx,self.original.cls_ctx)
        self.assertEqual([n for n,p in learner.named_parameters()],['cls_ctx'])

    def test_token_positions_lengths_and_eot_for_all_modes(self):
        for mode,count in [('neutral',1),('modality',3),('view',2),('both',6)]:
            learner=ScenePromptLearner(self.original,self.embedding,mode)
            prompts,ids=learner(torch.zeros(6,dtype=torch.long),torch.arange(6))
            self.assertEqual(prompts.shape,(6,77,8))
            self.assertEqual(len(set(templates(mode))),count)
            self.assertTrue(torch.equal(prompts[:,5:9],self.original.cls_ctx[0].expand(6,-1,-1)))
            self.assertTrue(torch.equal(prompts[torch.arange(6),ids.argmax(1)],self.embedding[ids.max(1).values]))

    def test_gradients_accumulate_in_shared_identity_only(self):
        learner=ScenePromptLearner(self.original,self.embedding,'both')
        prompts,_=learner(torch.tensor([2,2]),torch.tensor([0,5]))
        grad=torch.autograd.grad(prompts.sum(),learner.cls_ctx)[0]
        self.assertTrue(torch.equal(grad[2],torch.full_like(grad[2],2.)))
        self.assertEqual(grad[[0,1,3,4]].abs().sum(),0)
        self.assertTrue(all(not b.requires_grad for b in learner.buffers()))

    def test_scene_matched_logits_and_gradient_use_all_classes(self):
        images=torch.randn(6,8,requires_grad=True);text=torch.randn(6,5,8)
        scenes=torch.tensor([5,0,2,3,1,4])
        actual=matched_scene_logits(images,dict(features=text,scene_to_bank=torch.arange(6)),scenes)
        expected=torch.stack([images[i]@text[s].t() for i,s in enumerate(scenes)])
        torch.testing.assert_close(actual,expected)
        self.assertEqual(actual.shape,(6,5))
        ga=torch.autograd.grad(actual.sum(),images,retain_graph=True)[0]
        gb=torch.autograd.grad(expected.sum(),images)[0]
        torch.testing.assert_close(ga,gb)

    def test_neutral_logits_exact(self):
        x=torch.randn(7,8);text=torch.randn(1,5,8)
        actual=matched_scene_logits(x,dict(features=text,scene_to_bank=torch.zeros(6,dtype=torch.long)),torch.arange(7)%6)
        self.assertTrue(torch.equal(actual,x@text[0].t()))

    def test_no_mutation_or_random_draw(self):
        learner=ScenePromptLearner(self.original,self.embedding,'both')
        before=self.original.cls_ctx.detach().clone();rng=torch.get_rng_state().clone()
        learner(torch.tensor([0,1]),torch.tensor([0,5]))
        self.assertTrue(torch.equal(before,learner.cls_ctx))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_invalid_scene_rejected(self):
        learner=ScenePromptLearner(self.original,self.embedding,'both')
        for scenes in (None,torch.tensor([-1]),torch.tensor([6]),torch.tensor([0,1]),torch.tensor([1.5]),torch.tensor([float('nan')]),torch.tensor([True])):
            with self.assertRaises(ValueError):learner(torch.tensor([0]),scenes)


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main(verbosity=2)
