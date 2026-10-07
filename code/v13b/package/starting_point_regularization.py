"""Frozen Stage1 visual-parameter anchors; no new model parameters."""
import hashlib
import math

import torch
from torch import nn


COEFFICIENT = 1e-3
PREFIX = 'image_encoder.'


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class StartingPointRegularizer(nn.Module):
    """L_reg = coefficient/2 * SUM of visual parameter squared deviations.

    No division by batch, number of tensors, or number of elements. Classifiers,
    BN necks, prompt/text and the detached prototype memory are excluded.
    Anchors are persistent nontrainable buffers external to the learned model.
    """
    def __init__(self, model, coefficient=COEFFICIENT):
        super().__init__()
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Invalid starting-point coefficient')
        self.coefficient = float(coefficient)
        self.names = sorted(n for n,p in model.named_parameters() if n.startswith(PREFIX))
        if not self.names or PREFIX+'proj' not in self.names:
            raise ValueError('Expected CLIP visual encoder including projection')
        parameters = dict(model.named_parameters())
        self.initial_hashes = {}
        self.element_count = 0
        for index, name in enumerate(self.names):
            parameter = parameters[name]
            self.register_buffer('anchor_'+str(index), parameter.detach().float().clone())
            self.initial_hashes[name] = tensor_sha(parameter)
            self.element_count += parameter.numel()

    def _pairs(self, model):
        parameters = dict(model.named_parameters())
        for index,name in enumerate(self.names):
            parameter = parameters[name]
            anchor = getattr(self,'anchor_'+str(index))
            if parameter.shape != anchor.shape or parameter.device != anchor.device:
                raise ValueError('Anchor shape/device differs: '+name)
            yield name,parameter,anchor

    def forward(self, model):
        terms = [((p.float()-a).square().sum()) for _,p,a in self._pairs(model)]
        return .5*self.coefficient*torch.stack(terms).sum()

    def specification(self):
        return {'coefficient':self.coefficient,'reduction':'0.5*coefficient*sum_all_visual_parameter_squared_differences',
                'prefix':PREFIX,'parameter_names':self.names,'parameter_tensors':len(self.names),
                'parameter_elements':self.element_count,'trainable_parameters':0,
                'reference':'approved_strict_loaded_stage1','excludes':['classifier','bottleneck','text','prompt','memory'],
                'teacher_forward':False}

    @torch.no_grad()
    def drift(self, model):
        rows={};total=reference=0.
        for name,p,a in self._pairs(model):
            difference=float((p.float()-a).square().sum().cpu())
            squared_norm=float(a.square().sum().cpu())
            rows[name]={'squared_deviation':difference,'reference_squared_norm':squared_norm}
            total+=difference;reference+=squared_norm
        return {'squared_deviation_sum':total,'rms_deviation':math.sqrt(total/self.element_count),
                'relative_frobenius_deviation':math.sqrt(total/max(reference,1e-30)),
                'weighted_penalty':.5*self.coefficient*total,'per_parameter':rows}

    def anchor_hashes(self):
        return {name:tensor_sha(getattr(self,'anchor_'+str(i))) for i,name in enumerate(self.names)}


def compare_visual_states(initial, final):
    names=sorted(n for n in initial if n.startswith(PREFIX))
    if any(n not in final or initial[n].shape!=final[n].shape for n in names):
        raise ValueError('Visual checkpoint schema differs')
    count=sum(initial[n].numel() for n in names)
    total=sum(float((final[n].float()-initial[n].float()).square().sum()) for n in names)
    reference=sum(float(initial[n].float().square().sum()) for n in names)
    return {'squared_deviation_sum':total,'rms_deviation':math.sqrt(total/count),
            'relative_frobenius_deviation':math.sqrt(total/max(reference,1e-30)),
            'weighted_penalty':.5*COEFFICIENT*total}
