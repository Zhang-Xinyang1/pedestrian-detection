"""Shared identity context with fixed modality/view language for official CLIP-ReID."""
from pathlib import Path
import sys
import torch
from torch import nn

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'upstream'))
from model.make_model_clipreid import build_transformer, clip
from quality_gate import ReliabilityGate

MODALITIES=('visible light','near infrared','thermal infrared')
VIEWS=('ground-level','aerial')
SCENES=('G-RGB','G-NIR','G-TIR','A-RGB','A-NIR','A-TIR')


def templates(mode):
    if mode not in ('neutral','modality','view','both'):
        raise ValueError('Unknown prompt mode')
    result=[]
    for scene in range(6):
        prefix='A photo of a X X X X person'
        if mode in ('modality','both'):
            prefix+=' captured by a '+MODALITIES[scene%3]+' camera'
        if mode in ('view','both'):
            prefix+=' from an '+VIEWS[scene//3]+' viewpoint' if scene//3 else ' from a '+VIEWS[0]+' viewpoint'
        result.append(prefix+'.')
    return result


def validate_scene(scene, batch, device):
    if scene is None:
        raise ValueError('Scene metadata is required for text prompts')
    scene=torch.as_tensor(scene,device=device)
    if scene.dtype==torch.bool or scene.is_complex() or (scene.is_floating_point() and not bool((torch.isfinite(scene)&(scene==scene.round())).all())):
        raise ValueError('Scene indices must be finite integers')
    scene=scene.long()
    if scene.ndim!=1 or len(scene)!=batch or bool(((scene<0)|(scene>=6)).any()):
        raise ValueError('Expected scene vector in [0,5] matching the batch')
    return scene


class ScenePromptLearner(nn.Module):
    def __init__(self, original, embedding_weight, mode):
        super().__init__()
        self.mode=mode
        # Reuse the exact initialized official identity token Parameter, with
        # no scene-specific identity parameters or extra random draws.
        self.cls_ctx=original.cls_ctx
        self.num_class=original.num_class
        self.n_cls_ctx=original.n_cls_ctx
        self.register_buffer('token_prefix',original.token_prefix.detach().clone())
        self.register_buffer('token_suffix',original.token_suffix.detach().clone())
        texts=templates(mode)
        ids=clip.tokenize(texts)
        neutral=clip.tokenize('A photo of a X X X X person.')
        if not torch.equal(ids[:,:9],neutral[:,:9].expand(6,-1)):
            raise ValueError('Template changed the official four identity token positions')
        if not bool((ids.argmax(-1)>9).all()):
            raise ValueError('Prompt EOT is before the learned context ends')
        with torch.no_grad():
            embeddings=nn.functional.embedding(ids,embedding_weight.cpu()).to(
                device=self.cls_ctx.device,dtype=self.cls_ctx.dtype)
        self.register_buffer('scene_token_ids',ids.to(self.cls_ctx.device))
        self.register_buffer('scene_prefix',embeddings[:,:5,:].clone())
        self.register_buffer('scene_suffix',embeddings[:,9:,:].clone())
        if mode=='neutral':
            if not torch.equal(self.scene_prefix[0].cpu(),self.token_prefix[0].cpu()) or not torch.equal(self.scene_suffix[0].cpu(),self.token_suffix[0].cpu()):
                raise ValueError('Neutral prompt embedding differs from official')

    def forward(self,labels,scene):
        scene=validate_scene(scene,len(labels),self.cls_ctx.device)
        contexts=self.cls_ctx[labels]
        return torch.cat((self.scene_prefix[scene],contexts,self.scene_suffix[scene]),dim=1),self.scene_token_ids[scene]

    def specification(self):
        return dict(mode=self.mode,templates=templates(self.mode),scene_order=list(SCENES),
            identity_tokens=4,shared_across_scenes=True,learned_scene_tokens=False,
            text_feature_normalization='unchanged official unnormalized dot product',
            inference='visual only, official 1280-D concat then L2')


class ScenePromptModel(build_transformer):
    def __init__(self, *args, quality_gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_gate_enabled = bool(quality_gate)
        self.reliability_gate = ReliabilityGate() if self.quality_gate_enabled else None

    def forward(self,x=None,label=None,get_image=False,get_text=False,cam_label=None,view_label=None,scene=None):
        if get_text:
            prompts,token_ids=self.prompt_learner(label,scene)
            return self.text_encoder(prompts,token_ids)
        # No scene metadata is injected into the image encoder or retrieval path.
        result = super().forward(x=x,label=label,get_image=get_image,get_text=False,
            cam_label=cam_label,view_label=view_label)
        # Stage 1 uses get_image=True and must remain the exact modality
        # baseline.  In Stage 2, scale only the existing projected branch.
        if self.quality_gate_enabled and not get_image and self.training:
            scores, feats, image_features = result
            alpha = self.reliability_gate(x)
            image_features = image_features * alpha[:, None]
            feats = list(feats)
            feats[-1] = image_features
            result = (scores, feats, image_features)
        elif self.quality_gate_enabled and not get_image and not self.training:
            # Retrieval must use the same learned gate, while preserving the
            # official 768+512 concatenation and final normalization.
            alpha = self.reliability_gate(x)
            result = result.clone()
            result[:, -512:] = result[:, -512:] * alpha[:, None]
        return result


def attach_scene_prompts(model,weights,mode):
    checkpoint=torch.jit.load(str(weights),map_location='cpu').eval()
    embedding=checkpoint.state_dict()['token_embedding.weight'].detach().float().clone()
    del checkpoint
    model.prompt_learner=ScenePromptLearner(model.prompt_learner,embedding,mode)


@torch.no_grad()
def build_text_banks(model,classes,chunk):
    device=next(model.parameters()).device
    texts=templates(model.prompt_learner.mode)
    unique=list(dict.fromkeys(texts))
    representatives=[texts.index(text) for text in unique]
    scene_to_bank=torch.tensor([unique.index(text) for text in texts],device=device)
    banks=[]
    for scene in representatives:
        features=[]
        for start in range(0,classes,chunk):
            labels=torch.arange(start,min(start+chunk,classes),device=device)
            scenes=torch.full_like(labels,scene)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                features.append(model(label=labels,get_text=True,scene=scenes))
        banks.append(torch.cat(features))
    return dict(features=torch.stack(banks),scene_to_bank=scene_to_bank)


def matched_scene_logits(images,banks,scene):
    scene=validate_scene(scene,len(images),images.device)
    features=banks['features']
    if features.shape[0]==1:
        return images@features[0].t()  # Exact official neutral matrix multiply.
    indices=banks['scene_to_bank'][scene]
    # Compute full-batch GEMMs, then select each row's scene bank. Keeping the
    # same operation shape avoids tiny mixed-scene sub-batches under AMP.
    all_logits=torch.stack([images@bank.t() for bank in features],dim=1)
    return all_logits[torch.arange(len(images),device=images.device),indices]
