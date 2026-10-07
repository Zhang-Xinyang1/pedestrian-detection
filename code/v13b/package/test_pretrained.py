"""GPU contract test with the actual pretrained CLIP (not a metric experiment)."""
import argparse
import json
from pathlib import Path
import torch
from run_official import config,models
from scene_prompt import ScenePromptModel,attach_scene_prompts,templates


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    cfg=config(argparse.Namespace(data_root=Path('.'),output=Path('.'),local_smoke=True))
    models.clip._download=lambda *args,**kwargs:str(a.weights)
    model=ScenePromptModel(500,6,2,cfg).cuda().eval()
    identity=model.prompt_learner.cls_ctx
    count=sum(p.numel() for p in model.parameters())
    labels=torch.full((6,),7,device='cuda',dtype=torch.long);scenes=torch.arange(6,device='cuda')
    images=torch.randn(2,3,256,128,device='cuda')
    with torch.no_grad(),torch.autocast('cuda'):
        reference_text=models.build_transformer.forward(model,label=labels,get_text=True)
        reference_visual=model(images)
    report={}
    for mode in ['neutral','modality','view','both']:
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        attach_scene_prompts(model,a.weights,mode);model.cuda().eval()
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        assert model.prompt_learner.cls_ctx is identity
        assert sum(p.numel() for p in model.parameters())==count
        with torch.no_grad(),torch.autocast('cuda'):
            text=model(label=labels,get_text=True,scene=scenes)
            visual=model(images)
        assert torch.isfinite(text).all() and torch.equal(visual,reference_visual)
        if mode=='neutral':assert torch.equal(text,reference_text)
        wording=templates(mode)
        for i in range(6):
            for j in range(6):
                assert torch.equal(text[i],text[j])==(wording[i]==wording[j])
        report[mode]=dict(unique_templates=len(set(wording)),unique_embeddings=len(torch.unique(text,dim=0)),
            text_change_l2_from_neutral=(text.float()-reference_text.float()).norm(dim=-1).cpu().tolist(),
            visual_identical=True,parameter_count=count,shared_identity_parameter=True)
    print('SCENE_PROMPT_PRETRAINED_OK '+json.dumps(report),flush=True)


if __name__=='__main__':main()
