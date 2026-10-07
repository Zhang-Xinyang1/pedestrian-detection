"""Import the complete real training entry, then exercise tokenizer and CUDA ops."""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--local-validation',action='store_true')
    args=parser.parse_args()
    sys.path[:0]=[str(args.package),str(args.package/'upstream')]
    roots=['numpy','PIL.Image','yaml','yacs.config','ftfy','regex','wcwidth','tqdm',
           'huggingface_hub','safetensors','torch','torchvision',
           'timm.data.random_erasing','timm.models.layers',
           'run_stage2_only','verify_results_csf','l2sp_diagnostics','local_observed_auxiliary','verify_local_auxiliary','partial_local_matching']
    failures={}
    for name in roots:
        try:importlib.import_module(name)
        except Exception as error:failures[name]=f'{type(error).__name__}: {error}'
    if failures:
        raise RuntimeError('Full import check failed: '+json.dumps(failures))
    import torch
    import torchvision
    from model.clip.simple_tokenizer import SimpleTokenizer
    from torchvision.ops import nms
    from solver.lr_scheduler import WarmupMultiStepLR
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required'
    gpu=torch.cuda.get_device_name(0)
    if not args.local_validation:
        assert sys.version_info[:2]==(3,12),'This wheelhouse targets Python3.12'
        assert torch.__version__.split('+')[0]=='2.6.0','Use the confirmed PyTorch2.6.0 CUDA12.4 image'
        assert torchvision.__version__.split('+')[0]=='0.21.0','Expected torchvision0.21.0 paired with torch2.6.0'
        assert 'A800' in gpu,'One A800 GPU required'
    token=SimpleTokenizer().encode('A photo of a person captured by a thermal infrared camera.')
    assert token and all(isinstance(v,int) for v in token)
    boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]],device='cuda')
    keep=nms(boxes,torch.tensor([1.,.5],device='cuda'),.5)
    assert keep.tolist()==[0],'torch/torchvision CUDA binary mismatch'
    def lr_sequence(milestones):
        p=torch.nn.Parameter(torch.zeros(1))
        opt=torch.optim.Adam([p],lr=5e-6)
        schedule=WarmupMultiStepLR(opt,milestones,.1,.1,10,'linear')
        rows=[]
        for epoch in range(1,101):
            schedule.step()
            rows.append(opt.param_groups[0]['lr'])
            p.sum().backward();opt.step();opt.zero_grad()
        return rows
    before=lr_sequence([60,100]);after=lr_sequence([21])
    assert before[:20]==after[:20]
    assert all(abs(lr-5e-7)<1e-15 for lr in after[20:])
    record={'status':'PASS','python':sys.version,'torch':torch.__version__,
            'torchvision':torchvision.__version__,'gpu':gpu,'imports':roots,
            'tokenizer':'PASS','cuda_nms':'PASS','learning_rate':after,
            'declared_budget':100,'decay_actual_epoch':21,'first20_lr_equal':True,
            'local_validation':args.local_validation,
            'versions':{name:metadata.version(name) for name in
                       ['timm','yacs','PyYAML','ftfy','regex','numpy','Pillow','huggingface-hub','safetensors']}}
    args.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('SCNET_FULL_ENVIRONMENT_PASS '+json.dumps({k:v for k,v in record.items() if k not in ('imports','learning_rate')}),flush=True)


if __name__=='__main__':main()
