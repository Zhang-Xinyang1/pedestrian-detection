"""Flat three-modality WHU dataset plus the shared mixed-gallery protocol."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from whu_metrics import retrieval_metrics, diagnostic_matrices

MODALITIES=('RGB','IR','Thermal')
PATTERN=re.compile(r'^(\d+)_c(\d+)_m(\d+)_')


class WHUMARS:
    def __init__(self,root,**kwargs):
        self.root=Path(root)
        original={}
        for split in ['train','query','test']:
            rows=[]
            for modality,folder in enumerate(MODALITIES):
                paths=sorted((self.root/'WHU-MARS'/split/folder).glob('*.jpg'))
                if not paths: raise ValueError('Missing split/modality: '+split+'/'+folder)
                for p in paths:
                    match=PATTERN.match(p.name)
                    if match is None: raise ValueError('Unexpected filename: '+p.name)
                    pid,camera,mid=map(int,match.groups())
                    if mid!=modality+1 or not 1<=camera<=7: raise ValueError('Invalid WHU metadata')
                    rows.append((str(p),pid,camera-1,modality))
            original[split]=rows
        train_ids=sorted({r[1] for r in original['train']})
        test_ids={r[1] for r in original['query']}|{r[1] for r in original['test']}
        if set(train_ids)&test_ids: raise ValueError('Train/test identity leakage')
        mapping={pid:i for i,pid in enumerate(train_ids)}
        self.train_meta=[(p,mapping[pid],cam,mod) for p,pid,cam,mod in original['train']]
        self.query_meta=original['query']; self.gallery_meta=original['test']
        self.num_train_pids=len(train_ids)
        # Fourth field is physical view for official loader; modality stays in metadata.
        convert=lambda rows:[(p,pid,cam,int(cam>=5)) for p,pid,cam,mod in rows]
        self.train=convert(self.train_meta); self.query=convert(self.query_meta); self.gallery=convert(self.gallery_meta)
        self.num_train_cams=len({r[2] for r in self.train})
        self.num_train_vids=len({r[3] for r in self.train})
        self.train_signature=hashlib.sha256(json.dumps([(Path(p).name,y,c,m) for p,y,c,m in self.train_meta]).encode()).hexdigest()


def evaluator_class(dataset,output,eval_period,stage2_epochs,smoke=False):
    class WHUEvaluator:
        def __init__(self,num_query,max_rank=50,feat_norm=True,reranking=False):
            if reranking or not feat_norm: raise ValueError('Fixed normalized/no-reranking protocol')
            assert num_query==len(dataset.query_meta)
            self.num_query=num_query; self.counter=0; self.reset()

        def reset(self):
            self.feats=[]; self.pids=[]; self.cams=[]

        def update(self,values):
            features,pids,cams=values
            if features.ndim!=2 or features.shape[1]!=1280 or not torch.isfinite(features).all():
                raise ValueError('Expected finite official 1280-D retrieval features')
            self.feats.append(features.detach().float().cpu())
            self.pids.extend(int(v) for v in pids); self.cams.extend(int(v) for v in cams)

        def compute(self):
            rows=dataset.query_meta+dataset.gallery_meta
            if self.pids!=[r[1] for r in rows] or self.cams!=[r[2] for r in rows]:
                raise ValueError('Evaluation order/metadata differs')
            f=torch.cat(self.feats); qf=f[:self.num_query]; gf=f[self.num_query:]
            result=retrieval_metrics(qf,gf,dataset.query_meta,dataset.gallery_meta)
            result['matrices']=diagnostic_matrices(qf,gf,dataset.query_meta,dataset.gallery_meta)
            self.counter+=1; epoch=self.counter*eval_period
            result.update(epoch=epoch,protocol='WHU_all_same_camera_excluded',feature_dim=1280,local_smoke=smoke)
            (output/f'eval_stage2_{epoch:03d}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
            print('OFFICIAL_CLIPREID_EVAL_OK '+json.dumps({k:v for k,v in result.items() if k!='matrices'}),flush=True)
            cmc=np.zeros(50,dtype=np.float64)
            cmc[0]=result['r1']; cmc[4]=result['r5']; cmc[9]=result['r10']
            # Official stage2 only reads those three ranks; no dense distance allocation.
            return cmc,result['ap'],None,self.pids,self.cams,qf,gf
    return WHUEvaluator
