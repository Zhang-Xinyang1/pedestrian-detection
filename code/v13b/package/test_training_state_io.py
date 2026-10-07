"""Checkpoint restore reproduces the next stochastic optimizer update."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import random
import torch
from lr_tail_schedule import MatchedTailScheduler
from training_state_io import save_training_state,restore_training_state

class Config:
    def dump(self):return 'unit_restore_test'

def main(device):
    torch.manual_seed(83);random.seed(83);np.random.seed(83)
    model=torch.nn.Linear(3,2).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
    center=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1,device=device))],lr=.5)
    scheduler=MatchedTailScheduler(optimizer,[21],.1,.1,10,'linear',arm='B',tail_start=2,tail_end=4)
    scaler=torch.cuda.amp.GradScaler(enabled=device=='cuda')
    memory=torch.nn.Linear(2,2).to(device);aux=torch.nn.Linear(2,2).to(device)
    def step():
        scheduler.step();optimizer.zero_grad()
        x=torch.randn(5,3,device=device);target=torch.randn(5,2,device=device)
        loss=(model(x)-target).square().mean();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        return [p.detach().clone() for p in model.parameters()],random.random(),np.random.rand()
    step();step()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'training_state_latest.pth'
        save_training_state(path,model,optimizer,center,scheduler,scaler,memory,aux,2,{'test':True},Config())
        payload=torch.load(path,map_location='cpu',weights_only=False)
        expected=step()
        epoch,ledger=restore_training_state(payload,model,optimizer,center,scheduler,scaler,memory,aux)
        actual=step()
        assert epoch==2 and ledger=={'test':True}
        assert all(torch.equal(a,b) for a,b in zip(expected[0],actual[0]))
        assert expected[1:]==actual[1:]
        assert not path.with_suffix('.pth.tmp').exists()
    print('V13_FULL_STATE_NEXT_STOCHASTIC_UPDATE_REPRODUCED '+device,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');main(parser.parse_args().device)
