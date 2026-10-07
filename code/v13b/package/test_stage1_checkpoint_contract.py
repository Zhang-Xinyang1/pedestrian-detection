"""Verify the approved Stage 1 artifact on CPU without training or output writes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args=parser.parse_args()
    approved=json.loads((ROOT/"APPROVED_STAGE1.json").read_text())
    digest=hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda:handle.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest()!=approved["file_sha256"]:
        raise ValueError("Approved checkpoint file SHA256 differs")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    hashes={}
    for name,tensor in state.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite checkpoint tensor: "+name)
        hashes[name]=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    mapping_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if mapping_hash!=approved["tensor_hash"]:
        raise ValueError("Approved tensor hash differs")
    if not all(hashes[name]==value for name,value in approved["parameter_hashes"].items()):
        raise ValueError("Approved parameter audit differs")
    print("CSF_STAGE1_CHECKPOINT_CONTRACT_OK",digest.hexdigest())

if __name__=="__main__":
    torch.set_num_threads(4)
    main()
