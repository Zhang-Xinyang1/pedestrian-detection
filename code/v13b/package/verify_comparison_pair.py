"""Verify that all smoke arms started from identical state and protocols."""
import argparse
import hashlib
import json
from pathlib import Path
from prototype_comparison import METHODS

def read(path):
    return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("prefix",type=Path)
    args=parser.parse_args()
    reference=None
    for method in METHODS:
        directory=Path(str(args.prefix)+"_"+method)
        manifest=read(directory/"manifest.json")
        audit=read(directory/"stage2_audit.json")
        marker=read(directory/"VERIFICATION.json")
        initial=read(directory/"stage2_initial_parameters.json")
        cfg=manifest["config"].copy()
        cfg["OUTPUT_DIR"]="<arm>"
        record={
            "seed":manifest["info"]["seed"],
            "source_sha":manifest["info"]["source_stage1_checkpoint_sha256"],
            "initial_parameters":initial,
            "config":cfg,
            "train_scene_counts":manifest["train_scene_counts"],
            "counts":manifest["actual_run_counts"],
            "query":manifest["eval_query"],
            "gallery":manifest["eval_gallery"],
            "candidate_policy":manifest["info"]["candidate_policy"],
            "temperature":manifest["info"]["csf_temperature"],
            "reconstruction_weight":manifest["info"]["csf_reconstruction_weight"],
            "sampler_plan_sha256":manifest["info"]["sampler_plan_sha256"],
        }
        assert marker["status"]=="PASS" and marker["smoke"]
        assert manifest["info"]["prototype_method"]==method
        assert audit["csf"]["prototype_method"]==method
        assert audit["final_epoch"]==4 and not audit["stopped_early"]
        assert audit["actual_batches_per_epoch"]==[6,6,6,6]
        assert audit["csf"]["weight"]==(0.0 if method=="control" else 0.1)
        assert manifest["info"]["early_stop_policy"]["enabled"] is False
        if reference is None:
            reference=record
        else:
            assert record==reference, "Mismatched comparison contract: "+method
    digest=hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()
    print("PROTOTYPE_COMPARISON_PAIR_OK",digest)

if __name__=="__main__":
    main()
