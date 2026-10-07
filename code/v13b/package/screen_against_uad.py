"""Compare a 30-epoch screening run with the saved UAD first-30 curve."""
import json, sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
BASELINE=json.loads((PACKAGE/"UAD_FIRST30_BASELINE.json").read_text(encoding="utf-8"))
def main():
    if len(sys.argv)!=2: raise SystemExit("usage: screen_against_uad.py OUTPUT_DIR")
    directory=Path(sys.argv[1]); observations={}; failures=[]; passing_epochs=[]
    for epoch in (10,20,30):
        path=directory/f"eval_stage2_{epoch:03d}.json"
        if not path.is_file():
            failures.append(f"missing evaluation epoch {epoch}"); continue
        result=json.loads(path.read_text(encoding="utf-8"))
        candidate={"map_percent":100.0*float(result["ap"]),"rank1_percent":100.0*float(result["r1"])}
        threshold=BASELINE["epochs"][str(epoch)]
        win=candidate["map_percent"]>threshold["map_percent"] and candidate["rank1_percent"]>threshold["rank1_percent"]
        observations[str(epoch)]={"candidate":candidate,"uad":threshold,"strict_both_exceed":win}
        if win: passing_epochs.append(epoch)
        else:
            failures.append(f"epoch {epoch} did not exceed both UAD metrics")
    if len(passing_epochs)<2:
        failures.append(f"only {len(passing_epochs)}/3 checkpoints passed; at least 2 required")
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    payload={"status":"PASS" if len(passing_epochs)>=2 and not any(x.startswith("missing evaluation") for x in failures) else "REJECT",
             "method":manifest["info"]["prototype_method"],"seed":1,"screening_epochs":30,
             "passing_checkpoints":passing_epochs,"passing_checkpoint_count":len(passing_epochs),
             "uad_baseline":BASELINE,"observations":observations,"failures":failures,
             "rule":BASELINE["comparison_rule"]}
    (directory/"SCREENING_GATE.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("CSF_SCREENING_GATE_"+payload["status"]+" "+json.dumps(payload,sort_keys=True),flush=True)
    return 0 if payload["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
