"""Report the fixed100 outcome independently of engineering verification."""
import json
import sys
from pathlib import Path

def main():
    directory=Path(sys.argv[1]);path=directory/'FIXED100_TARGET_REPORT.json'
    if path.exists():raise FileExistsError('Never overwrite a formal target report')
    result=json.loads((directory/'eval_stage2_100.json').read_text())
    passed=result['ap']>=.12 and result['r1']>=.30
    record={'status':'TARGET_MET' if passed else 'TARGET_NOT_MET','fixed_epoch':100,
        'mAP_percent':100*result['ap'],'Rank1_percent':100*result['r1'],
        'targets':{'mAP_percent':12.,'Rank1_percent':30.},'valid_queries':result['valid_queries'],
        'reference_paper_UAD':{'mAP_percent':11.,'Rank1_percent':29.5},
        'reference_historical_UAD':{'mAP_percent':10.83,'Rank1_percent':28.4},
        'fixed_result_not_test_peak':True,'note':'Engineering PASS does not establish target achievement.'}
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('V13_FIXED100_TARGET '+json.dumps(record),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
