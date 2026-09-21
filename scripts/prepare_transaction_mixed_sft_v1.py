"""Freeze a balanced replay+repair SFT corpus without confirmation leakage."""
import hashlib,json
from pathlib import Path
from scripts.prepare_control_v1 import ROOT,digest
from training.transaction_data import canonical

OLD=ROOT/'data'/'transaction_v1'/'train_trajectories.jsonl'
REPAIRED=ROOT/'data'/'transaction_repaired_v1'/'train_trajectories.jsonl'
OUT=ROOT/'data'/'transaction_mixed_sft_v1'
PLAN=ROOT/'data'/'transaction_mixed_sft_v1.json'

def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def build():
    old=read(OLD); repaired=read(REPAIRED)
    by={}
    for row in old: by.setdefault(row['group_id'],[]).append(row)
    # Four deterministic replay trajectories per old group: broad old coverage,
    # with no confirmation examples and no random selection at GPU runtime.
    replay=[]
    for gid in sorted(by): replay.extend(sorted(by[gid],key=lambda r:r['id'])[:4])
    assert len(replay)==196 and len({r['group_id'] for r in replay})==49
    rows=[]
    for r in sorted(replay,key=lambda r:r['id']): rows.append(dict(r,mix_source='original_train_replay'))
    for r in sorted(repaired,key=lambda r:r['id']): rows.append(dict(r,mix_source='repaired_train'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'train_trajectories.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
    manifest={
        'protocol':'transaction_mixed_sft_v1','seed':20260921,
        'counts':{'original_replay':len(replay),'repaired_train':len(repaired),'total':len(rows)},
        'groups':{'original_replay':49,'repaired_train':12},
        'source_sha256':{'original_train':digest(OLD),'repaired_train':digest(REPAIRED)},
        'confirmation_excluded':True,'probe_excluded':True,
        'selection':'first four sorted trajectories per original train group',
        'source_code_sha256':digest(Path(__file__))}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    m=build(); print(json.dumps(m,ensure_ascii=False,indent=2))
