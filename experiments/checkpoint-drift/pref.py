import json, numpy as np, torch
from pathlib import Path
from utils.config import *
from model.note_model import HarmonicNoteGPT
from train.finetune_dpo import preference_pairs, sequence_logprob

CFG={"vocab_size":NOTE_VOCAB_SIZE,"context_length":NOTE_STEPS+4,"emb_dim":128,
     "n_heads":4,"n_layers":3,"drop_rate":0.1,"qkv_bias":False}
DEV=torch.device("cpu"); RUNS=sorted((TRAINING_DATA/"dpo").iterdir())
out={}
for role in ("bass","arp"):
    pairs=[]
    for rd in RUNS:
        if not rd.is_dir(): continue
        try:
            found,_,_=preference_pairs(rd, role)
            pairs+=found
        except SystemExit as e: print(f"  skip {rd.name}/{role}: {e}")
    if not pairs:
        print(f"[{role}] no pairs"); continue
    def col(side,f): return torch.tensor(np.array([p[side][f] for p in pairs]))
    xc,yc=col(0,0).long(),col(0,1).long(); xr,yr=col(1,0).long(),col(1,1).long()
    g,c=col(0,2).float(),col(0,3).float()
    print(f"[{role}] {len(pairs)//12} sections x 12 keys = {len(pairs)} pairs")
    out[role]={"n_sections":len(pairs)//12}
    for st,fn in [("base",f"{role}_notes_gpt.pt")]+[(f"ft{i}",f"{role}_notes_gpt_ft{i}.pt") for i in(1,2,3)]:
        d=torch.load(CKPT_DIR/fn,map_location=DEV,weights_only=False)
        m=HarmonicNoteGPT(CFG); m.load_state_dict(d["model"]); m.eval()
        with torch.no_grad():
            lc=sequence_logprob(m,xc,yc,g,c); lr=sequence_logprob(m,xr,yr,g,c)
        se=lambda v: float(v.std(unbiased=True)/np.sqrt(len(v)))
        out[role][st]={"chosen":float(lc.mean()),"chosen_se":se(lc),
                       "rejected":float(lr.mean()),"rejected_se":se(lr),
                       "margin":float((lc-lr).mean()),"margin_se":se(lc-lr),
                       "acc":float(((lc-lr)>0).float().mean())}
        o=out[role][st]
        print(f"  {st:5} logP(edit)={o['chosen']:8.2f}±{o['chosen_se']:.2f}  "
              f"logP(orig)={o['rejected']:8.2f}±{o['rejected_se']:.2f}  "
              f"margin={o['margin']:+7.2f}±{o['margin_se']:.2f}  acc={o['acc']:.0%}")
json.dump(out,open("/tmp/pref.json","w"),indent=1)
