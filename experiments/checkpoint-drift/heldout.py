import json, numpy as np, torch
from pathlib import Path
from utils.config import *
from model.note_model import HarmonicNoteGPT
from train.finetune_dpo import preference_pairs, sequence_logprob
CFG={"vocab_size":NOTE_VOCAB_SIZE,"context_length":NOTE_STEPS+4,"emb_dim":128,
     "n_heads":4,"n_layers":3,"drop_rate":0.1,"qkv_bias":False}
rd=Path("output/0820-142701_a_minor_dark_8bar_ft3")
print("held-out run:", rd.name, "(not in training_data/dpo)")
pairs,_,_=preference_pairs(rd,"bass")
print(f"  {len(pairs)//12} sections x 12 = {len(pairs)} pairs")
def col(s,f): return torch.tensor(np.array([p[s][f] for p in pairs]))
xc,yc,xr,yr=col(0,0).long(),col(0,1).long(),col(1,0).long(),col(1,1).long()
g,c=col(0,2).float(),col(0,3).float()
res={}
for st,fn in [("base","bass_notes_gpt.pt")]+[(f"ft{i}",f"bass_notes_gpt_ft{i}.pt") for i in(1,2,3)]:
    d=torch.load(CKPT_DIR/fn,map_location="cpu",weights_only=False)
    m=HarmonicNoteGPT(CFG); m.load_state_dict(d["model"]); m.eval()
    with torch.no_grad():
        lc=sequence_logprob(m,xc,yc,g,c); lr=sequence_logprob(m,xr,yr,g,c)
    se=lambda v: float(v.std(unbiased=True)/np.sqrt(len(v)))
    res[st]={"chosen":float(lc.mean()),"chosen_se":se(lc),"rejected":float(lr.mean()),
             "rejected_se":se(lr),"margin":float((lc-lr).mean()),"margin_se":se(lc-lr),
             "acc":float(((lc-lr)>0).float().mean())}
    r=res[st]
    print(f"  {st:5} logP(edit)={r['chosen']:8.2f}  logP(orig)={r['rejected']:8.2f}  "
          f"margin={r['margin']:+7.2f}±{r['margin_se']:.2f}  acc={r['acc']:.0%}")
json.dump(res,open("/tmp/heldout.json","w"),indent=1)
