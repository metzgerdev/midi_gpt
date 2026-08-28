import json, glob, numpy as np, torch
from utils.config import *
from model.note_model import HarmonicNoteGPT, generate_notes
from utils.scoring import fit_and_lock
CFG={"vocab_size":NOTE_VOCAB_SIZE,"context_length":NOTE_STEPS+4,"emb_dim":128,
     "n_heads":4,"n_layers":3,"drop_rate":0.1,"qkv_bias":False}
out={}
for role in ("bass","arp"):
    files=sorted(glob.glob(f"training_data/{role}_notes_midi/*.npz"))
    rng=np.random.default_rng(3); conds=[]
    for p in rng.choice(files,200,replace=False):
        d=np.load(p); g64=d["grid"].astype(np.float32); ch=d["chord"].astype(np.float32)
        conds.append((np.concatenate([g64,g64[-1:]]),np.concatenate([ch,ch[-1:]],0),ch,g64))
    out[role]={}
    for st,fn in [("base",f"{role}_notes_gpt.pt")]+[(f"ft{i}",f"{role}_notes_gpt_ft{i}.pt") for i in(1,2,3)]:
        d=torch.load(CKPT_DIR/fn,map_location="cpu",weights_only=False)
        m=HarmonicNoteGPT(CFG); m.load_state_dict(d["model"]); m.eval()
        torch.manual_seed(7); fits=[]; locks=[]
        for gt,ct,ch,g64 in conds:
            o=generate_notes(m,gt,NOTE_BOS,NOTE_EOS,max_steps=NOTE_STEPS,temperature=1.0,
                             top_p=0.98,chord_tok=ct,device=torch.device("cpu"))
            t=o[0,1:].cpu().numpy()
            t=np.pad(t,(0,max(0,NOTE_STEPS-len(t))),constant_values=NOTE_REST)[:NOTE_STEPS]
            f,l=fit_and_lock(t,ch,g64); fits.append(float(f)); locks.append(float(l))
        out[role][st]={"fit":fits,"lock":locks}
        se=lambda v: float(np.std(v,ddof=1)/np.sqrt(len(v)))
        print(f"[{role}] {st:5} fit {np.mean(fits):.3f} ±{se(fits):.3f}   "
              f"lock {np.mean(locks):.3f} ±{se(locks):.3f}", flush=True)
json.dump(out,open("/tmp/se.json","w"))
