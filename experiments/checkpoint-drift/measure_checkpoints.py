"""Measure what each fine-tuning stage did to the shipped checkpoints."""
import json, glob, numpy as np, torch, torch.nn.functional as F
from utils.config import (CKPT_DIR, NOTE_VOCAB_SIZE, NOTE_STEPS, NOTE_BOS, NOTE_EOS,
                          NOTE_REST, NOTE_SUSTAIN, NOTE_PITCH0, NOTE_MIDI_HI, NOTE_MIDI_LO)
from model.note_model import HarmonicNoteGPT, generate_notes
from train.finetune_dpo import base_corpus_loader
from utils.scoring import fit_and_lock

CFG = {"vocab_size": NOTE_VOCAB_SIZE, "context_length": NOTE_STEPS + 4, "emb_dim": 128,
       "n_heads": 4, "n_layers": 3, "drop_rate": 0.1, "qkv_bias": False}
DEV = torch.device("cpu")
W = torch.ones(NOTE_VOCAB_SIZE); W[NOTE_SUSTAIN] = 0.3
W[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0

def load(path):
    d = torch.load(path, map_location=DEV, weights_only=False)
    m = HarmonicNoteGPT(CFG); m.load_state_dict(d["model"]); m.eval()
    return m, {k: v for k, v in d.items() if k not in ("model", "config")}

def corpus_loss(m, loader):
    tot = n = 0.0
    with torch.no_grad():
        for x, y, g, c in loader:
            lg = m(x, cond=None, cond_seq=g, chord_seq=c)
            tot += F.cross_entropy(lg.flatten(0, 1), y.flatten(), weight=W).item() * len(x); n += len(x)
    return tot / n

def drift(base_sd, sd):
    groups = {}
    for k in base_sd:
        if base_sd[k].dtype != torch.float32: continue
        g = ("trf_blocks." + k.split(".")[1]) if k.startswith("trf_blocks") else k.split(".")[0]
        d = (sd[k] - base_sd[k]).float()
        a, b = groups.get(g, (0.0, 0.0))
        groups[g] = (a + d.pow(2).sum().item(), b + base_sd[k].float().pow(2).sum().item())
    return {g: (num ** .5) / (den ** .5) if den else 0.0 for g, (num, den) in groups.items()}

def gen_stats(m, conds, seed=7):
    torch.manual_seed(seed)
    rest = sus = ons = 0; lens = []; fits = []; locks = []
    for grid_tok, chord_tok, chroma, grid64 in conds:
        out = generate_notes(m, grid_tok, NOTE_BOS, NOTE_EOS, max_steps=NOTE_STEPS,
                             temperature=1.0, top_p=0.98, chord_tok=chord_tok, device=DEV)
        t = out[0, 1:].cpu().numpy()
        t = np.pad(t, (0, max(0, NOTE_STEPS - len(t))), constant_values=NOTE_REST)[:NOTE_STEPS]
        rest += int((t == NOTE_REST).sum()); sus += int((t == NOTE_SUSTAIN).sum())
        ons += int((t >= NOTE_PITCH0).sum())
        run = 0
        for tok in t:
            if tok >= NOTE_PITCH0: lens.append(run) if run else None; run = 1
            elif tok == NOTE_SUSTAIN and run: run += 1
            else:
                if run: lens.append(run)
                run = 0
        if run: lens.append(run)
        f, l = fit_and_lock(t, chroma, grid64)
        fits.append(float(f)); locks.append(float(l))
    tot = rest + sus + ons
    return dict(p_rest=rest / tot, p_sustain=sus / tot, p_onset=ons / tot,
                mean_note_len=float(np.mean(lens)) if lens else 0.0,
                onsets_per_clip=ons / len(conds),
                chord_fit=float(np.mean(fits)), kick_lock=float(np.mean(locks)))

out = {}
for role in ("bass", "arp"):
    loader = base_corpus_loader(role, n=900, batch_size=64)
    files = sorted(glob.glob(f"training_data/{role}_notes_midi/*.npz"))
    rng = np.random.default_rng(3)
    conds = []
    for p in rng.choice(files, 40, replace=False):
        d = np.load(p)
        g64 = d["grid"].astype(np.float32); chroma = d["chord"].astype(np.float32)
        conds.append((np.concatenate([g64, g64[-1:]]),
                      np.concatenate([chroma, chroma[-1:]], 0), chroma, g64))
    base_m, _ = load(CKPT_DIR / f"{role}_notes_gpt.pt")
    base_sd = base_m.state_dict()
    with torch.no_grad():
        cx = torch.stack([torch.tensor(np.concatenate([[NOTE_BOS], np.load(p)["tokens"]]).astype(np.int64))
                          for p in rng.choice(files, 64, replace=False)])
        cg = torch.stack([torch.tensor(c[0]) for c in conds[:1] * 64]).float()
        cc = torch.stack([torch.tensor(c[1]) for c in conds[:1] * 64]).float()
        logq = F.log_softmax(base_m(cx, cond=None, cond_seq=cg, chord_seq=cc), -1)

    stages = [("base", f"{role}_notes_gpt.pt")] + [(f"ft{i}", f"{role}_notes_gpt_ft{i}.pt") for i in (1, 2, 3)]
    rows = []
    for name, fn in stages:
        m, meta = load(CKPT_DIR / fn)
        with torch.no_grad():
            logp = F.log_softmax(m(cx, cond=None, cond_seq=cg, chord_seq=cc), -1)
            kl = float((logp.exp() * (logp - logq)).sum(-1).mean())
        r = dict(stage=name, file=fn, method=meta.get("method", "sft" if "edits" in meta else "base"),
                 parent=str(meta.get("finetuned_from", "—")),
                 n_data=int(meta.get("edits", meta.get("pairs", 0))),
                 corpus_loss=corpus_loss(m, loader), kl_from_base=kl,
                 drift=drift(base_sd, m.state_dict()), **gen_stats(m, conds))
        rows.append(r); print(f"[{role}] {name:5} loss={r['corpus_loss']:.4f} kl={kl:.4f} "
                              f"p_onset={r['p_onset']:.3f} fit={r['chord_fit']:.3f}", flush=True)
    out[role] = rows
json.dump(out, open("/tmp/ckpt_measure.json", "w"), indent=1)
print("-> /tmp/ckpt_measure.json")
