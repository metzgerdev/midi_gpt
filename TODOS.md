# TODOs

## Remove the dead CLAP conditioning path

`HarmonicNoteGPT.cond_proj` is a `Linear(512, 128)` for a CLAP clip embedding that is
never supplied. It should come out.

**It is dead, not merely unused.** Every call site passes `cond=None` explicitly —
`train_notes.py:121`, `finetune_sft.py:124,139`, `finetune_dpo.py:158,264`, and
`model.py:94` inside `generate_notes`. No code in the repo can produce a CLAP vector:
there is no encoder, no import, and `CLAP_DIM` in `config.py` is the only other mention.

**It was never trained.** `forward` skips the layer when `cond is None`, so it never
enters the graph and never receives a gradient. In all eight checkpoints the weights are
untouched PyTorch init — uniform over ±1/sqrt(512) = ±0.0442, std 0.0255, matching a
freshly constructed layer exactly. `cond_proj.weight` is bit-identical between each base
checkpoint and every ftN descendant (`max|diff| = 0.00e+00`), while layers that did train
moved on the order of 1e-2.

**Cost:** 65,664 of 700,720 parameters — 9.4% of every checkpoint, ~2.1 MB across the
eight files — serialized, loaded to device, and shipped as random numbers.

### Blast radius

Four sites load a checkpoint, all with `strict=True` (three by default, one explicit).
Dropping the layer makes all four raise `RuntimeError: Unexpected key(s) in state_dict`
against the existing checkpoints:

| file | line |
| --- | --- |
| `make_track.py` | 282 |
| `finetune_sft.py` | 105 |
| `finetune_dpo.py` | 239 |
| `test_inference.py` | 197 |

`inference.ipynb` is **not** affected directly — it never loads a model, it calls
`make_track.main()` and only globs checkpoint *filenames* for its `WEIGHTS` selector. It
breaks only transitively through `make_track.load_note_model`.

### Shape of the fix

Give the four sites one shared loader that strips the retired keys, rather than four
hand-rolled `torch.load` + `load_state_dict` pairs:

```python
def load_checkpoint(path, device):
    """Load a checkpoint, dropping the retired CLAP projection if it is present."""
    saved = torch.load(path, map_location=device, weights_only=False)
    for key in ("cond_proj.weight", "cond_proj.bias"):
        saved["model"].pop(key, None)
    return saved
```

`pop(..., None)` is idempotent, so it handles both old checkpoints that carry the keys
and new ones written after the change. Then drop `cond_proj` and the `cond` parameter
from `HarmonicNoteGPT.forward`, and `CLAP_DIM` from `config.py`.

The consolidation is the real win — those four load sites have already drifted, with only
the test passing `strict=True` deliberately.

### Note

This changes no generated output. If text-prompt conditioning is ever wanted, `cond_proj`
is the slot it would go back into — but it would need a CLAP encoder and retraining with
real embeddings, so keeping a dead layer against that possibility is not buying anything.
