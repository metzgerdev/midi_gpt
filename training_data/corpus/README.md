# training_data/corpus

The source MIDI `mine_corpus.py` reads. Everything here except this file is gitignored:
it is purchased sample-pack MIDI verbatim, and the licence covers using the packs, not
redistributing them. `../manifests/*.json` is the tracked substitute — it names every
file each shipped example came from.

```
corpus/
├── genre_corpus/       495 folders of {harmony,bass,melody}.mid   — arrangements
├── genre_corpus_aug/  5928 folders, the same 495 x 12 semitones   — pre-augmented
├── Midi/               133 descriptively-named pack loops
└── old_school_house_midi/  125 the same, from a second pack
```

Packs represented: SO UKG 140, JAFUNK, HN2, OSG/Osh, TSP SDG. The default `--filter ukg`
keeps only paths mentioning `ukg`/`garage`/`2step`, which in practice means the `SOUKG_*`
arrangements and the `SO_UKG_140_*` loops; the JAFUNK, kpop and kaggle material in
`genre_corpus/` is present but filtered out.

## Why both `genre_corpus/` and `genre_corpus_aug/`

`genre_corpus_aug/` is the corpus the shipped checkpoints were trained from, so it is
here to make that reproducible — mining it reproduces all 7,704 `.npz` byte for byte.
It is not the corpus to mine going forward. It is `genre_corpus/` already transposed to
twelve keys, and `mine_corpus.py` then transposes twelve times again:

|                               | `genre_corpus_aug/` | `genre_corpus/` |
| ----------------------------- | ------------------- | --------------- |
| bass files after dedup        | 354                 | 45              |
| distinct phrase families      | 45                  | 45              |
| examples written              | 4,248               | 540             |
| bit-identical duplicates among them | 1,420 (33%)   | 0               |

Both columns hold the same 45 basslines. Content hashing cannot collapse the left one
because transposition changes every byte, so the twelve keys of one phrase enter as
twelve phrases and leave as 144 examples.

## Where it came from

Copied from `LLMs-from-scratch/zynar-music/multitrack-collaborator`, a separate project
of mine that assembled the packs into role-split arrangement folders. Its `output/`
directory — that project's own model generations — is deliberately **not** here, and no
example in this repo traces to it. `training_data/README.md` records how that was
checked, and why the earlier claim that it did was wrong.
