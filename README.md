---
license: mit
tags:
  - music-generation
  - abc-notation
  - symbolic-music
  - gpt
  - char-level
  - from-scratch
library_name: pytorch
pipeline_tag: text-generation
---

# slay-piano-gpt — a tiny char-level GPT for Irish jig generation (rendered to piano)

A **0.82M-parameter** decoder-only Transformer trained **from scratch** on ~12k Irish jigs in
[ABC notation](https://abcnotation.com/), sourced from [thesession.org](https://thesession.org/).
It generates new 6/8 jig melodies one character at a time.

Built as a learning + research project for the **Slayer** collective. This is the "small LLM" half of an
**n-gram → mini-transformer** comparison: the same next-token objective as a frontier LLM, at a scale you
can train on a CPU in ~12 minutes.

## Model details
- **Architecture:** decoder-only Transformer (GPT-style) — token + positional embeddings, causal
  multi-head self-attention, GELU MLP, residual + LayerNorm, weight-tied output head.
- **Size:** 4 layers, 4 heads, `d_model=128`, context 128 chars, vocab 52 (character-level).
- **Parameters:** 816,384.
- **Tokenizer:** character-level (52 ABC symbols) — no external tokenizer.

## Training data
- 12,106 jigs (meter 6/8) from thesession.org, ABC notation.
- Chord-symbol annotations (`"..."`) stripped; ornaments (`~`), chords (`[ace]`) and accidentals kept.
- ~2.45M characters, 90/10 train/val split.
- The raw dataset is **not redistributed here** — rebuild it with `src/prepare_data.py` from the
  thesession.org data dump, and please respect thesession.org's terms.

## Training
- Objective: next-character cross-entropy.
- Optimizer: AdamW (`lr=3e-4`, `wd=0.1`), grad-clip 1.0, warmup + cosine decay.
- 2000 iterations, batch 32, block 128, CPU.
- **Best validation loss: 1.335 → perplexity 3.80.** Train ≈ val (no overfitting).

## Usage
```bash
pip install torch music21
python src/make_midi.py --key G --n 3 --out out   # -> ABC + MIDI (piano)
```
Or load directly:
```python
import torch
from gpt import GPT
ck = torch.load("data/gpt_ckpt.pt", weights_only=False)
model = GPT(ck["config"]); model.load_state_dict(ck["model"]); model.eval()
# seed "X:1\nM:6/8\nK:D\n" -> model.generate(...)
```

## Results
Compared against a character-level **n-gram (order-6)** baseline on the same corpus. The Transformer
reaches lower perplexity and noticeably more coherent phrasing, thanks to its 128-character attention
window vs. the n-gram's 6-character context. Full write-up: Slayer research blog.

## Limitations (honest)
- Tiny model: captures local/phrase structure, **not** long-form musical form (no reliable AABB themes).
- ~128-character memory; no global planning.
- MIDI rendering is mechanical (fixed velocity, no pedal/phrasing).
- Occasional malformed repeat markers in raw output.
- Trained only on 6/8 jigs — don't expect other styles.

## Acknowledgements
- Data: the [thesession.org](https://thesession.org/) community.
- Built by Arkadiusz Słota for the **Slayer** collective. Educational / research project.

## License
MIT (code & weights). Training data belongs to thesession.org contributors.
