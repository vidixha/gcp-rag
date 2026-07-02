# Code-Switched Visual Language Routing in VLMs — Research Report

*Curated analysis, 2026-07-02. Auto-generated figures in `figures/`; a full-scale
rerun (larger samples per stage) is in progress — numbers below are from the
current clean-data run and will be refreshed.*

## 1. Research question

When a VLM processes the same code-switched (CS) content, does it recruit the same
language-sensitive decoder heads regardless of input modality? We compare three
conditions per sentence: **A** text-only tokens, **B** rendered image + text query,
**C** rendered image only. Upstream, we ask whether the ViT already separates
scripts (Latin vs Devanagari) at the patch level.

## 2. Data

All text is real, human-produced, with gold token-level language labels — nothing
synthesized. Rendered as black text on 512×512 white canvases with per-token
bounding boxes; font auto-shrinks so no text is ever cut off.

| Subset | Source | Size (post-filter) |
|---|---|---|
| Monolingual en/es/hi | FLORES-200 dev+devtest (3-way aligned triples) | 750 (250/lang) |
| Code-switched hi-en | GLUECoS POS (UD, Devanagari Twitter CS) | 938 |
| Code-switched es-en | GLUECoS POS (Bangor Miami speech) | 119 |
| Natural (held-out hi-en) | GLUECoS, disjoint slice | 134 |
| Controls | Hindi romanized to Latin (IAST, diacritics stripped) | 100 |

Cleaning: PTB detokenization (apostrophes/contractions restored, punctuation
rejoined), noise/script-consistency filters, and a two-stage fluency filter
(anonymization-hole heuristics + Qwen2.5-0.5B perplexity, worst quartile dropped).
Kept 1,057/1,608 CS and 134/200 natural sentences. `metadata.prefilter.json`
preserves the removed records for the data appendix.

## 3. Models

| Key | HF id | Decoder | ViT grid |
|---|---|---|---|
| llava15_7b | llava-hf/llava-1.5-7b-hf | Vicuna/Llama-2-7B (32L) | 24×24 (+CLS) |
| qwen2vl_7b | Qwen/Qwen2-VL-7B-Instruct | Qwen2-7B (28L) | 36×36, 2×2-merged to 18×18 LLM tokens |
| internvl3_8b | OpenGVLab/InternVL3-8B-hf | Qwen2.5-7B (28L) | 32×32 (+CLS), pixel-shuffled to 16×16 LLM tokens |

internvl3_8b replaces paligemma_3b, which failed the language-competence gate
(mean ppl 5.4k–16.4k). Decoder-only perplexity on monolingual FLORES (extracted
decoder + full model's lm_head), competence threshold ppl < 100:

| Model | en | es | hi |
|---|---|---|---|
| llava15_7b | 39.7 | 28.9 | 6.1 |
| qwen2vl_7b | 62.8 | 27.7 | 4.5 |
| internvl3_8b | 65.0 | 31.1 | 5.3 |

## 4. Results

### 4.1 ViT script probes (fig1)
Linear probes on frozen per-layer ViT patch features predict patch script
(Latin/Devanagari, balanced accuracy, 5-fold CV, CLS-aligned):

- **llava15_7b (CLIP ViT-L):** 0.87 at L0 → **peak 0.96 at L9** → 0.90 at L23.
- **qwen2vl_7b (native ViT):** 0.61 at L0 → **peak 0.76 at L27** (last layers).
- internvl3_8b (InternViT): recomputing (an extraction bug produced empty features;
  fixed).

Script identity is linearly decodable inside the vision tower well before the LLM
sees anything — strongly in CLIP (mid-stack), more weakly and later in Qwen's ViT.

### 4.2 Language-head overlap across modalities (fig2, central result)
LAHIS scores per head (layers × heads), top-20 heads per condition; overlap =
Jaccard of top-sets. Chance ≈ 0.01.

| Model | lang | A∩B | A∩C | B∩C |
|---|---|---|---|---|
| llava15_7b | en / es / hi | 0.08 / 0.21 / 0.21 | 0.29 / 0.25 / 0.21 | 0.21 / 0.54 / **0.90** |
| qwen2vl_7b | en / es / hi | 0.25 / 0.29 / 0.25 | 0.29 / 0.33 / 0.21 | 0.29 / 0.29 / 0.29 |
| internvl3_8b | en / es / hi | 0.18 / 0.25 / **0.54** | 0.43 / 0.33 / 0.33 | 0.25 / 0.21 / 0.18 |

Three observations:
1. **Partial unified routing.** A∩C sits at 0.21–0.43 everywhere — far above
   chance but below 0.5: a consistent minority of language-sensitive heads is
   shared between reading text as tokens and reading the same text from pixels.
2. **A language-agnostic core.** Within each model, 8–11 of the top-20 text-mode
   heads are identical across en/es/hi (llava 11, internvl 10, qwen 8),
   concentrated at layers 0–1 plus a late cluster (llava L31, qwen/internvl L27).
   The head set is more "language-processing" than "per-language".
3. **Decoder family predicts coordinates.** qwen2vl and internvl — different
   vision towers, same Qwen decoder family — share exact head coordinates
   ((0,0), (0,11), (0,13), (0,15), (0,20) and a L27 cluster); llava's set is
   disjoint. The circuitry appears inherited from language pretraining, not
   induced by multimodal finetuning.

### 4.3 Spatial attention locality (fig3)
Do top language heads attend preferentially to patches of the matching script?
For llava the on-script/off-script attention ratio is ≈ 0.94–0.96 (en/hi), i.e.
**no spatial script-selectivity** — language heads integrate over the whole text
region rather than fixating their language's patches. qwen2vl/internvl are being
recomputed (the LLM-side token grid is 2×2-merged / pixel-shuffled; the first pass
mapped labels on the wrong grid).

### 4.4 Causal ablation (fig4)
Zero-ablating the top-20 language heads while the model reads a hi-en CS image
(n=50/model, generation script tracked):

| Model | baseline output Devanagari | script changed after en-head ablation | after hi-head ablation |
|---|---|---|---|
| llava15_7b | 1/50 | 8/50 | 5/50 |
| qwen2vl_7b | 3/50 | 42/50 | 37/50 |
| internvl3_8b | 17/50 | 46/50 | 48/50 |

The heads are causally load-bearing in the Qwen-family models (ablation flips
output script in 74–96% of cases and often halves output length); llava is more
robust, consistent with its lower A∩C overlap. Caveat: en- vs hi-head ablations
flip at similar rates, so part of the effect is generic degradation rather than
language-selective control — mean-ablation and random-head baselines are needed
to separate the two (planned).

## 5. What the results mean

The picture is a **two-stage architecture**: script separation happens early and
visually (linearly decodable in the ViT), while language handling in the decoder
runs through a small, language-agnostic head set that is partially shared across
input modalities and strongly conserved within a decoder family. Code-switched
input does not get per-language circuits; it gets a general "language routing"
apparatus that the visual pathway taps into only partially (A∩C ≈ 0.2–0.4) — and
in llava's case, the image conditions essentially form their own circuit
(B∩C up to 0.90 for Hindi).

Positioning: language-selective heads are known in text-only LLMs (arXiv
2511.07498) and OCR heads in VLMs (arXiv 2505.15865); the contribution here is
the modality contrast on *code-switched* input with script/language decoupling
(transliteration controls) and the decoder-family conservation result.

## 6. Limitations / in progress

- Top-K=20 overlap is threshold-sensitive → K-sensitivity + permutation baseline
  (running).
- Sample sizes: LAHIS 30/condition, patching 200 imgs, ablation 50 — full-scale
  rerun in progress (100 / all 1,057 / 300).
- 2 of 3 decoders are Qwen-family; a non-Qwen fourth model (e.g. Gemma-based)
  would strengthen generality.
- Zero-ablation is harsh; add mean-ablation + random-head controls.
- Rendered text ≠ natural scene text; three languages, two scripts.
