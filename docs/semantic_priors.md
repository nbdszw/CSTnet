# Semantic Prior Construction and File Specification (Chapter 4)

This document explains **how to construct** and **how to use** semantic priors for Chapter 4.

## 1) What is implemented in code

We provide a full pipeline script:

- `semantic_priors/scripts/build_semantic_priors.py`

It implements:

1. hierarchical text generation
   - coarse (scene-invariant): \(T_i^{(c)}\)
   - fine-source/fine-target (scene-aware): \(T_{i,src}^{(f)}, T_{i,tgt}^{(f)}\)
2. multi-generation with K candidates per class/level
3. rule-based aggregation with stability threshold \(\rho\)
4. text embedding + L2 normalization
5. consistency-based confidence weights
6. export prototype banks (`.npy`) + metadata (`.json`)

The script supports two generation backends:

- `template` (offline deterministic fallback)
- `openai` (OpenAI-compatible chat-completions endpoint)

---

## 2) Input config format

Use a YAML config (example: `semantic_priors/examples/houston_semantic_builder.yaml`).

Required fields:

- `dataset`: dataset name
- `classes`: list of class specs (`id`, `name`, optional `alias`, optional `definition`)
- `scene_info`: source/target scene metadata with keys:
  - `sensor`, `season`, `region`, `resolution`, `illumination`

Unknown values should be set to `unknown` (do not hallucinate missing metadata).

---

## 3) Run semantic-prior construction

### 3.1 Offline template backend

```bash
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

### 3.2 OpenAI-compatible backend

1. set `generation.backend: openai` in config
2. set API key env var (default `OPENAI_API_KEY`)
3. run the same command

```bash
export OPENAI_API_KEY="<your_key>"
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

---

## 4) Output files and meanings

Suppose `output_dir: semantic_priors/Houston`, generated files include:

- `semantic_bank_coarse.npy`: \(\mathbf{P}_c\), shape `[C, d_s]`
- `semantic_bank_fine_src.npy`: \(\mathbf{P}_f^{(src)}\), shape `[C, d_s]`
- `semantic_bank_fine_tgt.npy`: \(\mathbf{P}_f^{(tgt)}\), shape `[C, d_s]`
- `semantic_bank_combined.npy`: merged bank for current training code (recommended)
- `semantic_weights_coarse.npy`: \(w_i^{(c)}\), shape `[C]`
- `semantic_weights_fine_src.npy`: \(w_i^{(f,src)}\), shape `[C]`
- `semantic_weights_fine_tgt.npy`: \(w_i^{(f,tgt)}\), shape `[C]`
- `semantic_metadata.json`: generated texts, confidence, config snapshot

---

## 5) File format constraints for training

Current training code (`semantic_loader.py`) accepts:

- `.npy`: `[num_class, d_sem]`
- `.pt/.pth`: tensor or dict key `embeddings`
- `.json`: list or dict key `embeddings`

For Chapter 4 training, you can directly use:

- `semantic_bank_combined.npy`

Example:

```bash
python main.py --config param.yaml --data_dir ./Dataset/Houston --num_bands 48 \
  --use_semantic_branch True \
  --semantic_path ./semantic_priors/Houston/semantic_bank_combined.npy
```

---

## 6) Class-order alignment (IMPORTANT)

Rows must align with model class ids:

- row `i` corresponds to class id `i`
- `num_class` must match training dataloader output classes

A mismatch will cause semantic misalignment and degrade training.

---

## 7) Suggested ablations

- baseline (Chapter 3): `use_semantic_branch=False`
- +source alignment only: `use_semantic_branch=True`, `semantic_tgt_weight=0`
- +target consistency only: `use_semantic_branch=True`, `semantic_src_weight=0`
- full Chapter 4: both source and target semantic weights > 0

You can also swap prior variants:

- coarse only: use `semantic_bank_coarse.npy`
- fine src only: use `semantic_bank_fine_src.npy`
- fine tgt only: use `semantic_bank_fine_tgt.npy`
- combined: use `semantic_bank_combined.npy`
