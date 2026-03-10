# Semantic Prior File Specification (Chapter 4)

This repository supports class-level semantic priors for the optional semantic branch.

## 1. What is a semantic prior file?
A semantic prior file stores one semantic embedding vector per class:

- shape: **[num_class, d_sem]**
- row index: class index used by training labels
- each row: one class semantic embedding

During training, this file is loaded once and used as class semantic prototypes.

## 2. Supported file formats
The loader currently supports:

- `.npy`: `numpy.ndarray` with shape `[num_class, d_sem]`
- `.pt` / `.pth`:
  - `torch.Tensor` of shape `[num_class, d_sem]`, or
  - `dict` with key `"embeddings"` and tensor value `[num_class, d_sem]`
- `.json`:
  - `[[...], [...], ...]`, or
  - `{ "embeddings": [[...], [...], ...] }`

If `semantic_path` is empty or the file does not exist, code falls back to identity one-hot semantic priors (`[num_class, num_class]`).

## 3. Class-order alignment (IMPORTANT)
Rows must align with the **effective class ids used by model training**.

In this codebase:
- the classifier output dim is `n_class` from dataloader
- semantic row `i` is treated as class `i`

So semantic rows MUST follow the same class id ordering as your labels in training/testing batches.

## 4. Minimal examples

### 4.1 Save `.npy`
```python
import numpy as np

num_class = 7
d_sem = 384
emb = np.random.randn(num_class, d_sem).astype('float32')
np.save('semantic_priors/Houston_semantic.npy', emb)
```

### 4.2 Save `.pt` with dict
```python
import torch

num_class = 7
d_sem = 384
emb = torch.randn(num_class, d_sem)
torch.save({'embeddings': emb}, 'semantic_priors/Houston_semantic.pt')
```

### 4.3 Save `.json`
```python
import json
import numpy as np

num_class = 7
d_sem = 128
emb = np.random.randn(num_class, d_sem).tolist()
with open('semantic_priors/Houston_semantic.json', 'w', encoding='utf-8') as f:
    json.dump({'embeddings': emb}, f)
```

## 5. Recommended workflow
1. Prepare your class list and fixed class order.
2. Generate class text descriptions from LLM (offline).
3. Encode descriptions into vectors with your text encoder (offline).
4. Save vectors to one of supported formats.
5. Start training with:

```bash
python main.py --config param.yaml \
  --use_semantic_branch True \
  --semantic_path ./semantic_priors/Houston_semantic.npy
```

## 6. Ablation suggestions
- Baseline (Chapter 3): `use_semantic_branch=False`
- +Source semantic alignment: `use_semantic_branch=True`, `semantic_tgt_weight=0`
- +Target semantic consistency: `use_semantic_branch=True`, `semantic_src_weight=0`
- Full Chapter 4: both source and target semantic weights > 0
