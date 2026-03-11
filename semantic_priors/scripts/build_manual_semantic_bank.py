import argparse
import json
import os
from typing import List

import numpy as np
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def _fit_to_dim(z: np.ndarray, dim: int) -> np.ndarray:
    """Project/pad vectors to target dim while keeping deterministic behavior."""
    z = z.astype(np.float32)
    if z.shape[1] < dim:
        pad = np.zeros((z.shape[0], dim - z.shape[1]), dtype=np.float32)
        z = np.concatenate([z, pad], axis=1)
    else:
        z = z[:, :dim]
    return l2_normalize(z)


def encode_texts_tfidf(texts: List[str], dim: int = 256) -> np.ndarray:
    tfidf = TfidfVectorizer(max_features=max(1024, dim * 8), ngram_range=(1, 2))
    x = tfidf.fit_transform(texts)

    if x.shape[1] > dim:
        svd_dim = min(dim, x.shape[0] - 1, x.shape[1] - 1)
        if svd_dim >= 2:
            svd = TruncatedSVD(n_components=svd_dim, random_state=0)
            z = svd.fit_transform(x)
        else:
            z = x.toarray()
    else:
        z = x.toarray()

    return _fit_to_dim(z, dim=dim)


def encode_texts_clip(
    texts: List[str],
    dim: int = 512,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> np.ndarray:
    """Encode texts with CLIP text encoder (transformers backend)."""
    try:
        import torch
        from transformers import AutoTokenizer, CLIPModel
    except Exception as e:
        raise ImportError(
            "CLIP encoding requires torch + transformers. "
            "Install them or switch encoding_backend to 'tfidf'."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()

    all_vec = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            token = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            token = {k: v.to(device) for k, v in token.items()}
            text_feat = clip_model.get_text_features(**token)
            all_vec.append(text_feat.cpu().numpy())

    z = np.concatenate(all_vec, axis=0).astype(np.float32)
    z = l2_normalize(z)
    return _fit_to_dim(z, dim=dim)


def parse_args():
    parser = argparse.ArgumentParser(description='Build semantic bank from manual text descriptions (no LLM).')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = cfg['dataset']
    out_dir = cfg.get('output_dir', f'semantic_priors/{dataset}')
    dim = int(cfg.get('encoding_dim', 512))
    merge_mode = cfg.get('merge_mode', 'concat')  # concat | coarse_only | fine_only
    encoding_backend = cfg.get('encoding_backend', 'clip')  # clip | tfidf
    clip_model_name = cfg.get('clip_model_name', 'openai/clip-vit-base-patch32')
    clip_device = cfg.get('clip_device', 'cpu')

    classes = sorted(cfg['classes'], key=lambda x: int(x['id']))
    coarse = [c['coarse'] for c in classes]
    fine = [c['fine'] for c in classes]

    if encoding_backend == 'clip':
        emb_coarse = encode_texts_clip(coarse, dim=dim, model_name=clip_model_name, device=clip_device)
        emb_fine = encode_texts_clip(fine, dim=dim, model_name=clip_model_name, device=clip_device)
    elif encoding_backend == 'tfidf':
        emb_coarse = encode_texts_tfidf(coarse, dim=dim)
        emb_fine = encode_texts_tfidf(fine, dim=dim)
    else:
        raise ValueError(f'Unsupported encoding_backend: {encoding_backend}')

    if merge_mode == 'coarse_only':
        emb_combined = emb_coarse
    elif merge_mode == 'fine_only':
        emb_combined = emb_fine
    elif merge_mode == 'concat':
        emb_combined = np.concatenate([emb_coarse, emb_fine], axis=1)
        emb_combined = l2_normalize(emb_combined)
    else:
        raise ValueError(f'Unsupported merge_mode: {merge_mode}')

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'manual_semantic_bank_coarse.npy'), emb_coarse)
    np.save(os.path.join(out_dir, 'manual_semantic_bank_fine.npy'), emb_fine)
    np.save(os.path.join(out_dir, 'manual_semantic_bank_combined.npy'), emb_combined)

    meta = {
        'dataset': dataset,
        'encoding_backend': encoding_backend,
        'clip_model_name': clip_model_name if encoding_backend == 'clip' else None,
        'encoding_dim': dim,
        'merge_mode': merge_mode,
        'num_classes': len(classes),
        'classes': classes,
        'output_files': [
            'manual_semantic_bank_coarse.npy',
            'manual_semantic_bank_fine.npy',
            'manual_semantic_bank_combined.npy',
        ],
    }
    with open(os.path.join(out_dir, 'manual_semantic_bank_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'[Done] Manual semantic banks saved to: {out_dir}')
    print(f'- backend: {encoding_backend}')
    print('- manual_semantic_bank_combined.npy (recommended for training)')


if __name__ == '__main__':
    main()
