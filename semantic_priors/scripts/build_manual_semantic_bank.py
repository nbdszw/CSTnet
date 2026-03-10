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


def encode_texts(texts: List[str], dim: int = 256) -> np.ndarray:
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

    z = z.astype(np.float32)
    if z.shape[1] < dim:
        pad = np.zeros((z.shape[0], dim - z.shape[1]), dtype=np.float32)
        z = np.concatenate([z, pad], axis=1)
    else:
        z = z[:, :dim]

    return l2_normalize(z)


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
    dim = int(cfg.get('encoding_dim', 256))
    merge_mode = cfg.get('merge_mode', 'concat')  # concat | coarse_only | fine_only

    classes = sorted(cfg['classes'], key=lambda x: int(x['id']))
    coarse = [c['coarse'] for c in classes]
    fine = [c['fine'] for c in classes]

    emb_coarse = encode_texts(coarse, dim=dim)
    emb_fine = encode_texts(fine, dim=dim)

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
    print('- manual_semantic_bank_combined.npy (recommended for training)')


if __name__ == '__main__':
    main()
