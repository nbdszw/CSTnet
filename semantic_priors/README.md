This folder stores semantic prior assets for Chapter 4.

## Recommended structure
- `scripts/build_semantic_priors.py`: builder pipeline
- `examples/*.yaml`: builder config examples
- `<Dataset>/semantic_bank_*.npy`: generated semantic banks

## Quick start
```bash
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

Detailed spec and integration guide: `docs/semantic_priors.md`.
