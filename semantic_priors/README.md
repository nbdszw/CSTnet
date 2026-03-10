该目录用于存放第四章语义先验相关资源。

## 推荐目录结构
- `scripts/build_semantic_priors.py`：语义先验构建脚本
- `examples/*.yaml`：构建配置示例
- `<Dataset>/semantic_bank_*.npy`：构建产出的语义原型库

## 快速开始
```bash
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

完整说明请见：`docs/semantic_priors.md`。
