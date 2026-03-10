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


## 无 LLM 的手工文本构建
可将论文中的 coarse/fine 类别语义直接写入 YAML（示例：`examples/pavia_manual_semantics.yaml`），然后运行：

```bash
python semantic_priors/scripts/build_manual_semantic_bank.py   --config semantic_priors/examples/pavia_manual_semantics.yaml
```
