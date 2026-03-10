# 第四章语义先验构建与文件规范（中文）

本文档说明两件事：
1. **如何构建**语义先验（LLM 生成 + 聚合 + 编码 + 置信度）
2. **如何接入**当前训练代码（`semantic_loader.py`）

---

## 1. 代码中已实现的构建流程

仓库提供了完整构建脚本：

- `semantic_priors/scripts/build_semantic_priors.py`

该脚本实现了如下流程：

1. 层次化语义生成：
   - 粗粒度（跨场景稳定语义）\(T_i^{(c)}\)
   - 细粒度（场景条件语义）\(T_{i,src}^{(f)}, T_{i,tgt}^{(f)}\)
2. 每类每层级进行 `K` 次生成
3. 基于稳定阈值 \(\rho\) 的规则化聚合
4. 文本编码与 \(\ell_2\) 归一化
5. 基于多次生成一致性的置信度估计
6. 导出语义原型库（`.npy`）与元数据（`.json`）

支持两种生成后端：

- `template`：离线模板生成（无 API 依赖，便于复现）
- `openai`：OpenAI 兼容接口（`/v1/chat/completions`）

> 当场景元信息缺失时，建议显式填 `unknown`，避免臆造先验。

---

## 2. 输入配置（YAML）说明

示例文件：`semantic_priors/examples/houston_semantic_builder.yaml`

核心字段：

- `dataset`：数据集名
- `classes`：类别列表，每个类别建议包含：
  - `id`（必须，建议从 0 连续）
  - `name`（必须）
  - `alias`（可选）
  - `definition`（可选但推荐）
- `scene_info`：源域/目标域场景信息，建议包含：
  - `sensor`, `season`, `region`, `resolution`, `illumination`

其余关键超参：

- `generation.k`：每类每层级生成次数
- `generation.rho`：稳定语义单元保留阈值
- `encoding.dim`：导出语义向量维度
- `confidence.eps`：置信度下界
- `output.merge_mode`：原型融合方式

---

## 3. 构建命令

### 3.1 模板后端（推荐先跑通）

```bash
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

### 3.2 OpenAI 兼容后端

1. 将配置中的 `generation.backend` 设为 `openai`
2. 设置 API Key（默认读 `OPENAI_API_KEY`）
3. 执行同一命令

```bash
export OPENAI_API_KEY="<your_key>"
python semantic_priors/scripts/build_semantic_priors.py \
  --config semantic_priors/examples/houston_semantic_builder.yaml
```

---

## 4. 输出文件说明

若 `output_dir: semantic_priors/Houston`，会生成：

- `semantic_bank_coarse.npy`：粗粒度原型库 \(\mathbf{P}_c\)，形状 `[C, d_s]`
- `semantic_bank_fine_src.npy`：源域细粒度原型库 \(\mathbf{P}_f^{(src)}\)，形状 `[C, d_s]`
- `semantic_bank_fine_tgt.npy`：目标域细粒度原型库 \(\mathbf{P}_f^{(tgt)}\)，形状 `[C, d_s]`
- `semantic_bank_combined.npy`：融合原型库（当前训练代码推荐直接使用）
- `semantic_weights_coarse.npy`：粗粒度置信权重 `[C]`
- `semantic_weights_fine_src.npy`：源域细粒度置信权重 `[C]`
- `semantic_weights_fine_tgt.npy`：目标域细粒度置信权重 `[C]`
- `semantic_metadata.json`：文本、权重、配置快照等元信息

---

## 5. 与训练代码对接

当前训练加载器 `semantic_loader.py` 支持：

- `.npy`：`[num_class, d_sem]`
- `.pt/.pth`：tensor 或 `{"embeddings": tensor}`
- `.json`：list 或 `{"embeddings": list}`

第四章训练建议直接使用：

- `semantic_bank_combined.npy`

示例命令：

```bash
python main.py --config param.yaml --data_dir ./Dataset/Houston --num_bands 48 \
  --use_semantic_branch True \
  --semantic_path ./semantic_priors/Houston/semantic_bank_combined.npy
```

---

## 6. 类别顺序对齐（非常重要）

语义矩阵的**第 i 行必须对应训练标签中的 class id=i**。

- 行索引与标签 id 不一致会导致语义错配
- `num_class` 必须与训练时类别数一致

建议：先确认 dataloader 的类别编码，再准备 `classes.id` 与原型行顺序。

---

## 7. 消融建议

- 第三章基线：`use_semantic_branch=False`
- 仅源域语义对齐：`use_semantic_branch=True, semantic_tgt_weight=0`
- 仅目标域一致性：`use_semantic_branch=True, semantic_src_weight=0`
- 第四章完整：`semantic_src_weight>0` 且 `semantic_tgt_weight>0`

语义先验层级也可做消融：

- 仅粗粒度：`semantic_bank_coarse.npy`
- 仅源域细粒度：`semantic_bank_fine_src.npy`
- 仅目标域细粒度：`semantic_bank_fine_tgt.npy`
- 融合：`semantic_bank_combined.npy`
