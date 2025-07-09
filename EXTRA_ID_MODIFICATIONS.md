# Extra ID Token 训练修改说明

本文档说明了对 LLaMA Factory 框架的三个主要修改，以支持 `<extra_id_1>`, `<extra_id_2>` 等特殊 token 的训练。

## 修改内容

### 1. 冻结 Extra ID Token 的 Embedding（前 K 步）

**功能**：在训练的前 K 步中冻结 `<extra_id_1>`, `<extra_id_2>` 等 token 的 embedding 层。

**实现**：
- 新增参数：`freeze_extra_id_steps`
- 新增回调函数：`ExtraIdFreezeCallback`
- 通过循环生成 `<extra_id_{i}>` 格式并使用 tokenizer.encode() 精确检测 token
- 支持检测 `<extra_id_0>` 到 `<extra_id_999>` 范围内的所有 token
- 在指定步数后自动解冻

**配置示例**：
```yaml
freeze_extra_id_steps: 100  # 前100步冻结extra_id token的embedding
```

### 2. 预热数据训练

**功能**：先训练预热数据，再训练正式数据。

**实现方法**：
通过数据集配置文件实现，无需修改代码。在 `dataset_info.json` 中按顺序配置数据集：

```json
{
  "warmup_data": {
    "file_name": "warmup_data.json",
    "formatting": "alpaca"
  },
  "main_data": {
    "file_name": "main_data.json", 
    "formatting": "alpaca"
  }
}
```

然后在训练配置中指定：
```yaml
dataset: warmup_data,main_data
```

### 3. Extra ID Token 的损失权重调整

**功能**：为输出中的 `<extra_id_1>`, `<extra_id_2>` 等 token 设置不同的损失权重。

**实现**：
- 新增参数：`extra_id_loss_weight`
- 修改 `CustomSeq2SeqTrainer.compute_loss()` 方法
- 通过 tokenizer.encode() 精确识别 extra_id token 并应用权重
- 其他 token 保持权重为 1.0

**配置示例**：
```yaml
extra_id_loss_weight: 0.1  # extra_id token的损失权重为0.1，其他token为1.0
```

## 使用方法

### 1. 配置文件设置

创建训练配置文件（如 `extra_id_config.yaml`）：

```yaml
### 基础配置
model_name_or_path: your_model_path
stage: sft
do_train: true
finetuning_type: lora

### Extra ID Token 特殊配置
freeze_extra_id_steps: 100      # 前100步冻结embedding
extra_id_loss_weight: 0.1        # extra_id token损失权重

### 其他训练参数
dataset: your_dataset
output_dir: ./saves/your_model
# ... 其他参数
```

### 2. 启动训练

```bash
python src/train.py --config examples/train_sft/extra_id_config.yaml
```

或者使用命令行参数：

```bash
python src/train.py \
    --model_name_or_path your_model_path \
    --stage sft \
    --do_train \
    --dataset your_dataset \
    --freeze_extra_id_steps 100 \
    --extra_id_loss_weight 0.1 \
    --output_dir ./saves/your_model
```

## 修改的文件列表

1. **参数定义**：
   - `src/llamafactory/hparams/finetuning_args.py`：新增 `freeze_extra_id_steps` 和 `extra_id_loss_weight` 参数

2. **回调函数**：
   - `src/llamafactory/train/callbacks/extra_id_callback.py`：新增 embedding 冻结回调
   - `src/llamafactory/train/callbacks/__init__.py`：导出回调函数

3. **训练器修改**：
   - `src/llamafactory/train/sft/trainer.py`：修改损失计算逻辑
   - `src/llamafactory/train/sft/workflow.py`：集成回调函数

4. **示例配置**：
   - `examples/train_sft/extra_id_config.yaml`：使用示例

## 注意事项

1. **Token 检测**：系统通过 tokenizer.encode() 方法精确检测 `<extra_id_0>` 到 `<extra_id_999>` 范围内的 token，确保准确性，无需手动指定

2. **兼容性**：这些修改与现有的训练方法（LoRA、Freeze、Full）完全兼容

3. **性能影响**：
   - Embedding 冻结对性能影响很小
   - 自定义损失计算会有轻微的计算开销

4. **调试信息**：训练开始时会打印检测到的 extra_id token 数量和相关配置信息

## 验证方法

训练开始时，查看日志输出：

```
INFO - Found 10 extra_id tokens: [32000, 32001, 32002, ...]
INFO - Added ExtraIdFreezeCallback for 100 steps
INFO - Froze embeddings for 10 extra_id tokens for 100 steps
```

在第 100 步时：

```
INFO - Unfroze extra_id token embeddings at step 100
```

这表明修改已正确生效。