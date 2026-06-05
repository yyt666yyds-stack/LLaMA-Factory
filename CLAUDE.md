# CLAUDE.md — Fine-tuning Qwen3-8B for AIFriends 客服 Agent

## 你在做什么

你在为 **AIFriends** 项目微调 **Qwen3-8B-Instruct**，让它成为一个能调用工具的智能客服 Agent。

### 整体架构

```
AIFriends 聊天应用 (Django + Vue)
  └─ LangGraph Agent (5节点: Router→Planner→Executor(ReAct)→Reflector→Finalizer)
       ├─ get_time              — 时间日期查询
       ├─ web_search            — 搜索引擎检索
       ├─ search_knowledge_base — 内部知识库/LanceDB向量检索
       └─ code_interpreter      — 数学计算/数据处理
```

微调后的 LoRA 权重会被部署回 AIFriends 后端，替换 DeepSeek API 调用，实现本地推理。

### 本项目 (LLaMA-Factory fork)

这是 https://github.com/hiyouga/LLaMA-Factory 的 fork，专门用来微调 Qwen3-8B。
- GitHub: `yyt666yyds-stack/LLaMA-Factory` (main 分支)
- GPU 服务器路径: `/mnt/workspace/LLaMA-Factory`

## 数据集

三个数据集都在 `data/` 目录下，配置在 `data/dataset_info.json`：

| 数据集 | 文件 | 内容 |
|--------|------|------|
| `customer_service` | `cs_simple.jsonl` | 纯文本客服对话（无工具调用） |
| `msagent` | `msagent_simple.jsonl` | MSAgent 通用对话 |
| `tool_calling` | `tool_calling_samples.jsonl` | **你刚生成的 2018 条工具调用数据** |

### tool_calling 数据分布（2018 条）
- `get_time`: 171 条 — 100+ 种时间问法
- `web_search`: 524 条 — 65 个搜索主题
- `search_knowledge_base`: 657 条 — 85 个知识库主题
- `code_interpreter`: 779 条 — 15 种计算类型（折扣/分期/统计/换算等）
- 多轮对话: 329 条 — 工具链组合（搜索→知识库、连环计算等）
- 直接聊天: 94 条

### 数据格式 (ShareGPT format)
```json
{
  "conversations": [
    {"from": "human", "value": "帮我查一下退货政策"},
    {"from": "function_call", "value": "[{\"name\": \"search_knowledge_base\", \"arguments\": {\"query\": \"退货 政策\"}}]"},
    {"from": "observation", "value": "查询结果：退货政策的详细信息已找到。"},
    {"from": "gpt", "value": "为您查到了退货政策的详细信息。"}
  ],
  "system": ""
}
```

### 生成新数据
在本地运行 `python data/generate_tool_data.py` 重新生成 `data/tool_calling_samples.jsonl`，然后 git push。

## 训练配置

使用 **QLoRA** (4-bit 量化 + LoRA)：

```yaml
# examples/train/custom/qwen3_8b_cs_qlora_v2.yaml
模型: Qwen3-8B-Instruct (模板: qwen3_nothink)
量化: 4-bit
LoRA: rank=16, alpha=32, dropout=0.05, target=all 模块
数据集: customer_service + msagent + tool_calling (3个混合)
cutoff: 2048 tokens
epochs: 3
batch: 2 × 8(梯度累积) = 有效 batch 16
lr: 5e-5, cosine schedule, 5% warmup
验证: 5% split, 每 200 步评估
输出: saves/qwen3-8b-customer-service-v2/
```

v1 配置 (`qwen3_8b_cs_qlora.yaml`) 不含 tool_calling 数据集，v2 是加了工具调用的版本。

## 启动训练

```bash
cd /mnt/workspace/LLaMA-Factory

# 拉取最新数据
git pull myfork main

# 启动训练
DISABLE_VERSION_CHECK=1 WANDB_DISABLED=true llamafactory-cli train \
  examples/train/custom/qwen3_8b_cs_qlora_v2.yaml
```

## 训练后：导出 & 部署

```bash
# 导出 LoRA 为合并模型
llamafactory-cli export \
  --model_name_or_path /mnt/workspace/models/Qwen/Qwen3-8B-Instruct \
  --adapter_name_or_path saves/qwen3-8b-customer-service-v2/checkpoint-best \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir /mnt/workspace/models/qwen3-8b-cs-tool-v2
```

导出的合并模型放到 `/mnt/workspace/models/` 下，AIFriends 后端通过 OpenAI-compatible API (vLLM) 调用。

## 常见操作

### 恢复训练
```bash
# 从 checkpoint 恢复
llamafactory-cli train examples/train/custom/qwen3_8b_cs_qlora_v2.yaml \
  --resume_from_checkpoint saves/qwen3-8b-customer-service-v2/checkpoint-1000
```

### 查看训练日志
```bash
# TensorBoard（如果 plot_loss: true）
tensorboard --logdir saves/qwen3-8b-customer-service-v2/
```

### 快速测试模型
```bash
# 交互式对话测试
llamafactory-cli chat examples/train/custom/qwen3_8b_cs_qlora_v2.yaml
```

### 切换数据集组合
在 yaml 里改 `dataset` 字段：
- 只训客服: `dataset: customer_service`
- 客服+工具: `dataset: customer_service, msagent, tool_calling`

### GPU 内存不够
- 降低 `per_device_train_batch_size: 1`
- 提高 `gradient_accumulation_steps` 保持有效 batch 不变
- 降低 `cutoff_len: 1024`

## 关联项目

- **AIFriends 主项目**: `F:\code\Myproject\AIFriends` (本地) — Django+Vue 聊天应用
  - Agent 定义: `agent/graph.py` (LangGraph 5节点)
  - 工具定义: `agent/tools.py` (get_time, search_knowledge_base, web_search, code_interpreter)
  - RAG 检索: `rag/` (LanceDB 向量库)
  - 后端: `examples/aifriends/backend/` (Django + DRF)
  - 前端: `examples/aifriends/frontend/` (Vue3 + Vite + Pinia)
- **微调模块**: `ft/` (embedding/reranker 微调数据)
