# BERT 意图识别模型训练

## 目录结构

```
train/
├── data/
│   ├── raw/              # 原始数据（未处理）
│   └── processed/        # 处理后的 JSONL 数据
│       ├── train.jsonl
│       ├── valid.jsonl
│       ├── test.jsonl
│       └── ambiguity_test.jsonl
├── scripts/
│   ├── preprocess.py    # 数据预处理脚本
│   ├── train.py         # 训练脚本
│   └── evaluate.py      # 评测脚本
├── outputs/
│   └── checkpoints/     # 模型输出目录
└── README.md            # 本文档
```

## 快速开始

### 1. 安装训练依赖

```bash
cd /Users/liangjiajian/Desktop/AI-dundun/ai-be-py
pip install -e ".[train]"
```

### 2. 准备数据

按 `doc/技术文档/意图识别模型/07-JSONL数据集格式规范.md` 格式准备数据，放入：

```
train/data/processed/
├── train.jsonl    # 训练集
├── valid.jsonl     # 验证集
└── test.jsonl      # 测试集
```

如果你已经有多个按 intent 拆分的 JSONL 文件，可以一键合并、校验、划分：

```bash
python -m train.scripts.preprocess \
    --mode build \
    --input_dir ./train/data/原始 \
    --output ./train/data/processed \
    --pattern "*.jsonl"
```

只校验某个文件：

```bash
python -m train.scripts.preprocess \
    --mode validate \
    --input ./train/data/processed/train.jsonl
```

### 3. 生成示例数据（可选，用于测试流程）

```bash
cd /Users/liangjiajian/Desktop/AI-dundun/ai-be-py

# 生成 50 条示例数据（用于快速测试训练流程）
python -m train.scripts.preprocess --mode generate --intent QUERY_SHIP --output ./train/data/processed/train.jsonl --count 50
```

### 4. 开始训练

```bash
# 激活环境
source .venv/bin/activate

# 训练（使用 CPU 或 CUDA）
python -m train.scripts.train \
    --data_dir ./train/data/processed \
    --train_file train.jsonl \
    --valid_file valid.jsonl \
    --max_epochs 5 \
    --batch_size 16 \
    --max_length 256 \
    --device cpu
```

### 5. 评测

```bash
python -m train.scripts.evaluate \
    --model_path ./train/outputs/checkpoints/best_model.bin \
    --test_file ./train/data/processed/test.jsonl
```

## 数据格式

每行一个 JSONL 样本：

```json
{
  "id": "sample_000001",
  "history": [
    {"role": "user", "content": "查船 俞垛79"},
    {"role": "assistant", "content": "已为您查到船舶俞垛79的位置"}
  ],
  "query": "俞垛在哪里",
  "label": {
    "intent": "QUERY_SHIP",
    "slots": {"ship_name": "俞垛79"},
    "need_clarify": false
  }
}
```

## 意图标签（14 个）

| Intent | 说明 |
|--------|------|
| DOC_QA | 文档问答 |
| FIND_SHIP | 找船 |
| SAVE_ORDER | 发布运单 |
| QUERY_ORDER | 查询订单 |
| QUERY_SHIP | 查船位置 |
| QUERY_FREIGHT | 运价查询 |
| QUERY_WEATHER | 天气查询 |
| QUERY_WATER_LEVEL | 水位查询 |
| DISPATCH_MONITOR | 在途监控 |
| IMAGE_OCR | 图片识别 |
| FEEDBACK | 反馈 |
| QUERY_OIL_STATION | 加油站查询 |
| QUERY_SHIP_INFO | 船舶档案 |
| TALK | 闲聊 |

## 槽位标签（8 个）

| Slot | 说明 |
|------|------|
| ship_name | 船名 |
| area_name | 区域 |
| port_name | 港口 |
| route_from | 起点 |
| route_to | 终点 |
| cargo_name | 货名 |
| cargo_weight | 吨位 |
| date_time | 时间 |

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --max_epochs | 5 | 训练轮数 |
| --batch_size | 16 | 批大小 |
| --lr | 2e-5 | 学习率 |
| --device | cpu | 设备（cpu/cuda） |
| --max_length | 256 | 最大序列长度 |

## 输出

训练完成后，模型保存在：

```
train/outputs/checkpoints/best_model.bin
train/outputs/checkpoints/metadata.json
```

评测结果保存在：

```
train/data/processed/evaluation_results.json
```

## 训练代码说明

当前训练代码已经按方案实现三头联合任务：

- `intent_head`：预测 14 类 intent。
- `slot_head`：预测 `O/B-xxx/I-xxx` BIO 槽位标签，使用 fast tokenizer 的 `offset_mapping` 做字符到 token 对齐。
- `clarify_head`：预测 `need_clarify`，用于拦截信息不足或多意图冲突的问题。

评估脚本会输出 `Intent Accuracy`、`Intent Macro F1`、`Intent Weighted F1`、`Clarify Precision/Recall/F1`、`Slot Entity F1`、`Exact Match`、`Context Inheritance Rate`，并在结果 JSON 中保留最多 50 条 bad case 供人工复盘。
