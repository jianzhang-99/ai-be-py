# 意图识别模型训练

BERT 三头联合模型：意图分类 + 槽位 BIO 标注 + 澄清判断。

## 数据目录

```
train/
├── data/
│   ├── case/               # 原始 case（未处理）
│   ├── processed/          # 处理后数据集
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   └── realCase/           # 真实样本（按 intent 拆分）
├── scripts/
│   ├── preprocess.py       # 数据预处理：合并/校验/划分
│   ├── import_real_samples.py  # 真实话术样本解析
│   ├── train.py            # 训练脚本
│   └── evaluate.py         # 评测脚本
└── outputs/
    └── checkpoints/         # 模型输出
```

## 快速开始

### 1. 安装依赖

```bash
cd /Users/liangjiajian/Desktop/AI-dundun/ai-be-py
pip install -e ".[train]"
```

### 2. 准备数据

```bash
# 一键合并 case/ 目录下的所有 JSONL，校验并划分为 train/valid/test
python -m train.scripts.preprocess \
    --mode build \
    --input_dir ./train/data/case \
    --output ./train/data/processed
```

### 3. 训练

```bash
source .venv/bin/activate

python -m train.scripts.train \
    --data_dir ./train/data/processed \
    --train_file train.jsonl \
    --valid_file valid.jsonl \
    --max_epochs 5 \
    --batch_size 16 \
    --max_length 256 \
    --device cpu
```

### 4. 评测

```bash
python -m train.scripts.evaluate \
    --model_path ./train/outputs/checkpoints/best_model.bin \
    --test_file ./train/data/processed/test.jsonl
```

## 数据格式

每行一个 JSONL 样本：

```json
{
  "id": "real_00001",
  "history": [],
  "query": "华航118什么时候到港",
  "label": {
    "intent": "QUERY_SHIP",
    "slots": {"ship_name": "华航118"},
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
| FEEDBACK | 反馈投诉 |
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

模型保存在 `train/outputs/{version}/checkpoints/`，包含：
- `best_model.bin` — 模型权重
- `tokenizer/` — 分词器
- `metadata.json` — 训练元信息

评测结果保存在 `train/data/processed/evaluation_results.json`。