# 数据集格式规范（Dataset Format Specification）

## 1. JSONL 格式说明

所有训练/验证/测试数据集统一采用 JSONL（JSON Lines）格式，每行一个 JSON 对象。

### 1.1 基本格式

```json
{"id": "sample_0001", "history": [], "query": "俞垛79在哪", "label": {"intent": "QUERY_SHIP", "slots": {"ship_name": "俞垛79"}, "need_clarify": false}}
{"id": "sample_0002", "history": [], "query": "帮我查一下", "label": {"intent": "TALK", "slots": {}, "need_clarify": true}}
```

### 1.2 字段说明

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| id | string | 是 | 样本唯一标识，格式：`sample_XXXX` |
| history | array | 是 | 历史对话列表，每项包含 role 和 content |
| query | string | 是 | 当前用户输入 |
| label | object | 是 | 标注信息 |
| label.intent | string | 是 | 意图标签，必须是已定义的 14 个标签之一 |
| label.slots | object | 是 | 槽位实体字典，key 为槽位名，value 为实体值 |
| label.need_clarify | bool | 是 | 是否需要澄清 |
| label.clarify_question | string | 否 | 澄清问题内容，仅 need_clarify=true 时填写 |
| label.context_inherited | bool | 否 | 是否从上下文继承了信息 |
| label.note | string | 否 | 标注说明，用于歧义样本的标注解释 |

---

## 2. 字段详细定义

### 2.1 history 字段

历史对话列表，最多保留最近 3 轮。

```json
"history": [
  {"role": "user", "content": "查船 俞垛79"},
  {"role": "assistant", "content": "已为您查到船舶俞垛79的位置"}
]
```

**role 可选值**：
- `user`：用户输入
- `assistant`：系统回复

### 2.2 label.slots 字段

槽位实体字典，可包含 0 到多个槽位。

```json
"slots": {
  "ship_name": "俞垛79",
  "route_from": "南京",
  "route_to": "南通",
  "cargo_weight": "5000吨",
  "cargo_name": "砂石",
  "date_time": "明天"
}
```

**槽位名与 label对应**：

| 槽位名 | 说明 |
|--------|------|
| ship_name | 船名 |
| area_name | 区域/附近范围 |
| port_name | 港口/码头 |
| route_from | 起点/装货地 |
| route_to | 终点/卸货地 |
| cargo_name | 货名 |
| cargo_weight | 吨位/数量 |
| date_time | 时间/装期 |

### 2.3 need_clarify 字段

表示模型输出不确定，需要向用户澄清。

```json
// 需要澄清的样本
{"need_clarify": true, "clarify_question": "请问您想查船、找船、查天气，还是处理运单？"}

// 不需要澄清的样本
{"need_clarify": false}
```

---

## 3. 特殊样本格式

### 3.1 上下文继承样本

当当前 query 依赖历史信息时，需标注 `context_inherited: true`：

```json
{
  "id": "sample_0088",
  "history": [
    {"role": "user", "content": "查船 俞垛79"},
    {"role": "assistant", "content": "已为您查到船舶俞垛79的位置"}
  ],
  "query": "俞垛在哪里",
  "label": {
    "intent": "QUERY_SHIP",
    "slots": {"ship_name": "俞垛79"},
    "need_clarify": false,
    "context_inherited": true,
    "note": "从上文继承船名俞垛79"
  }
}
```

### 3.2 歧义/澄清样本

当需要澄清时，`clarify_question` 字段必填：

```json
{
  "id": "sample_0102",
  "history": [],
  "query": "帮我查一下",
  "label": {
    "intent": "TALK",
    "slots": {},
    "need_clarify": true,
    "clarify_question": "请问您想查船、找船、查天气，还是处理运单？"
  }
}
```

### 3.3 无历史样本

`history` 为空数组：

```json
{
  "id": "sample_0001",
  "history": [],
  "query": "南京港明天天气",
  "label": {
    "intent": "QUERY_WEATHER",
    "slots": {"port_name": "南京港", "date_time": "明天"},
    "need_clarify": false
  }
}
```

---

## 4. 数据集划分标准

### 4.1 划分原则

| 原则 | 说明 |
|------|------|
| 覆盖全部 intent | 每个 intent 在训练、验证、测试中都要有样本 |
| 同类样本隔离 | 同一模板改写的相似样本不要同时出现在训练集和测试集 |
| 歧义样本独立 | 专门的歧义专项测试集固定不动 |
| 线上 bad case 优先 | 真实 bad case 优先放入测试集 |

### 4.2 数据集构成

```
dataset/
├── train.jsonl       # 训练集（1500~3000 条）
├── valid.jsonl       # 验证集（300~500 条）
├── test.jsonl        # 测试集（400~800 条）
└── ambiguity_test.jsonl  # 歧义专项测试集（100~200 条）
```

### 4.3 各数据集用途

| 数据集 | 文件名 | 数量 | 用途 |
|--------|--------|------|------|
| 训练集 | train.jsonl | 1500~3000 条 | 模型训练 |
| 验证集 | valid.jsonl | 300~500 条 | 调参、早停 |
| 测试集 | test.jsonl | 400~800 条 | 最终评测，长期跟踪 |
| 歧义专项集 | ambiguity_test.jsonl | 100~200 条 | 专门测试上下文和澄清能力 |

---

## 5. 样本分布要求

### 5.1 Intent 分布

每个 intent 的最低样本量：

| Intent | 训练集建议 | 验证集建议 | 测试集建议 |
|--------|------------|------------|------------|
| QUERY_SHIP | 250 条 | 50 条 | 50 条 |
| FIND_SHIP | 250 条 | 50 条 | 50 条 |
| QUERY_FREIGHT | 200 条 | 40 条 | 40 条 |
| TALK | 200 条 | 40 条 | 40 条 |
| DOC_QA | 120 条 | 25 条 | 25 条 |
| SAVE_ORDER | 120 条 | 25 条 | 25 条 |
| QUERY_ORDER | 80 条 | 20 条 | 20 条 |
| DISPATCH_MONITOR | 80 条 | 20 条 | 20 条 |
| QUERY_SHIP_INFO | 60 条 | 15 条 | 15 条 |
| QUERY_OIL_STATION | 50 条 | 10 条 | 10 条 |
| QUERY_WATER_LEVEL | 50 条 | 10 条 | 10 条 |
| IMAGE_OCR | 50 条 | 10 条 | 10 条 |
| FEEDBACK | 50 条 | 10 条 | 10 条 |
| QUERY_WEATHER | 80 条 | 20 条 | 20 条 |
| **总计** | **~1650 条** | **~345 条** | **~345 条** |

### 5.2 分布均匀性要求

- 同一 intent 的样本量不超过最高意图的 5 倍
- 训练集中 P0 场景（QUERY_SHIP、FIND_SHIP、QUERY_FREIGHT、TALK）占比不超过 60%

---

## 6. 数据质量要求

### 6.1 标签一致性

- 每条样本需经过双人标注
- Kappa 系数要求 >= 0.85
- Kappa < 0.85 的样本进入仲裁流程

### 6.2 格式校验脚本

```python
import json
import sys

def validate_jsonl(filepath: str) -> bool:
    """校验 JSONL 文件格式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    required_fields = {'id', 'history', 'query', 'label'}
    valid_intents = {'DOC_QA', 'FIND_SHIP', 'SAVE_ORDER', 'QUERY_ORDER',
                     'QUERY_SHIP', 'QUERY_FREIGHT', 'QUERY_WEATHER', 'QUERY_WATER_LEVEL',
                     'DISPATCH_MONITOR', 'IMAGE_OCR', 'FEEDBACK', 'QUERY_OIL_STATION',
                     'QUERY_SHIP_INFO', 'TALK'}

    errors = []
    for i, line in enumerate(lines, 1):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON decode error - {e}")
            continue

        # 检查必填字段
        missing = required_fields - set(sample.keys())
        if missing:
            errors.append(f"Line {i}: Missing fields {missing}")
            continue

        # 检查 intent 有效性
        intent = sample['label'].get('intent', '')
        if intent not in valid_intents:
            errors.append(f"Line {i}: Invalid intent '{intent}'")

        # 检查 need_clarify 为 true 时 clarify_question 必填
        if sample['label'].get('need_clarify') and not sample['label'].get('clarify_question'):
            errors.append(f"Line {i}: need_clarify=true but no clarify_question")

    if errors:
        for e in errors:
            print(e)
        return False

    print(f"✓ {filepath}: {len(lines)} samples validated")
    return True

if __name__ == "__main__":
    sys.exit(0 if all(validate_jsonl(f) for f in sys.argv[1:]) else 1)
```

---

## 7. 数据来源

### 7.1 样本来源优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | 历史真实对话 | 从线上系统提取真实业务话术 |
| 2 | Bad Case 回流 | 已知的误判/歧义样本 |
| 3 | 强关键词扩展 | 基于已有规则的相似表达 |
| 4 | 人工补充 | 标注员补充少数类和边界样本 |

### 7.2 样本扩写策略

对于样本量不足的 intent，可以：
1. **同义改写**：保持相同 intent，换不同表达方式
2. **槽位替换**：相同模板，不同实体值
3. **上下文扩展**：为同一 query 添加不同历史

---

## 8. 标注工具要求

### 8.1 功能需求

标注工具必须支持：
1. 显示当前 query 和历史上下文
2. 选择 intent 标签
3. 高亮标注槽位实体
4. 设置 need_clarify 和 clarify_question
5. 批量保存为 JSONL
6. 双人标注模式（标注员 A 和 B）
7. Kappa 系数自动计算

### 8.2 数据安全

- 标注过程不能看到测试集样本
- 标注完成的 JSONL 文件需加密存储
- 标注员权限分级管理

---

## 9. 文件结构模板

```
ai-be-py/
└── doc/
    └── 技术文档/
        └── 意图识别模型/
            └── 数据集/
                ├── 原始/
                │   ├── raw_train.jsonl      # 未标注原始数据
                │   ├── raw_valid.jsonl
                │   └── raw_test.jsonl
                ├── 标注完成/
                │   ├── train.jsonl          # 已标注
                │   ├── valid.jsonl
                │   └── test.jsonl
                ├── 歧义专项/
                │   └── ambiguity_test.jsonl
                └── schema.json             # 数据格式 schema
```

---

## 10. 更新记录

| 版本 | 日期 | 更新内容 | 负责人 |
|------|------|----------|--------|
| v1.0 | 2026-05-12 | 初稿创建，定义 JSONL 格式和数据集划分 | 待定 |