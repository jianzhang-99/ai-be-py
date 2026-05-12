# Slot 槽位标注规范（BIO Annotation Specification）

## 1. 概述

本文档定义第一阶段意图识别模型的槽位抽取标注规范，采用 BIO 标注体系。标注员必须严格按此规范执行，确保槽位边界一致。

### 1.1 BIO 标注体系说明

- **B-XXX**：某类实体的开始（Beginning）
- **I-XXX**：某类实体的延续（Inside）
- **O**：不属于任何实体的普通 token（Outside）

### 1.2 第一阶段槽位标签

| 槽位标签 | 中文说明 | 示例 |
|----------|----------|------|
| ship_name | 船名 | 俞垛79、华航118、长江之星6号 |
| area_name | 区域/附近范围 | 南京港附近、江北一带 |
| port_name | 港口/码头 | 南京龙潭港、重庆果园港 |
| route_from | 起点/装货地 | 南京、镇江港 |
| route_to | 终点/卸货地 | 重庆、武汉阳逻 |
| cargo_name | 货名 | 砂石、煤炭、钢材 |
| cargo_weight | 吨位/数量 | 3000吨、5000吨左右、1万吨 |
| date_time | 时间/装期 | 明天、月底、3月15号 |

---

## 2. 各槽位标注规则

### 2.1 ship_name — 船名

**标注范围**：船舶名称，包括正式船名、简称、船舶编号。

**标注原则**：
- 船名通常由数字、汉字组成，如"俞垛79"、"华航118"
- 带"号"、"轮"、"号"等后缀的也是船名
- 需要结合上下文判断是否是船名（如"南京7"可能是船名也可能是地名）

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 俞垛79在哪 | B-ship_name I-ship_name | 船名连续两个字 |
| 华航118位置 | B-ship_name I-ship_name I-ship_name | 含数字船名 |
| 长江之星6号 | B-ship_name I-ship_name I-ship_name I-ship_name | 多字船名 |
| 上次那艘船到哪了 | B-ship_name I-ship_name I-ship_name | "上次那艘船"整体作为船名指代 |

**反例**：

| 输入 | 错误标注 | 正确标注 | 原因 |
|------|----------|----------|------|
| 南京7 | B-ship_name I-ship_name | B-area_name I-port_name | 歧义，应结合上下文 |
| 南京港附近有没有船 | 南京港 | B-area_name I-port_name | "南京港"是港口，不是船名 |

---

### 2.2 area_name — 区域/附近范围

**标注范围**：表示某个区域、范围、"附近"等模糊地理位置。

**标注原则**：
- 带"附近"、"一带"、"周边"等词的通常标注为 area_name
- 单纯的地名+港组合应标注为 port_name
- 江、河、航段等水系名称也可以是 area_name

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 南京港附近有没有船 | B-area_name I-port_name I-area_name | "南京港附近"整体 |
| 长江下游有空的船吗 | B-area_name I-area_name | "长江下游"是区域 |
| 江北一带有船吗 | B-area_name I-area_name | "江北一带"是区域 |

---

### 2.3 port_name — 港口/码头

**标注范围**：具体港口名称、码头名称、港口气息。

**标注原则**：
- 带"港"字的基本都是 port_name
- 常见港口：南京港、南通港、镇江港、重庆港等
- 与 area_name 的区别：port_name 是具体港口，area_name 是更宽泛的区域

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 南京龙潭港明天天气 | B-port_name I-port_name I-port_name | 具体港口 |
| 南通港有船吗 | B-port_name I-port_name | 港口 |
| 镇江港可以加油吗 | B-port_name I-port_name | 港口 |

---

### 2.4 route_from — 起点/装货地

**标注范围**：航线的起点位置，装货地。

**标注原则**：
- 常与"到"连用，表示从 A 到 B
- 可能出现在"从 X"、"X 到 Y"、"去 X"等表达中

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 南京到南通 | B-route_from I-route_to | 航线 |
| 从武汉到南京 | B-route_from I-route_to | "从"引导起点 |
| 武汉到南京 | B-route_from I-route_to | 航线 |

---

### 2.5 route_to — 终点/卸货地

**标注范围**：航线的终点位置，卸货地。

**标注原则**：
- 与 route_from 配对出现
- "到"后面的是 route_to

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 南京到南通 | B-route_from I-route_to | 航线 |
| 武汉到重庆 | B-route_from I-route_to | 航线 |

---

### 2.6 cargo_name — 货名

**标注范围**：货物名称、货物类型。

**标注原则**：
- 常见货名：砂石、煤炭、钢材、矿石、集装箱、散货等
- 货名通常出现在吨位前面
- 注意区分货名和包装类型（如"集装箱"是货名还是包装？）

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 砂石5000吨 | B-cargo_name I-cargo_name | 货名+吨位 |
| 煤炭运费 | B-cargo_name I-cargo_name | 货名 |
| 钢材价格 | B-cargo_name I-cargo_name | 货名 |
| 集装箱运价 | B-cargo_name I-cargo_name | 货名 |

---

### 2.7 cargo_weight — 吨位/数量

**标注范围**：货物重量、船舶载重吨位、数量。

**标注原则**：
- 包含数字 + 单位（吨、TEU、立方米等）
- "5000吨"、"3000吨左右"、"1万吨"都算 cargo_weight
- 注意：如果说的是船舶载重吨，而不是货物重量，仍标注为 cargo_weight

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 5000吨砂石 | B-cargo_weight I-cargo_name | 吨位+货名 |
| 3000吨左右 | B-cargo_weight I-cargo_weight | 带模糊量词 |
| 1万吨 | B-cargo_weight I-cargo_weight | 大数字 |
| 8000吨的船 | B-cargo_weight I-cargo_weight | 船舶载重吨 |

---

### 2.8 date_time — 时间/装期

**标注范围**：时间表达，包括具体日期、相对时间、时间范围。

**标注原则**：
- 相对时间：明天、后天、月底、下周
- 具体日期：3月15号、5月1日
- 时间范围：本周、下周
- 如果时间与装卸货相关（如"明天装"），仍标注为 date_time

**正例**：

| 输入 | 标注结果 | 说明 |
|------|----------|------|
| 明天装货 | B-date_time I-date_time | 相对时间 |
| 月底发货 | B-date_time I-date_time | 月底 |
| 3月15号到港 | B-date_time I-date_time | 具体日期 |
| 下周能到吗 | B-date_time I-date_time | 时间范围 |

---

## 3. BIO 标注示例

### 3.1 完整示例

**示例1：标准航线查询**

```
输入：南京 到 南通 5000吨 砂石料 明天 装
标注：B-route_from O B-route_to O B-cargo_weight O B-cargo_name I-cargo_name B-date_time O
```

**示例2：区域找船**

```
输入：南京港 附近 有没有 3000吨 的 船
标注：B-port_name I-port_name I-area_name O O B-cargo_weight O O O
```

**示例3：查船位置**

```
输入：俞垛79 在哪
标注：B-ship_name I-ship_name O O
```

**示例4：带上下文继承**

```
历史：[USER] 查船 俞垛79 [ASSISTANT] 已为您查到船舶俞垛79的位置
当前：俞垛79 在哪
标注：B-ship_name I-ship_name O O
（当前 query 中的船名仍标注为 ship_name，历史信息不重复标注）
```

---

## 4. 复杂场景标注规则

### 4.1 多实体嵌套

同一词可能同时属于多个槽位，优先标注更具体的标签。

```
输入：南京港附近
标注：B-area_name I-port_name I-area_name
分析：南京港是 port_name，但"附近"是 area_name，整体作为 area_name
实际：统一标注为 B-area_name I-port_name I-area_name，因为"附近"扩展了范围
```

### 4.2 歧义处理

**"南京7"类歧义**：

```
输入：南京7
分析：可能是船名"南京7号"，也可能是地名"南京七里"
规则：当无法判断时，标注为 port_name + need_clarify=true
标注：B-port_name I-port_name
```

**"俞垛在哪里"歧义**：

```
输入：俞垛在哪里
分析：
- 无上下文：无法确定是船名还是地名，触发澄清
- 有历史（查过俞垛79）：继承为 QUERY_SHIP

规则：在标注时记录上下文状态
有历史标注：B-ship_name I-ship_name O O
无历史标注：标注为 need_clarify=true 样本
```

### 4.3 实体截断

当实体跨度过长时，按最合理的实体边界切分。

```
输入：从南京龙潭港到南通港
标注：B-route_from I-port_name I-port_name O B-route_to I-port_name I-port_name
说明："南京龙潭港"是 route_from，不是 port_name（作为装货地）
```

---

## 5. 标注质量控制

### 5.1 Token 对齐策略

中文分词容易切错实体，训练时使用 tokenizer 后的 token 对齐策略：

1. **先分词**：用中文分词器（如 jieba）做预分词
2. **再对齐**：将 BIO 标签对齐到分词后的词序列
3. **未登录词**：未登录词按字符级标注

### 5.2 实体级评估要求

Entity F1 按完整实体评估，不只看单字是否命中。

```
预测实体：[南京到南通, 5000吨, 砂石]
标注实体：[南京, 南通, 5000吨, 砂石料]

实体 F1 = 2 * Precision * Recall / (Precision + Recall)
- 预测正确的完整实体数 = 1（5000吨）
- 预测的实体总数 = 3
- 标注的实体总数 = 4
- Entity_Precision = 1/3
- Entity_Recall = 1/4
- Entity_F1 ≈ 0.27
```

### 5.3 一致性检查

- 标注完成后，使用脚本自动检查：
  - BIO 标签是否连续（B-XXX 后必须有 I-XXX 或 O，不能跳到另一个 B-XXX）
  - 标签分布是否合理
  - 实体边界是否与分词结果对齐

---

## 6. JSONL 输出格式

标注完成的样本保存为 JSONL，格式如下：

```json
{
  "id": "sample_0001",
  "history": [
    {"role": "user", "content": "查船 俞垛79"},
    {"role": "assistant", "content": "已为您查到船舶俞垛79的位置"}
  ],
  "query": "俞垛79预计什么时候到港",
  "slots": {
    "ship_name": "俞垛79",
    "date_time": "预计到港时间"
  },
  "bio_labels": {
    "ship_name": [0, 1, 2],
    "date_time": [5, 6]
  }
}
```

说明：
- `slots`：完整实体字典，用于评估
- `bio_labels`：每个 token 的 BIO 标签 index，用于训练（与 tokenizer 输出对齐）

---

## 7. 更新记录

| 版本 | 日期 | 更新内容 | 负责人 |
|------|------|----------|--------|
| v1.0 | 2026-05-12 | 初稿创建，定义全部 8 个槽位标签及 BIO 标注规则 | 待定 |