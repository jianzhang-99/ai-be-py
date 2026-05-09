# 第一阶段 MVP 技术方案

## 1. 文档目标

定义 AI-BE-PY 第一阶段的技术落地范围。

这一阶段不追求功能齐全，而追求三件事：

1. 服务可以本地稳定启动
2. 主链路可以流式返回
3. 至少具备部分航运业务能力，证明 Python 重构方向可行

## 2. 第一阶段目标

### 2.1 业务目标

第一阶段交付一个可演示、可联调、可继续扩展的最小版本，支持：

- 健康检查
- SSE 流式对话
- 基础意图识别
- 通用闲聊 `TALK`
- 天气查询 `QUERY_WEATHER`
- 查船 `QUERY_SHIP`
- 运单信息抽取预览 `SAVE_ORDER`

### 2.2 技术目标

- 统一项目入口与配置加载
- 建立 FastAPI + LangGraph 主流程
- 建立 LLM 适配层
- 建立 Tool 注册与调用机制
- 明确状态模型和 SSE 事件协议
- 为第二阶段预留 RAG、Memory、持久化扩展位

## 3. 第一阶段范围

### 3.1 范围内

#### 基础能力

- `FastAPI` 服务启动
- `/health` 健康检查
- `/api/chat/stream` 流式接口
- `/api/chat` 非流式接口
- `.env` 配置加载

#### 编排能力

- LangGraph 主工作流
- 意图识别节点
- 路由节点
- 工具调用节点
- 响应生成节点

#### 场景能力

- `TALK`
- `QUERY_WEATHER`
- `QUERY_SHIP`
- `SAVE_ORDER`

#### 外部依赖

- 至少接入一个 LLM Provider
- 至少接入一个业务工具能力
- 允许通过 Mock 适配器替代暂未稳定的第三方服务

### 3.2 范围外

第一阶段明确不做：

- MySQL 持久化落库
- Redis 会话缓存
- Milvus RAG 检索
- 多层记忆系统
- 完整 MCP 动态配置中心
- 运价、找船列表、水位、知识库管理后台
- 后台任务队列
- 生产级权限体系

## 4. 为什么第一阶段只做这些

这是一个典型的 Agent 重构项目，第一阶段最容易失败的点不是“功能少”，而是“基础编排不稳”。

如果一开始就同时引入：

- LangGraph
- 多模型
- MCP
- Milvus
- Redis
- MySQL
- 一堆业务系统接口

那排错成本会非常高，最后很容易变成“文档看起来很完整，但项目跑不起来”。

所以第一阶段的设计原则是：

- 主链路先闭环
- 外部依赖先最小化
- 业务价值先可演示

## 5. 第一阶段架构裁剪

### 5.1 总体架构

```text
Client
  -> FastAPI Router
  -> ChatService
  -> ChatWorkflow
      -> intent_node
      -> routing_node
      -> tool_node / response_node
  -> SSE Event Stream
```

### 5.2 分层职责

| 层 | 第一阶段职责 | 暂不承担 |
|----|--------------|----------|
| API | 收请求、参数校验、输出 SSE | 复杂鉴权 |
| Service | 组织工作流调用 | 复杂业务聚合 |
| Graph | 场景编排、状态流转 | 多工作流并发调度 |
| Tools | 调业务工具和外部服务 | 动态工具市场 |
| Infra | LLM/配置/HTTP Client | 完整存储体系 |

## 6. 第一阶段建议目录落位

基于你当前的结构设计，第一阶段建议最少落这些文件：

```text
backend/
├── main.py
├── config.py
├── api/
│   ├── deps.py
│   ├── schemas.py
│   └── routers/
│       ├── chat.py
│       └── health.py
├── graph/
│   ├── state/
│   │   └── agent_state.py
│   ├── nodes/
│   │   ├── intent_node.py
│   │   ├── routing_node.py
│   │   ├── tool_node.py
│   │   └── response_node.py
│   └── workflows/
│       └── chat_workflow.py
├── services/
│   ├── chat_service.py
│   └── intent_service.py
├── tools/
│   ├── registry.py
│   ├── weather.py
│   ├── ship.py
│   └── order.py
└── infra/
    ├── llm/
    │   ├── client.py
    │   └── deepseek.py
    └── http/
        └── base.py
```

## 7. 第一阶段核心状态模型

建议第一阶段不要一上来就做很重的状态对象，只保留主链路必要字段。

```python
AgentState = {
    "request_id": str,
    "session_id": str,
    "user_id": str,
    "user_input": str,
    "history": list,
    "intent": dict | None,
    "scene": str | None,
    "tool_name": str | None,
    "tool_result": dict | list | str | None,
    "response_text": str | None,
    "error": str | None,
}
```

### 设计原则

- 第一阶段状态字段不超过 10 个
- 所有节点只写自己负责的字段
- 不在状态里塞 SDK 客户端、数据库连接等运行时对象

## 8. 第一阶段场景设计

### 8.1 TALK

#### 目标

验证最基础的 LLM 对话链路可用。

#### 实现方式

- 意图识别命中 `TALK`
- 直接走 `response_node`
- 调用 LLM 生成文本

#### 价值

- 可以验证 FastAPI、LangGraph、LLM、SSE 四段链路都已通

### 8.2 QUERY_WEATHER

#### 目标

验证“意图识别 -> 工具调用 -> 结果回传”的工具链路。

#### 第一阶段建议

- 第一优先：实现 `weather.py` 工具，直接通过 HTTP 或 MCP 查询天气
- 第二优先：如果第三方依赖未打通，先提供 Mock 天气结果

#### 返回形式

- 先返回文本总结
- 再返回结构化天气卡片或 JSON

### 8.3 QUERY_SHIP

#### 目标

验证航运领域的第一个真实业务场景。

#### 第一阶段建议

- 输入船名
- 调用 `ship.py`
- 返回单船基础信息

#### 第一阶段范围控制

- 先不做地图截图
- 先不做在途监控卡片
- 先只返回“船名、MMSI、船型、载重”等基础信息

这样可以降低对运吨吨截图和地图链路的耦合。

### 8.4 SAVE_ORDER

#### 目标

验证“从自然语言抽结构化数据”的能力。

#### 第一阶段建议

- 输入一段运单话术
- 调用 `order.py`
- 返回结构化预览结果
- 不真正创建运单

#### 范围边界

第一阶段只做：

- 装货地
- 卸货地
- 装货日期
- 货物名称
- 货量
- 单位

不做：

- 真正保存到外部运吨吨系统
- 多船信息补全
- 二次确认工作流

## 9. 第一阶段 Tool 设计

### 9.1 Tool 注册机制

第一阶段推荐做静态注册，不做动态发现。

```python
TOOL_REGISTRY = {
    "query_weather": WeatherTool(),
    "query_ship": ShipTool(),
    "save_order_preview": OrderExtractTool(),
}
```

### 9.2 Tool 接口约定

```python
class BaseTool(Protocol):
    name: str

    async def run(self, user_input: str, state: dict) -> dict:
        ...
```

### 9.3 为什么先静态注册

- 调试成本低
- IDE 友好
- 对第一阶段足够
- 第二阶段再平滑升级为 registry + factory

## 10. 第一阶段接口设计

### 10.1 健康检查

- `GET /health`
- 返回运行状态、版本、时间

### 10.2 非流式对话

- `POST /api/chat`
- 用于调试和简单联调

### 10.3 流式对话

- `POST /api/chat/stream`
- 使用 `text/event-stream`

建议统一事件类型：

- `intent`
- `tool_start`
- `tool_result`
- `response`
- `done`
- `error`

## 11. 第一阶段外部依赖策略

### 11.1 LLM

第一阶段只保留一个主 Provider，建议：

- 默认 `DeepSeek`
- 预留 Provider 抽象

原因：

- 避免一开始就做多模型路由
- 先把调用、超时、异常处理稳定下来

### 11.2 业务外部系统

第一阶段采用“适配器 + 可 Mock”策略：

| 能力 | 第一阶段策略 |
|------|--------------|
| 天气 | 真实接口优先，失败可 Mock |
| 查船 | 真实接口优先，失败可 Mock |
| 运单抽取 | 本地 LLM 生成即可 |

### 11.3 存储

第一阶段全部做成可选依赖：

- 没有 MySQL 也能跑
- 没有 Redis 也能跑
- 没有 Milvus 也能跑

## 12. 第一阶段异常处理

### 12.1 分层原则

| 层 | 异常策略 |
|----|----------|
| Router | 参数校验失败直接返回 4xx |
| Service | 包装业务异常 |
| Graph Node | 写入 `state["error"]` |
| Tool | 统一抛 `ToolExecutionError` |
| LLM Client | 统一抛 `LLMError` |

### 12.2 SSE 错误输出

即使出错，也尽量输出标准事件：

```json
{
  "event": "error",
  "data": {
    "message": "ship tool timeout"
  }
}
```

然后显式发一个 `done` 结束。

## 13. 第一阶段可观测性要求

第一阶段不做复杂监控，但至少要有：

- request_id
- session_id
- 场景识别日志
- tool 调用耗时
- LLM 调用耗时
- 错误日志

建议日志字段：

| 字段 | 说明 |
|------|------|
| `request_id` | 单次请求唯一标识 |
| `session_id` | 会话标识 |
| `scene` | 当前场景 |
| `tool_name` | 工具名 |
| `latency_ms` | 耗时 |

## 14. 第一阶段完成定义

当以下条件全部满足时，第一阶段算完成：

1. `uvicorn backend.main:app --reload` 能成功启动
2. `/health` 返回正常
3. `/api/chat` 能返回非流式结果
4. `/api/chat/stream` 能稳定输出 SSE
5. `TALK` 场景可用
6. `QUERY_WEATHER` 可用
7. `QUERY_SHIP` 或 `SAVE_ORDER` 至少有一个真实业务场景可用
8. 有基础单元测试和至少一条集成测试

## 15. 结论

第一阶段不是“做一个简化版大系统”，而是“做一个能跑、能演示、能继续长出来的最小系统”。

从技术管理角度，这一阶段最重要的不是功能数量，而是建立以下基础共识：

- 目录结构是否合理
- 编排方式是否可持续
- Tool 模式是否顺手
- 状态模型是否够稳
- 外部依赖是否可控

如果这几个问题在第一阶段验证通过，第二阶段扩展业务深度就会顺很多。

