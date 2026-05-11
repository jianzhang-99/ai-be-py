# AI-BE4J 原始流程梳理

## 1. 文档目标

这份文档用于整理 `ai-be-master` 当前的原始业务流程，作为后续 `ai-be-py` 使用 `LangGraph` 重构时的基线。

这里关注的不是 Java 代码细节，而是：

- 请求从哪里进入
- 鉴权和用户上下文如何注入
- 核心业务如何分层流转
- 哪些流程适合后续映射成 LangGraph 节点
- 哪些流程仍应保留在 service / repository / tool 层

## 2. 原始系统总体分层

从当前 `ai-be-master` 来看，核心分层可以概括为：

```text
Controller / API
  -> Interceptor 鉴权与上下文注入
  -> Core Facade
  -> Intent / Router / FlowEngine
  -> Tool / LLM / Knowledge / Memory / DB
  -> Result
```

其中有两类典型入口：

- 后台管理入口
  - 路径通常为 `/auth`、`/admin/**`、`/ai/**`
- App 入口
  - 路径通常为 `/ai-api/**`

## 3. 请求入口与鉴权原始流程

## 3.1 Web 拦截链

入口配置见：

- `common/config/WebConfig.java`

当前拦截顺序大致为：

1. `LogInterceptor`
2. `PilotAuthInterceptor`
   - 负责 `/ai-api/**`
   - 从请求头读取 `userId`、`phone`
3. `DefaultAuthInterceptor`
   - 负责后台路径
   - 从 `Sa-Token` 读取登录用户
4. `WebInvokeTimeInterceptor`

公开路径中明确放行的有：

- `/**/login`
- `/ai/chat/history/share/**`
- Swagger / 文档 / test

## 3.2 后台鉴权上下文注入

关键实现见：

- `common/interceptor/DefaultAuthInterceptor.java`

原始流程如下：

1. 请求进入后台接口
2. `StpUtil.isLogin()` 判断是否登录
3. 未登录则直接返回：

```json
{
  "code": 401,
  "msg": "未登录或登录已过期，请重新登录",
  "data": null
}
```

4. 已登录则从 `Sa-Token` 中取：
   - `loginId -> userId`
   - `session.phone -> phone`
5. 将 `userId`、`phone` 写入：
   - `MDC`
   - `request attribute`
6. Controller 中通过 `SystemUserUtils` 获取当前用户

这个机制说明：

- Java 版聊天接口本身不接收前端传入的 `userId`
- 用户上下文主要由鉴权层统一注入

## 4. 登录原始流程

入口见：

- `manager/controller/AuthController.java`
- `manager/service/impl/AuthServiceImpl.java`

原始流程如下：

1. `POST /auth/login`
2. 校验手机号格式和密码非空
3. 查询 `sys_user`
   - 条件：`phone = ? and is_delete = false`
4. 校验用户存在且未禁用
5. 用 `PasswordEncoder` 校验 BCrypt 密码
6. 更新 `last_login_time`
7. 调用 `StpUtil.login(userId)` 建立登录态
8. 把 `phone` 写入 Sa-Token Session
9. 返回：
   - `userId`
   - `phone`
   - `token`

退出登录流程较简单：

1. `POST /auth/logout`
2. 若当前已登录则执行 `StpUtil.logout()`
3. 清理 MDC
4. 返回成功

## 5. AI 对话主链路原始流程

这是后续迁移到 `LangGraph` 时最重要的链路。

关键入口见：

- `manager/controller/ChatController.java`
- `manager/api/ChatApi.java`
- `core/AiRequestFacade.java`
- `core/InputProcessor.java`
- `core/IntentRecognizer.java`
- `core/SceneRouter.java`
- `core/FlowEngineProcessor.java`

## 5.1 后台对话入口 `/ai/chat`

原始流程如下：

1. 前端调用 `POST /ai/chat`
2. `ChatController` 从登录态注入：
   - `userId`
   - `phone`
3. 调用 `AiRequestFacade.processChatStream(aiReq)`
4. 返回 `Flux<AiResp>` 流式响应

App 侧 `/ai-api/chat` 也是同一主链路，只是用户上下文来自 `PilotUserUtils`

## 5.2 `AiRequestFacade` 统一入口流程

`AiRequestFacade` 是 Java 版聊天主链路的统一编排入口，原始流程如下：

1. 生成 `requestId`
2. 调用 `InputProcessor.preprocess(aiReq)`
3. 调用 `IntentRecognizer.resolveSceneAndIntent(aiReq, ctx, requestId)`
4. 调用 `SceneRouter.route(scene, ctx, intentMeta)`
5. 将结果组装为 `SceneContext`
6. 绑定 LLM trace 上下文
7. 调用 `FlowEngineProcessor.executeStream(sceneContext)`
8. 捕获异常并返回错误流

可以理解为：

```text
Controller
  -> AiRequestFacade
    -> InputProcessor
    -> IntentRecognizer
    -> SceneRouter
    -> FlowEngineProcessor
```

## 5.3 `InputProcessor` 原始职责

`InputProcessor` 主要负责多模态输入预处理：

1. 读取原始输入文本
2. 遍历附件
3. 如果是图片，则调用 `IOcrService` 做 OCR
4. 将 OCR 结果拼回最终输入文本
5. 构建 `PreprocessContext`
   - `sessionId`
   - `userId`
   - `phone`
   - `rawInput`
   - `model`

这一步后续迁移时更适合做：

- LangGraph 入口前的预处理步骤
- 或单独 `input_node`

## 5.4 `IntentRecognizer` 原始职责

`IntentRecognizer` 不是单一规则函数，而是一层识别编排器。

原始职责包括：

1. 读取当前输入
2. 结合历史上下文
3. 结合工作记忆状态
4. 先走规则网关
5. 未命中再走 LLM 分类
6. 由最终决策器做守卫校正
7. 特殊场景下再做路由改写

这里包含的真实决策比表面更复杂，说明后续不能把“意图识别”简单理解成一个 prompt 调用。

后续迁移时建议拆成：

- `intent_context_build`
- `intent_rule_gate`
- `intent_llm_classify`
- `intent_finalize`

但在第一版 LangGraph 中，也可以先收敛成一个 `intent_node`

## 5.5 `SceneRouter` 原始职责

`SceneRouter` 当前职责比较轻，主要是把识别结果和上下文组装成 `SceneContext`：

- scene
- sessionId
- userId
- phone
- rawInput
- model
- intentRecognition

这说明 Java 版的“路由器”更像：

- 上下文装配器
- 场景状态初始化器

而不是复杂分发中心

## 5.6 `FlowEngineProcessor` 原始职责

`FlowEngineProcessor` 是真正的场景执行中心。

它的原始职责包括：

1. 根据 `scene` 分发到不同执行方法
2. 调用工具、LLM、知识增强、记忆、价格服务、地图服务等
3. 在统一出口处写入 `chat_log`
4. 返回流式 `AiResp`

当前显式分发的场景包括：

- `DOC_QA`
- `SAVE_ORDER`
- `QUERY_ORDER`
- `FIND_SHIP`
- `QUERY_SHIP`
- `QUERY_WEATHER`
- `QUERY_WATER_LEVEL`
- `QUERY_FREIGHT`
- `FEEDBACK`
- `TALK`

这意味着 Java 版的主链路是：

- 意图识别之后进入“单一大执行器”
- 执行器内部再按场景切方法

后续迁移到 LangGraph 时，这里正是最适合逐步拆 node 的位置

## 6. 聊天历史原始流程

关键入口见：

- `manager/controller/ChatHistoryController.java`
- `manager/api/ChatHistoryApi.java`
- `manager/service/impl/ChatLogServiceImpl.java`

## 6.1 分页查询历史

后台入口：

- `GET /ai/chat/history/page`

原始流程：

1. 从登录态读取 `phone`
2. 构造 `ChatLogReq(phone=当前手机号)`
3. 调用 `chatLogService.pageHistory(pageReq, req)`
4. Service 调用 `ChatLogMapper.selectPageHistory`
5. 返回分页结构 `PageResponse<ChatHistoryResp>`

这个接口是当前前端已经在请求的关键接口之一

## 6.2 根据 sessionId 查会话详情

后台入口：

- `GET /ai/chat/history/listBySessionId/{sessionId}`

原始流程：

1. 从登录态读取 `phone`
2. 查询 `chat_log`
   - `session_id = ?`
   - `phone = ?`
3. 按 `seq asc, id asc` 排序
4. 返回完整对话列表

## 6.3 分享态查看会话

后台入口：

- `GET /ai/chat/history/share/listBySessionId/{sessionId}`

原始流程：

1. 不要求登录
2. 按 `sessionId` 查询 `chat_log`
3. 按 `seq asc, id asc` 排序
4. 只返回分享需要的字段 `ChatShareResp`

## 6.4 删除会话

后台入口：

- `DELETE /ai/chat/history/deleteBySessionId/{sessionId}`

原始流程：

1. 从登录态读取 `phone`
2. 删除当前手机号名下该 `sessionId` 的对话记录
3. 返回成功

## 7. 后台高频 CRUD 模块原始模式

从 `UserController`、`PromptController`、`McpToolConfigController`、`KnowledgeBaseController`、`KnowledgeDocController` 来看，Java 版后台模块基本遵循同一种模式：

```text
Controller
  -> Convert
  -> Service
  -> MyBatis Plus / Mapper / 外部服务
```

## 7.1 用户管理 `UserController`

入口：

- `/admin/user/page`
- `/admin/user/list`
- `/admin/user/{id}`
- `POST /admin/user`
- `PUT /admin/user`
- `DELETE /admin/user/{id}`

原始模式：

1. DTO -> Entity
2. 调 `IUserService`
3. 基于 MyBatis Plus 做 CRUD
4. Entity -> Resp

这是典型单表后台管理模块

## 7.2 Prompt 管理 `PromptController`

相比普通 CRUD 更复杂，原始流程包含：

1. 基础提示词 CRUD
2. 变更写历史版本
3. 回滚指定版本
4. 清理 `PromptLoader` 缓存
5. 导入导出 JSON 配置

这说明后续迁移时：

- Prompt 不只是一个表
- 还包含“版本历史 + 缓存刷新”这两个业务动作

## 7.3 MCP 工具配置 `McpToolConfigController`

原始流程包含两类：

1. 普通配置 CRUD
2. 动态连接 MCP Server
   - 测试连接
   - 拉取工具列表

这说明该模块不是纯数据库模块，而是：

- 配置存储
- 外部连通性测试
- 动态工具发现

## 7.4 知识库 `KnowledgeBaseController`

原始流程包括：

1. 知识库基础 CRUD
2. 创建时补 collectionName
3. 删除时联动删除 Milvus collection

这说明知识库模块是“数据库 + 向量库”的双写联动模块

## 7.5 知识文档 `KnowledgeDocController`

原始流程包括：

1. 基础文档 CRUD
2. 文件上传到 OSS
3. 生成文档记录
4. 调用向量化服务 `KnowledgeDocVectorizeService`

这说明知识文档模块是：

- 数据库存元数据
- OSS 存原文件
- 向量服务做切片与入库

## 8. 对 LangGraph 迁移的直接启示

从原始流程看，不是所有 Java 模块都应该迁成 LangGraph

## 8.1 适合 LangGraph 的部分

适合放进 LangGraph 的主要是聊天主链路中的“会改变处理路径”的步骤：

1. 输入预处理
2. 意图识别
3. 场景路由
4. 工具调用
5. RAG 知识增强
6. 记忆读写
7. 回复生成

## 8.2 不适合直接做成 Graph node 的部分

以下内容更适合保留在普通 service / repository / tool 层：

1. 登录鉴权
2. 用户管理 CRUD
3. Prompt 后台管理
4. MCP 配置 CRUD
5. 聊天历史分页查询
6. 单表查询和普通后台分页

这些模块更像“传统后台能力”，不是 Agent 编排问题

## 9. 第二阶段迁移建议顺序

结合当前前端现状，建议按下面顺序迁移：

1. 登录与鉴权
   - 已基本打通
2. 聊天主链路
   - `/ai/chat`
3. 聊天历史
   - `/ai/chat/history/page`
   - `/ai/chat/history/listBySessionId/{sessionId}`
4. `QUERY_WEATHER` / `QUERY_SHIP` 这类高频场景
5. Prompt / MCP 管理端
6. 知识库与向量化
7. 记忆系统与更复杂的工作流

## 10. 推荐的 LangGraph 映射起点

基于当前 Java 原始流程，第一版建议先映射成：

```text
request
  -> input_node
  -> intent_node
  -> route_node
  -> scene_node
      -> tool_node
      -> rag_node
      -> memory_node
  -> response_node
  -> persist_node
```

其中：

- `input_node`
  - 对应 Java 的 `InputProcessor`
- `intent_node`
  - 对应 Java 的 `IntentRecognizer`
- `route_node`
  - 对应 Java 的 `SceneRouter`
- `scene_node`
  - 对应 Java 的 `FlowEngineProcessor` 分发入口
- `persist_node`
  - 对应统一落 `chat_log`

这样做的好处是：

- 先保留 Java 的流程语义
- 再逐步把 `FlowEngineProcessor` 里的大方法拆小
- 避免一开始就过度设计 Graph

## 11. 当前结论

当前 `ai-be-master` 的核心特点不是“多 Agent”，而是：

- 传统后台 + AI 主链路并存
- AI 主链路集中在 `AiRequestFacade -> IntentRecognizer -> SceneRouter -> FlowEngineProcessor`
- 后台管理类接口大多仍是普通 CRUD / Service 风格

因此 `ai-be-py` 后续迁移时应采用“双轨设计”：

1. 传统后台接口
   - 用 FastAPI + service + repository 正常迁移
2. AI 对话主链路
   - 用 LangGraph 做流程编排

这比把所有模块都 Graph 化更符合原系统结构
