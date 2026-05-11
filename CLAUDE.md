# CLAUDE.md

本项目是 AI-dundun 的后端服务（ai-be-py），基于 FastAPI + LangGraph 构建航运助手。

## 项目背景

- **第一阶段**：能跑能演示，SSE 流式对话，意图识别，静态工具注册
- **第二阶段**（当前）：MySQL 持久化、Redis 会话缓存、运吨吨 Pilot API / 大数据 API 接入、动态工具注册
- **第三阶段**：Milvus RAG、三层记忆、Prometheus 可观测性、测试覆盖率 > 80%、Docker 部署

详细见 `doc/阶段设计/README.md` 和 `agent.md`。

## 快速熟悉路径

```
1. backend/auth/service.py       — 认证核心逻辑（login/logout/authenticate_token）
2. backend/auth/middleware.py   — JWT Bearer 中间件
3. backend/auth/deps.py         — FastAPI 依赖注入
4. backend/api/routers/auth.py  — 登录/退出/当前用户接口
5. backend/infra/db.py          — 数据库兼容层（同步/异步两套接口）
6. backend/infra/database/      — 仓储层（sys_user_repository 等）
7. tests/unit/test_auth_service.py — 认证服务单元测试
8. doc/阶段设计/               — 各阶段技术方案
```

## 开发流程约定

### 测试驱动（必须遵循）

1. **先理解功能目标**，再开始写测试
2. **先输出测试计划**，再执行测试
3. 覆盖：正常流程、异常流程、边界条件、状态切换、回归影响
4. 发现问题必须给出：问题描述、复现步骤、实际结果、预期结果、严重程度
5. **无法实际执行测试时不要假装测过**，明确说明是"测试设计"而不是"已测试"
6. 输出格式：功能理解 → 风险点 → 测试计划 → 测试结果 → 问题列表 → 剩余风险

### 代码规范

- 注释语言：**统一使用中文**（agent.md 明确规定）
- Python：PEP 8，line-length ≤ 100，PascalCase 类名，snake_case 函数/变量名
- 所有 async 函数内部必须实际使用 `await`，禁止假 async
- 业务异常使用自定义 Exception 类，携带 code 和 message
- 密码 / Token / Secret 一律不进日志

### 数据库

- 字段命名：snake_case
- 主键：bigint auto_increment
- 数据删除：一律用 `is_delete = 1` 打标记，不用 DELETE
- 每条 SQL 必须带 WHERE 条件，禁止全表更新/全表删除

### API 设计

- 错误码：1xxx 参数错误，2xxx 业务逻辑错误，5xxx 外部系统错误，9xxx 未预期异常
- 响应结构：`{ "code": 0, "data": ..., "msg": "成功" }`
- API 版本放入 URL path：`/v1/`、`/v2/`

## 本地开发

### 启动服务

```bash
cd /Users/liangjiajian/Desktop/AI-dundun/ai-be-py
source .venv/bin/activate
python -m backend.main
```

### 运行测试

```bash
pytest tests/ -v
```

### 环境变量

`.env` 文件必须在 `.gitignore` 中，不提交到仓库。参考 `.env.example`。

## Code Review 关注点

每次 PR 必查：
1. 是否有硬编码凭证（密码、Token、密钥）
2. 是否有未捕获的异常导致 500 错误
3. 是否有性能回退（如同步阻塞调用未包装为 async）
4. 新增接口是否在文档中同步更新

## 项目根目录结构

```
ai-be-py/
├── agent.md              # 项目开发规范（本文件同目录）
├── CLAUDE.md             # 项目特定约定（你在这里）
├── backend/
│   ├── auth/             # 认证模块
│   ├── api/              # API 路由
│   ├── infra/            # 基础设施（数据库、Redis、工具）
│   └── main.py           # 服务入口
├── doc/阶段设计/         # 各阶段技术方案文档
└── tests/                # 测试（unit/integration）
```
