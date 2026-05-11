# agent.md

## 项目开发规范

### 注释语言

**统一使用中文注释**，包括：
- 文件头 docstring
- 类和函数的 docstring
- 行内解释注释
- TODO / FIXME 注释

例外：纯技术术语（HTTP、SQL、API、TLS 等）可直接使用英文，不翻译。

### 代码风格

- Python 遵循 PEP 8，line-length ≤ 100
- 类名使用 PascalCase
- 函数名、变量名使用 snake_case
- 类型注解完整，不使用 `Any` 除非绝对必要
- 所有 async 函数内部必须实际使用 `await`，禁止假 async

### 异常处理

- 业务异常使用自定义 Exception 类，携带 code 和 message
- 不捕获后压制异常（至少要 log）
- 对外接口统一返回错误码，不直接暴露异常堆栈

### 测试规范

- 每个功能至少有一个正常流程测试 + 一个异常/边界测试
- 测试文件名：`test_<模块名>.py`
- 使用 pytest-asyncio 处理异步测试
- Mock 真实外部依赖（DB、外部 API），不Mock同层模块

### Git 提交规范

```
<type>: <简短描述>

type: feat | fix | refactor | test | docs | chore
```

示例：
```
feat: 接入 MySQL 登录认证
fix: 修复会话超时后 token 未清除问题
test: 新增 auth service 登录成功用例
```

---

### API 设计规范

#### 错误码区间

| 区间 | 含义 |
|------|------|
| 1xxx | 参数错误 / 参数校验失败 |
| 2xxx | 业务逻辑错误（如用户名密码错误、无权限） |
| 5xxx | 外部系统错误（数据库、缓存、第三方 API 超时） |
| 9xxx | 未预期异常（需记录日志） |

#### 版本控制

- API 版本放入 URL path：`/v1/`、`/v2/`
- 不做 URL 兼容切换时，先发新版本再废弃旧版本

#### 响应结构

所有接口统一包装为：
```json
{ "code": 0, "data": ..., "msg": "成功" }
```
失败时 `data` 为 `null`。

---

### 数据库规范

- 字段命名：统一 `snake_case`，与 Python 变量名一致
- 主键：一律使用 `bigint auto_increment`
- 数据删除：一律用 `is_delete = 1` 打标记，不执行 `DELETE`
- `status` 字段含义由业务层解释，数据库只存原始值
- 每条 SQL 必须带 WHERE 条件，禁止全表更新 / 全表删除

---

### 敏感信息安全

- 密码 / Token / Secret 一律不进日志
- 密码存储：bcrypt，成本因子 ≥ 12
- `.env` 文件必须在 `.gitignore` 中列出，不提交到仓库
- 外部系统密钥通过环境变量注入，不硬编码在代码里

---

### 异步编程规范

- 所有 DB 操作、外部 HTTP 调用必须用 `async` / `await`
- 禁止在 async 函数里用 `time.sleep()`，用 `asyncio.sleep()` 替代
- async 函数不得返回 `None` 后再判空，应直接返回 `Optional[T]`

---

### 文档要求

- 每个工具函数 docstring 不少于一句话，说明输入输出
- 每个新模块必须有对应测试文件（`test_<模块名>.py`）
- 新增接口需在 `doc/阶段设计/` 下更新接口清单

---

### Code Review 关注点

每次 PR 必查：
1. 是否有硬编码凭证（密码、Token、密钥）
2. 是否有未捕获的异常导致 500 错误
3. 是否有性能回退（如同步阻塞调用未包装为 async）
4. 新增接口是否在文档中同步更新