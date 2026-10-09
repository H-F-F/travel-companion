# AI Travel Companion（智能旅行管家）

用一句话描述需求，自动查天气、找景点、排路线、看车票，生成可执行的多日行程。基于 **LLM Agent 对话式规划 + 多源数据整合**的智能出行系统，前后端一体运行。

## 功能特性

1. **对话式智能规划**：点击「AI 对话规划」，用自然语言描述需求，LLM Agent 基于 Function Calling 自动编排地理编码、天气、POI、车站、行程生成等工具，多轮追问补全出发日期 / 城市 / 预算等信息后输出完整方案；未配置或调用失败时自动降级到规则引擎，功能始终可用。
2. **分步引导式向导**：顶部导航（AI 对话 / 开始规划 / 我的行程）+ 四步向导（出发与车次 → 兴趣点 → 酒店选择 → 生成确认），天气自动刷新、候选项分页浏览。
3. **多源数据整合**：高德地图（地理编码 / POI / 酒店）、Open-Meteo（天气）、OSM Overpass（POI 兜底）、12306 MCP（车站 / 直达 / 中转票务）。
4. **可执行行程**：景点 / 美食「想去 / 想吃 / 不感兴趣」二次筛选，结合天气、预算、距离、口味偏好、室内外属性、车次到达时间与酒店位置，生成多日 itinerary（含每日动线、天气提示与预算建议）。
5. **可靠性工程**：多数据源降级链、外部服务熔断、POI 进程级缓存、HTTPS 连接复用与并发抓取、异步线程池隔离，单次行程生成耗时从分钟级降至秒级，且不阻塞其他请求。
6. **历史行程与会话持久化**：生成的行程自动持久化到 SQLite（「我的行程」随时回看 / 复用 / 删除）；对话会话持久化，刷新页面后对话不丢失、可继续；每次 AI 调用落盘 AI Trace 日志（模式 / 策略 / 质量分 / Token / 耗时可回溯）；一键导出为自包含 HTML 旅行攻略（可打印 / 转 PDF）。

## 技术栈

- 后端：Python 3.10+ · FastAPI · Uvicorn
- AI：OpenAI 兼容 LLM API（Function Calling 编排工具，支持通义千问 / DeepSeek / 智谱 / 豆包 Ark 等）
- 数据：高德地图 API · Open-Meteo · OSM Overpass · 12306 MCP
- 前端：原生 HTML / CSS / JavaScript（单页应用，无构建依赖）

## 快速开始

### 1. 安装依赖

```powershell
pip install -r backend/requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写（`.env` 已被 `.gitignore` 忽略，不会提交）：

```powershell
Copy-Item .env.example .env
```

最低可用配置（不配任何 key 也能跑，POI 使用内置城市数据兜底）：

```env
# 高德 Web 服务 Key（推荐）：让景点 / 美食 / 酒店 / 地理编码使用真实数据
AMAP_API_KEY=your_amap_web_key

# LLM 规划 Agent（可选）：配置后对话模式使用大模型编排工具生成行程
# 任意 OpenAI 兼容端点均可，例如通义千问：
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=your_dashscope_key
LLM_MODEL=qwen-plus

# 12306 MCP（可选）：本地 MCP 服务地址，未配置或不可用时自动跳过票务查询
MCP_12306_URL=http://127.0.0.1:8001/mcp
```

### 3. 启动（单服务跑全栈：API + 前端页面）

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### 4. 访问

打开 http://127.0.0.1:8000/ 即可使用。

### 5. 运行测试（可选）

```powershell
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -v
```

覆盖行程持久化（保存 / 列表 / 删除 / date 序列化）、会话持久化（保存续写 / 列表 / 删除 / 失效回退）、HTML 攻略导出（字段完整性 / HTML 注入转义）与引擎参数校验（日期必填、天数边界、预算回退），不依赖外部网络与 LLM。

### 6. Docker 一键部署（可选）

需要 Docker + Docker Compose。先复制 `.env.example` 为 `.env`（可全部留空，零密钥也能跑）：

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

打开 http://localhost:8000/ 即可使用。说明：

- 密钥只通过 `env_file` 在运行时注入，不写入镜像；`.dockerignore` 已排除 `.env`。
- 行程数据库（`data/plans.db`）与 AI Trace 日志（`logs/`）通过卷挂载持久化，容器重建不丢失。
- 镜像内置 `/health` 健康检查（30s 间隔，3 次失败视为不健康）。

停止 / 重启：

```powershell
docker compose down        # 停止
docker compose up -d       # 重新启动
```

## 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `AMAP_API_KEY` | 否 | 高德 Web 服务 Key，用于地理编码、POI、酒店搜索 |
| `LLM_BASE_URL` | 否 | OpenAI 兼容 LLM 端点（如 DashScope、DeepSeek） |
| `LLM_API_KEY` | 否 | LLM 密钥 |
| `LLM_MODEL` | 否 | 模型名（如 `qwen-plus`、`deepseek-chat`） |
| `LLM_TIMEOUT` | 否 | LLM 请求超时秒数，默认 30 |
| `MCP_12306_URL` | 否 | 12306 MCP 服务地址 |
| `MCP_12306_TIMEOUT_SEC` | 否 | 12306 查询超时秒数，默认 12 |
| `MCP_12306_CIRCUIT_SEC` | 否 | 12306 熔断时长，默认 60 |

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/chat` | 对话式规划：`{message, history[]}` → `{reply_text, steps[], plan, mode}`，`mode ∈ {llm, fallback, needs_input}` |
| `POST` | `/api/v1/plan` | 完整行程生成（车票 + 候选 + 酒店 + 多日路线） |
| `POST` | `/api/v1/preview` | 候选项预览（第一步缓存用，含天气 / 交通 / 候选） |
| `POST` | `/api/v1/plan-check` | 生成前校验（天数 / 车次 / 酒店可行性） |
| `GET` | `/api/v1/geocode` | 地理编码 `?address=&city=` |
| `GET` | `/api/v1/weather` | 天气查询 `?address=&days=3` |
| `GET` | `/api/v1/search-poi` | POI 搜索 `?q=&kind=attraction\|food` |
| `GET` | `/api/v1/transport/stations` | 车站搜索 |
| `GET` | `/api/v1/transport/hubs` | 目的地枢纽 / 车站候选 |
| `GET` | `/api/v1/plans` | 历史行程列表 |
| `GET` | `/api/v1/plans/{id}` | 历史行程详情（含生成参数） |
| `DELETE` | `/api/v1/plans/{id}` | 删除历史行程 |
| `GET` | `/api/v1/plans/{id}/export` | 导出 HTML 旅行攻略（自包含，可打印 / 转 PDF） |
| `POST` | `/api/v1/chat` | 对话式规划（传 `session_id` 续写会话，否则新建并返回） |
| `GET` | `/api/v1/sessions` | 对话会话列表（标题 / 消息数，不含完整消息） |
| `GET` | `/api/v1/sessions/{id}` | 会话详情（完整消息，用于刷新后恢复对话） |
| `DELETE` | `/api/v1/sessions/{id}` | 删除对话会话 |

## LLM Agent 架构

```
用户一句话 → LLM（Function Calling 编排，最多 8 轮）
              ├─ geocode           定位目的地 / 出发城市
              ├─ get_weather       查询多日天气
              ├─ search_pois       搜索景点 / 美食候选
              ├─ search_stations   查询出发 / 到达车站
              └─ generate_itinerary 生成可执行多日行程
失败 / 未配置 → 规则引擎（自然语言参数提取 + 启发式规划），服务不中断
```

前端在对话面板中逐步展示每个工具的调用结果（步骤卡片），最终行程卡可直接预览。

## 可靠性设计

- **降级链**：LLM → 规则引擎；高德 → OSM → 内置城市数据（北京 / 上海 / 广州 / 成都 / 西安等）。
- **熔断**：高德 Key 无效快速失败、12306 MCP 不可用 60s 熔断、OSM 不可达 1h 熔断，避免反复等待外部服务超时。
- **缓存**：POI 结果 24h 进程级缓存，重复生成秒级返回。
- **性能**：高德 HTTPS 连接复用 + POI 并发抓取；`/plan`、`/preview` 等耗时接口经 `asyncio.to_thread` 隔离，生成期间不阻塞健康检查与其他请求。
- **代理兼容**：服务启动时清理系统代理环境变量，避免本地代理残留导致外部请求挂起。

## 项目结构

```
backend/
  app/
    main.py                 # FastAPI 入口：API 路由 + 静态前端挂载
    agent.py                # LLM Agent 编排（5 个 function-calling 工具 + 降级 + AI Trace）
    llm_client.py           # OpenAI 兼容 LLM 客户端
    engine.py               # 规则引擎：行程生成 / 评分 / 降级链
    amap_client.py          # 高德地图客户端（连接复用 + Key 熔断）
    poi_client.py           # OSM Overpass / Nominatim 兜底数据源
    weather_client.py       # Open-Meteo 天气客户端
    train_12306_client.py   # 12306 MCP 客户端（熔断）
    station_catalog.py      # 内置车站目录
    recommendation.py       # 评分 / 推荐模型
    models.py               # Pydantic 请求模型
    storage.py              # 持久化（SQLite：行程 plans + 对话会话 sessions）
    export.py               # 行程导出：自包含 HTML 旅行攻略渲染
frontend/
  index.html                # 单页前端（导航 + 向导 + 结果页 + 对话面板 + 我的行程）
logs/
  ai-trace.log              # AI Trace 日志（每次对话落盘一条 JSONL，运行时生成）
data/
  plans.db                  # 行程持久化数据库（运行时生成，不入库）
```

## 项目文档

- [项目大纲](./智能旅行管家-项目大纲.md) · [需求说明](./项目需求说明.md) · [路线图](./项目路线图.md) · [任务清单](./任务清单.md)
- [决策记录](./决策记录.md) · [问题记录](./问题记录.md) · [经验教训](./经验教训.md) · [风险登记](./风险登记.md)

## 当前状态

- v0.5.0：对话会话持久化（chat 支持 `session_id` 续写，刷新后对话可恢复；sessions 列表 / 详情 / 删除 API）。
- v0.4.1：新增 Docker / Docker Compose 一键部署（含健康检查、卷持久化、密钥运行时注入）。
- v0.4.0：对话回复打字机流式效果（流式视觉）；新增 pytest 自动化测试套件（9 项，覆盖存储 / 导出 / 引擎校验）。
- v0.3.0：新增行程持久化（SQLite）、AI Trace 日志、HTML 攻略导出与「我的行程」面板。
- v0.2.0：LLM Agent 对话式规划、分步引导式前端（顶部导航 + 四步向导）、多源数据降级与熔断、生成性能优化（分钟级 → 秒级）。
- 建议下一步：目标日期未开售的票务预测策略、天气突发重规划、SSE 真实流式输出。
