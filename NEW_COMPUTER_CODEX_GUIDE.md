# OpenManus 新电脑接手说明

这份说明是给新电脑上的 Codex 用的，目标只有一个：快速理解这个仓库、知道从哪里启动、以及改动时该看哪些地方。

## 项目是什么

OpenManus 是一个基于 LLM 的通用 Agent 项目。核心能力包括：

- 通过本地工具执行任务
- 通过 MCP 连接外部工具
- 通过浏览器、代码编辑、Python 执行等工具完成多步任务
- 可选地使用 sandbox / Daytona 环境跑更隔离的任务
- 有一个 planning flow，用来把复杂任务拆成步骤再执行

## 最重要的入口

- `main.py`：最常用入口，启动 `Manus`
- `run_flow.py`：启动多 agent planning flow，默认只带 `Manus`
- `run_mcp.py`：启动 MCP 相关运行方式
- `sandbox_main.py`：启动 sandbox 版 agent

## 代码结构

- `app/agent/`：各类 agent 实现
  - `manus.py`：主 agent
  - `sandbox_agent.py`：sandbox 版主 agent
  - `data_analysis.py`：数据分析 agent
  - `mcp.py`：MCP agent
  - `browser.py`、`toolcall.py`、`base.py`：agent 基础能力
- `app/tool/`：工具实现
  - `python_execute.py`、`bash.py`、`web_search.py`
  - `browser_use_tool.py`、`str_replace_editor.py`
  - `mcp.py`：MCP client 工具封装
  - `sandbox/`：sandbox 专用工具
  - `chart_visualization/`：数据可视化相关工具
- `app/flow/`：多 agent flow
  - `planning.py`：规划与执行流程
  - `flow_factory.py`：flow 工厂
- `app/config.py`：统一配置加载
- `config/`：配置模板
- `examples/`：示例用例
- `tests/`：测试

## 运行前提

- Python `>= 3.12`
- 推荐用 `uv` 或 `conda`
- 如果要用浏览器能力，通常需要 `playwright install`
- 如果要用 sandbox / Daytona / MCP，需要额外配置对应服务

## 配置规则

项目默认读取：

1. `config/config.toml`
2. 如果没有，就回退到 `config/config.example.toml`

所以新电脑上通常先做这件事：

```bash
cp config/config.example.toml config/config.toml
```

然后改里面的 LLM 配置。`app/config.py` 会把这些字段加载成运行时配置：

- `llm`
- `llm.vision`
- `browser`
- `search`
- `sandbox`
- `mcp`
- `runflow`
- `daytona`

MCP 服务器配置则默认看：

- `config/mcp.json`

如果不存在，就不会自动连接任何 MCP server。

## 启动方式

最常用：

```bash
python main.py
```

规划 flow：

```bash
python run_flow.py
```

MCP runner：

```bash
python run_mcp.py
```

sandbox 版：

```bash
python sandbox_main.py
```

## 当前实现的行为

### `Manus`

`app/agent/manus.py` 里的 `Manus` 是主 agent。

它默认带这些本地工具：

- Python 执行
- 浏览器工具
- 字符串替换编辑器
- 向人提问
- 终止任务

如果 `config/mcp.json` 里有 MCP server，会在创建 agent 时自动连上，并把对应工具加进工具集合。

### `PlanningFlow`

`app/flow/planning.py` 会先让模型生成计划，再按步骤执行。

`run_flow.py` 里默认只有 `Manus`，如果 `config.run_flow_config.use_data_analysis_agent = true`，就会额外加入 `DataAnalysis`。

### `SandboxManus`

`app/agent/sandbox_agent.py` 会先创建 Daytona sandbox，再挂载 sandbox 工具：

- browser
- files
- shell
- vision

所以这个分支更适合需要隔离执行环境的任务。

## 改代码时优先看哪里

- 想加新工具：`app/tool/`
- 想加新 agent：`app/agent/`
- 想改多步调度：`app/flow/`
- 想改配置加载：`app/config.py`
- 想看主 prompt：`app/prompt/`

## 调试建议

- 先跑 `python main.py` 验证基础链路
- 再确认 `config/config.toml` 里的模型、`base_url`、`api_key`
- 如果浏览器相关功能报错，先检查 `playwright` 和本机 Chrome / Chromium
- 如果 MCP 不生效，检查 `config/mcp.json`
- 如果 sandbox 不工作，检查 Daytona 相关配置和账号

## 给新电脑上的 Codex 的工作习惯

1. 先看 `main.py`、`app/config.py`、`app/agent/manus.py`
2. 再看你要改的工具或 flow
3. 只动相关模块，不要顺手重构整条链路
4. 改完后优先跑最短路径验证
5. 涉及配置或外部服务时，先确认本机是否真的装好了

## 一句话总结

这个仓库本质上是“LLM + 工具调用 + flow 编排 + 可选 MCP / sandbox”的 Agent 框架。新电脑上的 Codex 只要先看配置、入口和 agent/tool/flow 这三层，就能很快接上手。
