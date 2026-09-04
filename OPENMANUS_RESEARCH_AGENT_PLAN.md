# OpenManus 二次开发计划：Evidence-grounded Research Agent

> 面向：目标是大厂 Agent 算法实习的个人项目
>
> 基线仓库：https://github.com/FoundationAgents/OpenManus
>
> 最后更新：2026-08-28

## 1. 项目目标

基于 OpenManus 构建一个“可验证的深度研究 Agent”。用户输入一个需要多步调查的问题，Agent 自动完成：

1. 拆解研究问题并生成可执行计划；
2. 调用 Web Search、网页抓取和本地知识库进行多跳检索；
3. 对候选证据进行去重、相关性判断和可信度核验；
4. 在长任务中维护研究状态、已用证据和中间结论；
5. 生成带引用、可追溯的最终报告；
6. 输出完整轨迹，并用评测集量化成功率、引用质量、成本和延迟。

最终项目不是“给 OpenManus 加一个 RAG 工具”，而是一个可复现的 Agent 算法实验平台：有明确基线、有改进方法、有自动评测、有失败分析。

## 2. 为什么选择这个方向

OpenManus 已经提供了适合二次开发的基础设施：

- `app/agent/manus.py`：通用工具调用 Agent、MCP 连接和浏览器上下文；
- `app/agent/base.py`、`app/agent/react.py`：状态、记忆、执行循环和 ReAct 抽象；
- `app/flow/planning.py`：计划生成、步骤状态和执行器调度；
- `app/tool/web_search.py` 及 `app/tool/search/`：多搜索引擎接口；
- `app/tool/crawl4ai.py`、浏览器工具和沙箱：网页内容获取与受控执行；
- `app/schema.py`：消息、工具调用和 Agent 状态结构；
- `tests/`：已有沙箱测试，可作为工程质量起点。

你的 RAG 经验可以直接迁移到检索、重排、引用和知识库；项目新增的算法含量主要来自规划、多跳检索、证据验证、记忆、轨迹评测和成本优化。

## 3. 预期成果

### 3.1 可运行产品

- CLI 入口：输入问题，输出 Markdown 研究报告；
- 可选 Web/API 入口：提交任务、查看任务状态和轨迹；
- 每个结论都能回链到来源 URL、文档片段和证据 ID；
- 失败时能够重试、降级或明确报告“不足以支持结论”。

### 3.2 算法实验成果

至少完成以下对照实验：

| 实验 | 对照 | 改进 | 主要指标 |
|---|---|---|---|
| 规划 | 单轮直接搜索 | 子问题分解 + 依赖关系 | 任务成功率、步骤数 |
| 检索 | 单路向量检索 | BM25/向量混合 + 重排 | Recall@k、MRR、引用覆盖率 |
| 证据 | 直接生成 | 证据相关性/矛盾检测 | 引用正确率、幻觉率 |
| 记忆 | 全量历史拼接 | 摘要记忆 + 证据缓存 + 跨任务程序性记忆 | 长任务成功率、Token 成本 |
| 执行 | 固定重试 | 按错误类型恢复 | 工具成功率、平均延迟 |
| 模型路由 | 单模型执行全部阶段 | 按阶段难度路由到不同模型 | Token 成本、任务成功率、延迟 |

记忆与模型路由实验参考了 2026 年生产级框架的设计（Microsoft Agent Framework 的 Procedural Memory 与 Agent Harness、混合模型栈）。

目标数值只是开发期参考，最终以自建数据集和固定模型配置下的实测为准：

- 任务成功率提升至少 15%；
- 引用正确率达到 85% 以上；
- 关键结论引用覆盖率达到 90% 以上；
- 平均 Token 成本下降 20% 或平均延迟下降 20%；
- 所有失败案例都有分类和可复现记录。

## 4. 总体架构

```text
用户问题
   |
   v
Research Orchestrator
   |-- Planner：问题分解、依赖关系、停止条件
   |-- Retriever：Web / 本地知识库 / 混合检索
   |-- Evidence Verifier：相关性、来源质量、冲突检测
   |-- Synthesizer：带引用的阶段结论与最终报告
   |-- Memory：任务状态、证据缓存、摘要、失败记录、程序性记忆、上下文压缩
   |-- Evaluator：轨迹、指标、成本、回归测试
   |
   v
Markdown Report + citations + JSON trace + metrics
```

推荐先实现单 Agent + 明确工作流，再考虑多 Agent。对于这个项目，稳定的 Planner-Executor-Validator 流程比堆多个自主 Agent 更容易评测和解释。

## 5. 目录和模块规划

新增代码建议集中在 `app/research/`，尽量少改 OpenManus 原有通用模块：

```text
app/research/
  __init__.py
  models.py              # ResearchTask、ResearchStep、Evidence、Claim、Trace
  planner.py             # 计划生成和计划修订
  retriever.py           # Web、本地知识库、混合检索接口
  reranker.py            # 交叉编码器或 LLM 重排
  verifier.py            # 证据相关性、来源质量、冲突检测
  memory.py              # 任务记忆、证据缓存、摘要记忆、程序性记忆、上下文压缩
  skills/                # 各阶段提示词，按 Agent Skills 规范（SKILL.md）组织
    research-plan.md
    research-verify.md
    research-synthesize.md
  synthesizer.py         # 阶段结论、最终报告和引用格式
  orchestrator.py        # 主执行循环和恢复策略
  evaluator.py           # 指标计算和失败分类
  trace.py               # JSONL 轨迹与 Token/延迟记录
  config.py              # 研究 Agent 配置
  cli.py                 # 项目专用入口
tests/research/
  test_planner.py
  test_retriever.py
  test_verifier.py
  test_memory.py
  test_evaluator.py
data/research/
  tasks.jsonl            # 评测任务
  documents/             # 可选本地语料
  gold.jsonl             # 参考答案、关键事实和来源
  lessons.jsonl          # 跨任务程序性记忆（教训与有效策略）
outputs/research/        # 本地报告和轨迹，不提交 API 密钥
```

如需复用 OpenManus 工具，优先通过 `ToolCollection` 注册，不要把检索逻辑全部写进 `Manus` 类。若必须调整框架，记录每处修改的原因和兼容性影响。

## 6. 分阶段执行路线

### Phase 0：基线复现（1—2 天）

目标：确认环境和主流程可用，建立改动前基线。

- 配置 Python 3.12 虚拟环境和模型 API；
- 按 README 运行 `python main.py`；
- 运行 `python run_flow.py`，理解 `PlanningFlow`；
- 阅读 `app/agent/manus.py`、`app/agent/toolcall.py`、`app/flow/planning.py`、`app/tool/web_search.py`；
- 为 5 个固定研究问题保存原始运行结果、步骤数、耗时和 Token；
- 建立自己的 Git 分支：`codex/research-agent`。

验收：能画出一次任务的消息—工具—结果执行链；已有测试通过；基线结果可重复。

### Phase 1：结构化研究任务和轨迹（3—4 天）

目标：让 Agent 的内部状态可观察、可保存、可评测。

- 在 `app/research/models.py` 定义 Pydantic 数据模型：
  - `ResearchTask`：问题、约束、预算、状态；
  - `ResearchStep`：目标、输入、依赖、状态、重试次数；
  - `Evidence`：URL、标题、片段、来源质量、时间戳、哈希；
  - `Claim`：结论、支持证据、反证、置信度；
  - `TraceEvent`：LLM、工具、检索、验证、错误事件。
- 所有事件以 JSONL 写入 `outputs/research/`；
- TraceEvent 带 `span_id` / `parent_span_id`，按规划→检索→验证→综合的层级组织事件（OpenTelemetry 风格的轻量 trace）；
- 记录每次 LLM 调用的模型、输入/输出 Token、延迟和估算成本；
- 为每个任务生成唯一 `task_id`，支持中断后恢复。

验收：一次运行可以只靠 JSONL 重放关键步骤；敏感信息不会写入日志。

### Phase 2：检索和证据层（1 周）

目标：把你已有的 RAG 能力接入 Agent，并保证来源可追溯。

- 统一 `Retriever` 接口：`search(query, top_k, filters)`；
- 接入已有 Web Search 和网页正文抓取；
- 新增本地文档索引：Markdown/PDF/HTML 至少支持一种；
- 实现混合检索：BM25 + 向量检索，结果合并后去重；
- 加入重排器，先使用轻量交叉编码器，资源不足时提供 LLM 重排回退；
- 每条证据保留原文片段、来源、文档 ID、chunk ID 和检索分数；
- 防止提示词注入：网页内容只能作为不可信证据，不能覆盖系统指令。

验收：给定 20 个查询，输出 Recall@5、MRR、去重率和平均检索延迟；每条报告引用都能定位到证据对象。

### Phase 3：Planner-Executor-Validator（1 周）

目标：形成项目的核心 Agent 算法闭环。

- Planner 将复杂问题拆成 3—8 个子问题；
- 为步骤增加 `depends_on`、`success_criteria`、`max_retries` 和 `budget`；
- Executor 选择检索、抓取、分析和总结工具；
- Validator 在每个步骤后判断：是否完成、证据是否足够、是否需要补检索；
- Validator 的每次判断输出节点级指标（步骤完成度、证据充分性、工具选择是否合理），供 Phase 6 汇总；
- 对工具错误区分参数错误、网络错误、限流、空结果和内容质量问题；
- 当步骤失败时局部重试，不要重新执行整个任务；
- 设置停止条件：证据覆盖足够、预算耗尽、达到最大步骤或无法验证。

建议第一版采用结构化 JSON 计划，避免仅依靠自然语言计划。

验收：在 20 个多步任务上，成功生成计划并完成所有可验证步骤；循环次数、无效工具调用和节点级指标有统计。

### Phase 4：记忆和长任务能力（4—5 天）

目标：让 Agent 能处理长上下文和跨步骤依赖。

- 短期记忆：当前步骤最近消息；
- 任务记忆：计划、已完成步骤、未解决问题；
- 证据记忆：证据去重、缓存和引用关系；
- 摘要记忆：定期压缩历史，不直接无限拼接消息；
- 失败记忆：记录失败原因和已尝试策略，避免重复；
- 程序性记忆（跨任务，参考 Microsoft Agent Framework 的 Procedural Memory 设计）：每次运行结束，Validator 把有效/无效策略写入 `data/research/lessons.jsonl`；新任务开始时 Planner 检索相关教训注入计划；
- 上下文压缩：Token 达到阈值触发结构化压缩（而非固定步数），压缩产物保留 `[E#]` 证据 ID 引用，压缩前后做关键事实保留率检查；
- 对记忆召回做消融实验：无记忆、全量历史、摘要+证据缓存、+程序性记忆。

验收：长任务 Token 不随步骤数线性失控；压缩前后关键事实保留率 ≥ 90%；重复检索率和重复工具调用率下降；跨任务教训有实际复用。

### Phase 5：报告生成和事实核查（4—5 天）

目标：生成“可审计”的研究报告。

- Planner / Verifier / Synthesizer 的指令按 Agent Skills 规范组织为 `app/research/skills/*.md`（frontmatter 描述即触发条件），按需加载而非全部塞进 system prompt（渐进式披露，服务 Token 成本目标）；
- 报告结构：问题、方法、结论、证据、冲突观点、局限、参考来源；
- 结论和引用使用稳定的 `[E1]`、`[E2]` 证据 ID；
- 生成后执行 Claim-Evidence 检查：每个重要断言必须有证据；
- 检查引用是否真正支持断言，而不是仅主题相关；
- 对相互冲突的来源显式标注，不强行合并；
- 证据不足时输出“不确定”，不要让模型补写事实。

验收：自动检查报告中的引用存在性、覆盖率和支持关系；人工抽样评估引用正确率。

### Phase 6：系统评测和优化（1 周）

目标：把项目从 Demo 变成算法项目。

- 构造 50—100 个任务，覆盖事实查询、比较分析、时间约束、多跳关系和冲突来源；
- 每个任务标注：标准答案、关键事实、可接受来源、停止条件；
- 自动指标：
  - `task_success_rate`：任务是否满足成功条件；
  - `subtask_completion_rate`：子问题完成率；
  - `citation_coverage`：关键结论有引用的比例；
  - `citation_correctness`：引用是否支持对应断言；
  - `retrieval_recall@k`、`MRR`；
  - `tool_call_success_rate`、`invalid_call_rate`、`tool_selection_accuracy`；
  - `latency_p50/p95`、输入/输出 Token、估算成本；
  - `unsupported_claim_rate`、`duplicate_action_rate`。
- 新增模型路由消融：单模型 vs 按阶段难度路由（便宜模型做证据初筛、强模型做规划/冲突检测/综合）；
- 评测脚本支持固定随机种子、固定模型配置和结果 JSON 导出；
- 失败分类：检索不到、规划错误、工具错误、证据不足、综合幻觉、超时/预算。

验收：可以一条命令跑完整评测，并输出表格和失败样例；至少完成三项消融实验（建议含程序性记忆或模型路由）。

### Phase 7：工程化和简历打磨（3—5 天）

- 增加 `research_cli.py` 或 `python -m app.research.cli` 入口；
- 加入 `.env.example`，严禁提交真实 API Key；
- 使用异步并发处理互不依赖的检索步骤；
- 加入超时、指数退避、限流、缓存和幂等；
- 增加 Docker/Windows 启动说明；
- 为核心模块补单元测试和最小集成测试；
- 输出架构图、实验表、失败案例和 3 分钟演示视频；
- 在项目 README 中明确注明 OpenManus 原项目、许可证和你的改动范围。

## 7. 推荐的最小可行版本（MVP）

如果时间只有 2 周，按以下范围收敛：

1. 结构化计划；
2. Web Search + 网页抓取；
3. 证据对象和引用 ID；
4. Planner-Executor-Validator 循环；
5. JSONL 轨迹；
6. 20 个评测任务；
7. 基线与改进对照表。

暂不做：多 Agent 协作、在线 RL、复杂前端、分布式部署和自训练大模型。这些内容会稀释主线，且难以在实习面试中拿出可信数据。

## 8. 具体开发任务清单

### 第一个可交付版本

- [ ] 新建 `app/research/` 和 `tests/research/`；
- [ ] 定义 `Evidence`、`Claim`、`ResearchStep`、`TraceEvent`；
- [ ] 实现 JSONL TraceWriter；
- [ ] 用现有 `WebSearch` 返回标准化证据；
- [ ] 写一个只执行“计划—搜索—总结”的最小 Orchestrator；
- [ ] 为 5 个问题生成带 `[E#]` 引用的 Markdown 报告；
- [ ] 保存每次运行的耗时、Token 和失败原因。

### 第二个可交付版本

- [ ] 增加混合检索和重排；
- [ ] 增加 Validator 和局部重试；
- [ ] 增加证据去重、缓存、摘要记忆和上下文压缩；
- [ ] 建立 20 个任务的 `tasks.jsonl`；
- [ ] 输出自动评测结果；
- [ ] 完成一次基线/改进消融实验。

### 最终可投递版本

- [ ] 50—100 个任务评测集；
- [ ] 至少三项有效改进及统计结果（建议含程序性记忆和模型路由）；
- [ ] 失败案例分类和可复现命令；
- [ ] Docker 或清晰的 Windows 启动脚本；
- [ ] README、架构图、实验表、演示视频；
- [ ] 简历项目描述和面试问答稿。

## 9. 面试中必须讲清楚的问题

1. 为什么用 Planner-Executor-Validator，而不是让一个 Agent 自由循环？
2. 你的 RAG 与普通问答 RAG 有什么区别？
3. 如何判断一条引用真的支持一个结论？
4. 网页内容包含提示词注入时如何防御？
5. Agent 为什么会循环？你如何检测和恢复？
6. 如何在长任务中控制上下文长度和 Token 成本？
7. 哪个实验最能证明你的方法有效？是否有反例？
8. 失败时是检索问题、规划问题还是模型综合问题？如何定位？
9. 如果 API 限流或工具超时，系统如何降级？
10. 为什么不直接使用现成 Agent 框架，而要改 OpenManus？
11. 程序性记忆如何跨任务生效？如何防止 Agent 学到错误策略？
12. 用 Agent Skills 组织提示词与全量塞进 system prompt 相比，Token 和效果差异如何量化？

## 10. 风险和边界

- 不要把真实密钥写入 `config/config.toml`、日志或提交记录；
- 不要让研究 Agent 默认拥有任意文件写入、Shell 或外部提交权限；
- 网页、搜索结果和文档均视为不可信输入；
- 写入 `lessons.jsonl` 的教训只保留策略性结论，不保存网页原文，防止提示词注入跨任务传播；
- 对浏览器、Shell、MCP 工具设置白名单、超时和人工确认；
- 研究报告必须区分“来源事实”“模型推断”和“不确定结论”；
- 以 OpenManus 的 MIT 协议和原作者署名要求为准。

## 11. 建议的启动命令

在桌面项目目录中执行：

```powershell
cd C:\Users\13820\Desktop\OpenManus
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config\config.example.toml config\config.toml
python main.py
```

开始二次开发前，先不要修改通用 Agent。先创建 `app/research/`、跑通一个固定问题、保存基线轨迹，再逐步接入规划、检索、验证和评测。

## 12. 新对话的第一条指令

新建项目对话后，把本文件作为上下文，并发送：

> 请读取 `C:\Users\13820\Desktop\OpenManus\OPENMANUS_RESEARCH_AGENT_PLAN.md`，先检查当前仓库状态和依赖，不要直接大范围改代码。先完成 Phase 0 和“第一个可交付版本”的第一个未完成任务：设计 `app/research/models.py` 与 TraceWriter，并说明你准备修改哪些文件、如何测试。等待我确认后再实施。

