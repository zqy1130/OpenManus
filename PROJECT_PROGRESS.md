# 项目进度总结：Evidence-grounded Research Agent

> 本文件记录项目自开始以来的全部工作，按阶段组织。**每完成一个阶段，必须把新内容追加到本文件并随代码提交。**
>
> 基线仓库：FoundationAgents/OpenManus（MIT）｜个人 fork：github.com/zqy1130/OpenManus
> 分支：`codex/research-agent`｜开发计划：[OPENMANUS_RESEARCH_AGENT_PLAN.md](OPENMANUS_RESEARCH_AGENT_PLAN.md)

---

## 项目目标

基于 OpenManus 二次开发"可验证的深度研究 Agent"：问题分解 → 多跳检索 → 证据核验 → 状态维护 → 带 `[E#]` 引用的报告 → 完整轨迹与自动评测。最终形态是可复现的 Agent 算法实验平台（基线、改进方法、评测、失败分析）。

---

## 环境与基础设施

### 运行环境
- Python 3.12.3（conda base：`D:\miniconda3`）+ 项目内 `.venv`
- 模型：`qwen3.7-max`（阿里云百炼 MaaS compatible-mode 端点，OpenAI SDK 直连）
- 搜索引擎：DuckDuckGo 主引擎（国内网络实测唯一稳定），Baidu 回退；Bing 引擎在国内被抓 bot 检测返回无关结果，已弃用

### 上游依赖修复（requirements.txt，均有原因记录）
| 问题 | 处理 |
|---|---|
| `pillow~=11.1.0` 与 crawl4ai 冲突 | 改为 `pillow~=10.4` |
| `app/utils/logger.py` 需要 structlog 但未声明 | 新增 `structlog~=25.3.0` |
| daytona SDK 缺失（旧包名 daytona-sdk 已弃用） | 新增 `daytona~=0.207.1` |
| crawl4ai 依赖链（litellm 回溯到需 Rust 的 sdist） | 注释推迟到 Phase 2+，其导入是惰性的 |
| torch/transformers 的 c10.dll 加载失败（WinError 1114） | 卸载（本项目不需要），连带影响：**本地 embedding/交叉编码器不可用，改走 API embedding + LLM 重排** |

### 上游 bug 与规避
- `app/config.py` 的 `DaytonaSettings.daytona_api_key` 必填无默认，而 example 配置无 `[daytona]` 段 → 全新机器 import 即崩。规避：本地 `config.toml` 加占位 key；`app/research/trace.py` 不 import `app.config`（自己算项目根路径），保证研究模块独立可用

### Git / GitHub 工作流（从零搭建）
- 账号 ZQY1130（github.com/zqy1130），fork 上游仓库，remote `mine` 指向 fork
- 国内网络方案：**SSH over port 443**（`~/.ssh/config` 把 github.com 映射到 `ssh.github.com:443`），ed25519 密钥，无需代理
- `config/config.toml`、`outputs/`、`.venv` 均被 gitignore，密钥不进仓库
- 提交规范：`feat:/docs:/fix:` 前缀 + 说明 + Co-Authored-By

### gitignore 调整
- 新增 `outputs/`（研究产物不入库）
- `data/` 例外规则：`data/research/*.jsonl|*.json` 入库（评测集/gold/消融数据），语料和 embedding 缓存忽略（可脚本重建）。注意必须用 `data/*` 而非 `data/`（否则 git 不进入被忽略目录，例外不生效）

---

## Phase 0：基线复现 ✅

- `python main.py` 端到端跑通（Manus + qwen3.7-max + terminate 工具链）
- 5 个固定研究问题基线（存于 `data/research/tasks.jsonl`）：

| # | 类型 | 问题 |
|---|---|---|
| 1 | 事实查询 | 2024年图灵奖授予了谁？他们的获奖理由是什么？ |
| 2 | 对比分析 | 对比 OpenAI GPT-4o 与 Google Gemini 1.5 Pro 在多模态能力上的主要差异 |
| 3 | 时间约束 | 2024年7月至12月期间发布的重要开源大语言模型有哪些？各自有什么特点？ |
| 4 | 多跳推理 | 《Attention Is All You Need》的八位作者后来分别创办了哪些 AI 公司？ |
| 5 | 冲突来源 | 大语言模型是否真的具备"涌现能力"？支持方和反对方的论据分别是什么？ |

- 首次基线运行：19 子问题 / 85 证据 / 平均引用覆盖 91.8% / 总耗时 489s / 总 token 43k，0 错误
- 质量抽查：Q4 正确识别 8 作者与 4 家公司、证据缺失处明确写"证据不足"、发现并标记混淆作者名单的错误来源；Q5 准确呈现 Wei 2022 vs 斯坦福 Mirage 论文双方论据

---

## Phase 1：结构化任务与轨迹 ✅（第一个可交付版本）

### 新增模块（`app/research/`）
- **models.py**：`TaskStatus/StepStatus/ClaimStatus/EventType` 枚举；`Budget`（max_steps/max_retries_per_step/timeout_s/max_tokens）；`ResearchTask`；`ResearchStep`（depends_on、success_criteria、retry_count）；`Evidence`（sha256 content_hash 去重、doc_id/chunk_id 预留本地检索、`from_search_result` 鸭子类型转换不依赖工具层）；`Claim`；`LLMUsage`（token/延迟/成本）；`TraceEvent`（**span_id/parent_span_id，OpenTelemetry 风格层级 trace**）
- **trace.py**：`TraceWriter` — 每任务目录 `outputs/research/<task_id>/`，JSONL 逐事件落盘（每条 flush，可实时 tail）；span 栈管理（task→阶段→step）；敏感键递归脱敏（api_key/authorization/password…）；`summary.json` manifest（事件数/token/成本/耗时/错误）；`load_trace` + `build_span_tree` 纯轨迹重放
- **retriever.py**（web）：包装现有 `WebSearch` → 标准化 `Evidence`，URL+哈希去重，失败返回空列表 + 记 error 事件（不炸流程）
- **llm.py**：`call_llm`（非流式、逐次返回 usage）、`extract_json`（容忍 markdown 代码块/夹带散文）
- **orchestrator.py**：最小闭环 计划（LLM 生成 3-5 子问题 JSON）→ 逐步检索 → 全局去重编号 `[E1..En]` → 综合报告；计划解析失败回退单子问题；产物 report.md + evidence.json
- **cli.py**：`python -m app.research.cli --question "..."`，实时进度打印（计划/检索/去重/生成/汇总）

### 测试
- 32 个单元测试（当前 77 个）：模型、trace 重放/脱敏/manifest、检索、编排管线（LLM/检索全 mock，离线可跑）

---

## Phase 2：检索与证据层 ✅

### 新增模块
- **scripts/build_corpus.py**：语料构建脚本（可复现）。20 篇文档（17 篇英文 Wikipedia + 3 篇中文 Wikipedia），manifest.json 记录 doc_id/标题/源 URL。注意：Wikipedia 对裸 curl UA 拒绝，必须浏览器 UA
- **documents.py**：HTML（mw-content-text 提取）/MD 加载；段落边界切块（500 字符目标 + 50 重叠，长句硬切）
- **bm25.py**：**零依赖自实现 BM25**（k1=1.5, b=0.75）；中文用字符二元组分词（不引 jieba，面试可讲），英文单词分词
- **embeddings.py**：API embedding（qwen3.7-text-embedding，1024 维，batch 16）+ numpy 余弦索引 + 磁盘缓存（data/research/embeddings/，重建免费）
- **reranker.py**：`LLMReranker` 逐条打分 0-10 重排；**候选文本在 prompt 中显式标注 UNTRUSTED DATA**（防提示词注入）；交叉编码器因 torch 故障留待后续
- **evaluator.py**：Recall@k / MRR / 文档去重率 / 延迟，一条命令跑消融对比表

### 检索消融（20 查询 gold 标注 × 2128 chunks，top_k=5）

| 配置 | Recall@5 | MRR | uniq/top_k |
|---|---|---|---|
| BM25 | 0.800 | 0.975 | 0.43 |
| 向量 | 0.867 | 0.975 | 0.39 |
| **混合** | **0.883** | **1.000** | **0.48** |
| 混合+重排 | 0.883 | 0.975 | 0.48（延迟 13s/查询） |

### 检索层发现（诚实结论）
1. 小语料上 MRR 近满分，BM25 基线已强
2. **chunk 挤占效应**：同文档多 chunk 挤占 top-k 名额，doc 级 recall 受损（0.792@5→0.858@10→0.900@20）
3. LLM 重排在干净小语料上无增益（13s 延迟），价值场景是 web 级噪声候选——留待接入 web 混合检索后验证

---

## Phase 3：Planner-Executor-Validator 闭环 ✅（前半+消融）

### 新增/修改
- **结果多样性修复**（retriever.py）：`max_chunks_per_doc=2` → **混合检索 Recall@5 0.792→0.883**，uniq_doc_rate 0.31→0.48，MRR 保持 1.0（第一个可量化算法改进）
- **verifier.py**：`Validator` — LLM 判断步骤完成度 + 证据充分性 0-10 + 提出补检索查询；解析失败 **fail-open**（不卡任务）；每次判断输出**节点级指标**写入 trace（供 Phase 6 汇总）
- **orchestrator.py**：每步 校验-重试 循环（重试次数受 `Budget.max_retries_per_step` 约束），步骤软失败不中断任务；步骤级指标（queries_tried/retries_used/step_complete/evidence_sufficiency）入 trace
- **错误分类**（retriever.py）：`classify_error` → param_error / network_error / rate_limit / empty_results，写入 error 事件
- **cli.py**：`--no-validator` 消融开关

### Validator 消融实验（5 问题 × 2 组，同模型同配置）

| 指标（每题平均） | 基线 | +Validator | 变化 |
|---|---|---|---|
| 证据数 | 14.0 | 30.2 | +116% |
| 补检索次数 | 0 | 4.0 | — |
| 耗时 | 109s | 203s | +86% |
| 输入 Token | 2,215 | 9,861 | +345% |
| 输出 Token | 6,244 | 11,312 | +81% |
| 错误事件 | 0.6 | 0 | 失败被重试吸收 |

**决定性案例（Q4）**：基线漏掉 Gomez→Cohere（公司名被截断标"证据不足"）；Validator 补检索后 8 位作者公司全部找齐。
**度量发现**：现有"引用覆盖率=被引/收集"在收集变全后失真（1.0→0.74），Phase 6 必须改为"关键结论引用覆盖率"。数据存 `data/research/ablation_validator.json`。

### 测试
- 77 个单元测试全过（新增：validator 解析/重试循环/错误分类/局部检索）

---

## 已提交记录（本分支）

| commit | 内容 |
|---|---|
| `4558db6` | feat: 研究 agent 核心（models/trace/retriever/orchestrator）+ 32 测试 |
| `51c94c1` | docs: 开发计划 + 接手说明 |
| `22b634f` | feat: CLI 进度打印 |
| `4eebaea` | feat: 本地语料检索（BM25/向量混合/重排/评测） |
| `6a8e0a4` | feat: Validator + 重试循环 + 结果多样性 |
| `0abe150` | docs: 首次消融数据（baseline vs validator） |

---

## Phase 5：报告生成与事实核查（前半：Claim-Evidence 检查）✅

### 新增模块
- **claim_check.py**：`ReportAuditor` 两阶段审计
  - 断言提取：LLM 从报告提取逐条事实断言 + 其引用的 `[E#]` 编号（接受 `{"claims": [...]}` 字典形式）
  - 引用检查：LLM 四分类判断（supported / contradicted / irrelevant / unsupported），批量 8 条/次
  - **自修复重试**：发现 LLM 偶尔产出非法 JSON（字符串内引号未转义）→ 解析失败时把错误反馈给模型重试一次（两阶段都有），修复后 19 份报告全部审计成功
  - 未判定声明（unjudged）不计入正确率，保证指标诚实
  - 产物：每任务 `audit.json`（断言+判定+指标），聚合 `data/research/claim_audit_results.json`
  - CLI：`python -m app.research.claim_check --all | --task-id <id>`

### 新指标（替换失真的旧覆盖率）
- `citation_correctness` = supported / (supported+contradicted+irrelevant)
- `unsupported_claim_rate` = 无引用断言 / 全部断言

### 回评结果：基线 vs Validator（各 5 题，19 份报告全部审计）

| 指标 | 基线 | +Validator |
|---|---|---|
| 断言总数 | 83 | 109 |
| **引用正确率** | 0.889 | **0.966**（+7.7pp，达成计划 85% 目标） |
| 无引用断言率 | 1.0% | 0.7% |

**验证了 Phase 3 的判断**：旧"收集覆盖率"指标（validator 组 0.74 < 基线 1.0）确实是度量失真——在正确指标下 validator 组引用质量反而**更高**。收集更多证据 → 模型引用更精准，而不是"引不过来"。

### 测试
- 83 个单元测试全过（新增 claim_check 测试 5 个：指标计算、提取失败、unjudged 排除、非法引用、字典形式解析）

---

## Phase 6 前半：评测集扩展 ✅（任务集 5→23）

- `data/research/tasks.jsonl` 扩到 **23 题**（fact_lookup 5 / comparison 5 / time_constraint 4 / multi_hop 5 / conflicting_sources 4）
- 每题含完整 gold 标注：`gold_answer`（标准答案）、`key_facts`（关键事实清单）、`acceptable_sources`（可接受来源）、`stop_conditions`（停止条件）——按计划 Phase 6 的标注规范
- 新题覆盖：GPT-5/Claude 4 发布时间、Mistral AI/波士顿动力公司事实、DeepSeek-R1 vs o1、RAG vs 长上下文、LoRA vs 全参微调、OpenAI vs Anthropic 安全策略、2025 H1 模型清单（含 GPT-5 时间窗口排除测试）、OpenAI 2023-2024 人事风波、2024 融资事件、Sutton&Barto 教材、Shazeer→Transformer、Sakana AI、DeepMind→诺贝尔化学奖、Scaling Law 之争、开源管制之争、AI 意识之争
- 时效性事实（GPT-5 发布时间、Claude 4 版本、融资额）均经网络搜索验证后写入
- 题目设计要点：t14 特意埋了"GPT-5 是 8 月发布"的**时间窗口排除测试**，可检验 Agent 的时间约束能力

---

## Phase 4：记忆系统 ✅（程序性记忆 + 证据摘要）

### 新增模块
- **memory.py**：`Lesson`（策略级教训：strategy/outcome/task_id/category）+ `LessonStore`（JSONL 追加写入 `data/research/lessons.jsonl` + BM25 检索）
- **程序性记忆闭环**（orchestrator.py）：
  - 每次任务结束后 LLM 提取 2-4 条策略级教训写入 lessons.jsonl（**只存策略结论，不存网页原文**——防提示词注入跨任务传播，符合计划风险条款）
  - 下次规划时 BM25 检索 top-3 相关教训注入 PLANNER 系统提示词
- **证据摘要记忆**：证据超阈值（默认 15 条）时 LLM 逐条压缩为一句事实要点，保留 `[E#]` 编号映射，压缩后喂给综合器
- CLI 开关：`--no-memory` / `--no-summarize`（消融用）
- 测试 91/91

### 基础设施修复
- **IPv6 路由故障**：机器 IPv6 不通而 httpx 优先 IPv6 → APIConnectionError。修复：`app/llm.py` 给 OpenAI 客户端绑 `local_address="0.0.0.0"` 强制 IPv4（框架级修改，已记录原因，commit 89b2286）

### 记忆消融（5 新题 × 2 组，数据存 ablation_memory.json）

| 指标 | A 无记忆 | B 记忆+摘要 | 变化 |
|---|---|---|---|
| 引用正确率（claim 审计） | 0.969 | **0.982** | +1.3pp（小样本不显著） |
| 断言总数 | 89 | 93 | 部分题目 B 组断言 3×（Q3: 5→15） |
| 证据总数 | 132 | **176** | +33%（教训引导补检索更多） |
| 总耗时 | 817s | 1714s | **+110%** |
| 输出 token | 60k | 138k | +129% |

**诚实结论**：短任务上记忆组质量持平略升（天花板效应：两组正确率都已 ~0.97），但成本翻倍——**短任务不划算**；程序性记忆的价值场景是长任务/难任务/重复任务，待 Phase 6 大任务集验证。已提取 18 条教训质量高（如"主动验证问题前提的准确性"正是 t17 年份陷阱的解法），策略级、无网页内容、防注入有效。

---

## Phase 6：系统评测 ✅（评测体系 + 首轮全量 + 失败分析）

### 新增模块
- **success_judge.py**：gold 对比器——LLM 对每个关键事实判 covered/correct，产出 `task_success` 与 `gold_score`（0-1）
- **模型路由**（llm.py + 全模块接线）：`ROUTES` 把机械阶段（校验/摘要/教训提取/判定）路由到 qwen-plus、质量阶段（规划/综合）保持 qwen3.7-max；`call_llm` 支持 model 覆盖
- **run_eval.py**：一键评测——跑任务集 → claim 审计 → gold 判定 → 聚合（成功率/正确率/延迟 p50/p95/失败分类）→ JSON + 表格
- **覆盖修订循环**（orchestrator）：综合后用计划目标（非 gold，保持诚实）检查报告覆盖，遗漏时修订一次
- **修复**：GBK 控制台 UnicodeEncodeError（t04 崩溃，入口统一 UTF-8）；恢复 5 个被 NULL 字节损坏的文件（git + 重写）
- 测试 100/100

### 首轮全量评测（23 题 × 3 组）

| 指标 | v1 初始 | v2 +覆盖修订 |
|---|---|---|
| 完成率 | 22/23 | **23/23** |
| **task_success_rate** | 0.182 | 0.130 |
| 平均 gold_score | 0.600 | 0.584 |
| 引用正确率 | 0.909 | **0.944** |
| 无引用断言率 | 0.020 | 0.016 |
| LLM 延迟 p50/p95 | 13.9s/54.0s | 13.9s/52.6s |

逐题 v2 vs v1：提升 5 / 持平 12 / 下降 5（t17 0.33→1.00、t04 崩溃→0.86；t07 0.75→0、t05 1.0→0.33）

### 失败分析（按类别均分）

fact_lookup 0.79 > conflicting 0.62 > multi_hop 0.58 > time_constraint 0.51 > **comparison 0.49（最弱）**
主要失败模式：① 综合器漏枚举关键维度（对比题只写 1-2 个维度）② 时间窗口题漏项/错写时间 ③ **实时搜索方差**——同一题两次运行搜到的证据不同，是端到端评测最大的噪声源

### 诚实结论（面试素材）
1. **引用正确率 0.94 达成 85% 目标**，无引用断言率 1.6%——引用体系扎实
2. **task_success 低（~0.15）的主因是"关键事实全覆盖"太严格 + 搜索方差**，不是幻觉——judge 反馈显示大部分失败是"遗漏"而非"写错"
3. 覆盖修订循环在证据充分时有效（t17/t19），证据缺失时无力（t07 搜索没找到正确页面）
4. **模型路由被账号权限卡住**：qwen-plus/turbo/max 均 403（百炼控制台需手动开通），待开通后补跑 routed 消融
5. 下一步方法论改进：**评测期检索结果缓存**（固定每次评测的证据输入，消除搜索方差，才能做严格消融）

### 模型路由消融（t01-t08 同题对比，qwen-plus 做机械阶段）

| 指标 | 全强模型（v2） | 路由（+qwen-plus） |
|---|---|---|
| 平均 gold_score（强模型判定） | **0.639** | 0.481（-15.8pp） |
| LLM 延迟 p50 | 13.9s | **4.2s**（-70%） |
| 无引用断言率（强模型审计） | ~0.02 | 部分报告 0.36 |

**结论**：qwen-plus 承担校验/摘要/覆盖检查后质量显著下降（弱模型的摘要丢失引用细节、校验误判），延迟收益不抵质量损失。**真省钱路由应只把弱模型用在低风险环节**（教训提取/重排），质量关键环节（校验/摘要/综合）保持强模型——留作后续调优方向，实验数据存 eval_results_routed8.json。

### 消融数据总览（全部存 data/research/）
- ablation_validator.json：Validator +7.7pp 引用正确率、~2× 成本
- retrieval_eval_results.json：混合检索 Recall@5 0.883（多样性修复后）
- ablation_memory.json：记忆组正确率 0.982 vs 0.969、成本 +110%
- eval_results_full23.json / full23v2.json / routed8.json：全量评测与失败明细
- 另发现：MaaS 账号欠费时 API 表现为 PermissionDeniedError（403），易误判为权限问题

---

## 评测确定性改造 ✅（检索/计划缓存）

### 新增
- **retrieval_cache.py**：`RetrievalCache`（按查询词缓存证据，jsonl）+ `PlanCache`（按问题缓存规划结果）
- 三模式：`off`（线上行为）/ `on`（命中即用、未命中搜后写入）/ `read`（只用缓存、绝不联网，严格复现）
- **历史播种**：从全部历史运行的 evidence.json / trace 一键建缓存（当前已播种 1909 条证据 / 461 查询 / 27 个计划）
- `run_eval --cache-mode read --seed-cache` 一键确定性评测；测试 108/108

### 确定性验证（同题两跑）
- **t01 两跑 evidence.json 哈希完全一致** ✓——计划+检索双缓存生效，输入侧方差消除
- t03 不一致 ✗——根因：validator 的"是否补检索"决策是 LLM 采样，每次不同；其新查询在 read 模式缓存未命中返回空，导致证据集差异
- **诚实边界**：检索+计划缓存消除了主导噪声（搜索方差）；残余方差来自 LLM 判定阶段的采样差异。完全确定性需要全量 LLM 响应回放（replay 模式），留作后续选项——但 replay 改变实验性质（重放≠评测），当前方案是评测的最佳实践折中

---

## 下一步（待确认方向）

- **B（推荐）**：Phase 7 工程化收尾（README/架构图/简历材料/面试问答稿）
- 可选：用缓存模式重跑关键消融（严格复现版数据）
- 可选：完全回放模式（缓存全部 LLM 响应）

---

## 维护规则（永久生效）

每完成一个阶段（或重要里程碑）：
1. 在本文件对应位置**追加**新内容：新增/修改的模块与文件、实测指标表、发现的问题与解决方案、关键设计决策、新 commit
2. 与代码一起 commit 到 `codex/research-agent`
3. 新对话开始时先读本文件了解现状
