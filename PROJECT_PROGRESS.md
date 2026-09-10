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

## 下一步（已确认方向）

**Phase 5 前半：Claim-Evidence 自动检查**（当前中断于此，用户已确认开始）
1. 断言提取器：从报告 Key Findings 解析逐条 Claim（LLM 结构化提取，离线可跑）
2. Claim-Evidence 检查器：引用存在性 + 引用是否**真正支持**断言（支持/矛盾/无关三分类）
3. 回评 11+ 份存量报告，产出 `citation_correctness`、`unsupported_claim_rate` 指标，替换失真的旧覆盖率指标
4. 之后：Phase 4 记忆系统 → Phase 6 系统评测（50-100 任务 + 3 项消融）

---

## 维护规则（永久生效）

每完成一个阶段（或重要里程碑）：
1. 在本文件对应位置**追加**新内容：新增/修改的模块与文件、实测指标表、发现的问题与解决方案、关键设计决策、新 commit
2. 与代码一起 commit 到 `codex/research-agent`
3. 新对话开始时先读本文件了解现状
