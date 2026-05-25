# Agent 工作指南

## 项目定位

政策-基本面驱动的期货投研系统。Agent 的核心任务是：
1. **持续跟踪**各部委政策与大宗商品基本面变化
2. **深度分析**政策对产业链的传导路径与影响程度
3. **生成高确定性**的交易信号（方向、品种、逻辑、置信度）
4. **迭代更新**投研框架与知识库

## 代码规范

- Python 3.10+
- 类型注解强制使用
- 异步优先（aiohttp, asyncio）
- 配置外置（YAML），代码不硬编码 API Key
- 日志使用 `logging` 模块，统一输出到 `logs/`

## Alpha 派使用规范

- 调用 `skills/alphapai-research/scripts/alphapai_client.py` CLI 或导入 `AlphaPaiClient` 类
- **禁止对 Alpha 派输出做二次加工**，必须完整呈现原始内容（除非用户明确要求总结）
- 优先使用 `--mode Think` 进行深度问答，使用 `--json` 进行程序化数据获取
- 使用 `recall` 接口获取原始底层数据自行加工

## LLM 使用规范

- 通过 `research/llm_integration.py` 统一调用 OpenRouter
- 按任务选择模型：
  - 简单分类/提取 → 轻量模型（节省成本）
  - 深度推理/产业链分析 → 强模型（Claude 3.5 Sonnet / GPT-4o / DeepSeek-R1）
- 所有 LLM 调用必须保留原始 prompt 和 response，便于复盘

## 文件操作规范

- 读取文件使用 `ReadFile`
- 修改文件使用 `StrReplaceFile`
- 新建文件使用 `WriteFile`
- 批量查询使用 `Grep` / `Glob`

## 迭代机制

每次完成一项深度研究或交易复盘后：
1. 更新 `docs/research_framework.md` 中的认知框架
2. 如新增有效数据源，更新 `docs/policy_sources.md`
3. 如产生新的产业链逻辑，在 `docs/commodity_chains/` 下新建/更新文件
4. 如完成一次交易，在 `docs/trade_cases/` 下记录复盘

## 测试规范

- 每个核心模块必须有对应测试文件
- 测试使用 `pytest`
- 运行：`pytest tests/`
