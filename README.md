# 政策-基本面驱动期货投研系统

> 基于**国务院及各部委官方政策新闻**与**大宗商品基本面供需/产业链分析**驱动的期货交易机会挖掘系统。

## 核心投研流派

**确定性的来源**：
1. **政策驱动** — 发改委、商务部、工信部、农业农村部、生态环境部、能源局、央行等发布的产业政策、调控措施、进出口政策
2. **基本面驱动** — 库存周期、供需缺口、季节性规律、产业链利润传导、基差结构
3. **事件驱动** — 极端天气、地缘冲突、突发监管

**研究工具**：
- **Brave Search** — 实时搜索政策新闻、基本面数据、行业动态
- **Alpha 派** — 深度投研问答、行业/公司一页纸、数据检索、图表搜索
- **大语言模型（OpenRouter）** — 政策文本语义分析、产业链推理、多源信息综合评估

## 项目结构

```
├── skills/alphapai-research/     ← Alpha 派 v1.1 Skill（已安装）
├── config/
│   ├── settings.yaml             ← API Keys、监控品种、阈值参数
│   └── commodities.yaml          ← 品种与部委/产业链映射
├── docs/                         ← 核心：持续迭代的知识库
│   ├── research_framework.md     ← 投研框架总纲
│   ├── policy_sources.md         ← 部委政策源清单
│   ├── commodity_chains/         ← 产业链深度分析
│   │   ├── black_metal.md
│   │   ├── agriculture_soy.md
│   │   └── copper.md
│   └── trade_cases/              ← 交易案例复盘
├── research/                     ← 研究引擎
│   ├── llm_integration.py        ← OpenRouter 统一调用
│   ├── policy_analyzer.py        ← 政策分析引擎
│   ├── fundamental_analyzer.py   ← 基本面分析引擎
│   ├── chain_mapper.py           ← 产业链映射与传导
│   └── signal_generator.py       ← 交易信号生成
├── signals/
│   ├── models.py                 ← Pydantic 信号模型
│   └── evaluator.py              ← 信号过滤与板块风控
├── data/collectors/
│   ├── brave_search.py           ← Brave Search 集成
│   ├── alpha_pai.py              ← Alpha 派程序化封装
│   ├── policy_news.py            ← 政策新闻采集
│   └── fundamental.py            ← 基本面数据采集
├── scripts/
│   ├── daily_research.py         ← 每日自动投研流水线
│   ├── demo_research.py          ← 演示脚本（无需 LLM）
│   └── update_framework.py       ← 框架迭代更新
├── backtest/
│   └── fundamental_bt.py         ← 事件驱动回测框架
├── reports/daily/                ← 日报输出
└── tests/                        ← 测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

编辑 `config/settings.yaml`：

```yaml
llm:
  api_key: "sk-or-v1-xxxxxxxx"   # OpenRouter API Key

search:
  brave_api_key: "BSA..."        # Brave Search API Key
```

配置 Alpha 派：
```bash
python skills/alphapai-research/scripts/alphapai_client.py config --set-key <YOUR_ALPHAPAI_KEY>
```

### 3. 运行演示（无需 LLM/Alpha 派）

```bash
python scripts/demo_research.py
```

演示脚本使用 Brave Search 获取政策新闻和基本面信息，生成基础日报。

### 4. 运行完整每日投研

```bash
# 全部品种
python scripts/daily_research.py

# 聚焦特定品种
python scripts/daily_research.py --focus RB CU M
```

### 5. 更新投研框架

```bash
# 记录新的研究发现
python scripts/update_framework.py --insight "发现螺纹钢利润与政策的非线性关系" --category "black_metal"

# 记录交易复盘
python scripts/update_framework.py --trade --commodity RB --direction LONG --entry 3500 --exit 3680 --pnl 180 --logic "..." --review "..."

# AI 辅助更新框架
python scripts/update_framework.py --ai-update --topic "能耗双控转向碳排放双控" --evidence "..."
```

## 投研框架核心

### 共振模型

```
高确定性信号 = 政策方向 × 基本面验证 × 产业链逻辑 × 市场预期差
```

### 出手原则
- **只在高确定性时刻出手**：政策方向明确 + 基本面共振 + 产业链验证
- **不预测，只应对**：跟踪政策与数据的边际变化
- **小亏大赚**：单笔亏损有限，盈利让市场决定

### 风险控制
- 单笔风险 ≤ 2% 账户资金
- 同板块仓位 ≤ 10%
- 政策转向信号出现时，立即平仓

## 测试

```bash
pytest tests/ -v
```

## 待接入数据源

当前系统框架已完整，以下数据源需要用户自行接入以获取更精准的数据：

| 数据类型 | 推荐来源 |
|----------|----------|
| 政策原文 | 各部委官网爬虫 / Wind 政策数据库 |
| 黑色系数据 | Mysteel API |
| 能源化工数据 | 隆众资讯 API |
| 农产品数据 | 我的农产品网 / USDA |
| 有色数据 | SMM API |
| 行情数据 | CTP / 交易所 API |

---

*本项目强调高确定性的交易机会，不追求高频，只在认知差最大的时刻出手。*
