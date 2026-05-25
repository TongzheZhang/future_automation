# Spark 自进化期货投研系统 ⚡

> 基于 DGX Spark + OpenClaw + AKShare 的 AI 驱动期货日内策略投研系统

## 架构

```
数据采集(AKShare) → 策略回测(日内分钟级) → 信号生成 → 飞书推送
                                          ↓
                                    进化引擎(自动优化参数)

公开信息/RSS/网页 → 研究摘要(Ollama) → 策略想法候选池
```

## 当前策略

### 开盘缺口回补 (Opening Gap Fill)
- **逻辑**: 当日收盘价相对前日有较大跳空 → 次日开盘反向交易 → 30-90分钟回补出场
- **最优品种**: PTA (Sharpe 8.13), 原油 (Sharpe 6.29), 生猪 (Sharpe 6.74)
- **信号频率**: 每日 3-5 个

## 自动化调度

| 任务 | 时间 | 说明 |
|------|------|------|
| 每日信号生成 | 周一至五 15:30 | 分析收盘数据→生成次日信号→飞书推送 |
| 策略进化检查 | 每周日 | 分析策略表现→自动优化参数→更新策略库 |
| 策略研究候选 | 每日或每周 | 搜索公开信息→生成待测策略想法→保存研究报告 |

## 使用

```bash
# 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 运行一次
python daily_run.py

# 运行完整回测流水线
python main.py

# 运行一次网络搜索 + 策略研究
python -m research.runner --once

# 离线演示/测试，不访问网络
python -m research.runner --once --dry-run

# 自治编排入口：生产信号提醒
python -m automation.orchestrator daily

# 自治编排入口：研究、候选实验、周度报告、健康检查
python -m automation.orchestrator research
python -m automation.orchestrator experiment
python -m automation.orchestrator weekly
python -m automation.orchestrator health
```

## 策略研究模块

研究模块是旁路系统，只生成候选想法，不自动修改生产策略、不自动上线、不生成交易指令。

默认配置在 `config/research.yaml`：

- 无 Key 公开搜索/网页源聚合，失败源会记录并跳过
- 本地 Ollama：`http://127.0.0.1:11434`，默认模型 `qwen2.5:7b`
- Ollama 不可用时自动降级为关键词摘要和模板化策略假设
- 输出到 `research/cache/articles.jsonl`、`research/ideas/strategy_ideas.jsonl`、`research/reports/`

建议用 OpenClaw/Cron 每日或每周调度：

```bash
python -m research.runner --once
```

## 全自动自进化编排

自治编排配置在 `config/automation.yaml`。它把生产信号、研究、候选实验、周报和健康检查串起来，但安全边界固定为：

- 不自动下单
- 不自动启用新策略
- 不生成任意 Python 策略代码
- 通过回测的候选只标记为 `ready_for_review` 并发送复核提醒

OpenClaw/Cron 可按以下命令定时执行：

```cron
15 9 * * 1-5  cd /home/zhangtongzhe/Codework/Spark_automation && venv/bin/python -m automation.orchestrator daily
30 16 * * 1-5 cd /home/zhangtongzhe/Codework/Spark_automation && venv/bin/python -m automation.orchestrator research
0 10 * * 6    cd /home/zhangtongzhe/Codework/Spark_automation && venv/bin/python -m automation.orchestrator weekly
0 8 * * 1-5   cd /home/zhangtongzhe/Codework/Spark_automation && venv/bin/python -m automation.orchestrator health
```

离线验证：

```bash
python -m automation.orchestrator weekly --dry-run --no-send
```

## 项目结构

```
Spark_automation/
├── config/             # 品种列表、系统参数
├── data/               # 数据采集+存储
├── strategies/         # 策略实现
├── backtest/           # 回测引擎(日线+日内分钟级)
├── signals/            # 信号生成+飞书推送
├── research/           # 网络搜索+策略研究候选
├── automation/         # 自治编排、候选实验、健康检查
├── evolution/          # 策略进化引擎
├── logs/               # 运行日志
├── reports/            # 回测报告
├── daily_run.py        # 每日自动化入口
└── main.py             # 完整流水线入口
```

---

*Spark 自进化投研系统 v0.1 · DGX Spark + OpenClaw*
