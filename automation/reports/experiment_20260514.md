# Spark 候选策略实验报告

- 生成时间: 2026-05-14T07:07:27Z
- 候选数: 3

## 结果摘要

### bcfc45a2dc383ac0

- 状态: rejected
- 结论: sharpe_improvement<0.0
- 交易数: 20
- Sharpe: 7.070258 (baseline 7.296074)
- 胜率: 0.95
- 最大回撤: -0.000306

### ce1cb7b87b2541f7

- 状态: needs_more_data
- 结论: trades<20, sharpe<0.8, sharpe_improvement<0.0
- 交易数: 1
- Sharpe: 0.0 (baseline 2.716611)
- 胜率: 1.0
- 最大回撤: 0.0

### db22756fa503a35f

- 状态: needs_more_data
- 结论: trades<20
- 交易数: 4
- Sharpe: 3.014882 (baseline 2.569524)
- 胜率: 1.0
- 最大回撤: 0.0

## 安全边界

通过候选只标记为 ready_for_review，并发送提醒；不会自动进入生产信号。
