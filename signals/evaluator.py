"""
信号评估与过滤
- 对生成的信号进行质量评估
- 板块风控（同板块仓位限制）
- 信号冲突检测
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

import yaml

from signals.models import TradingSignal, Direction

logger = logging.getLogger(__name__)


class SignalEvaluator:
    """信号评估器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = self.config.get("signal_thresholds", {})
        self.commodities = self.config.get("monitored_commodities", {})
        
        # 构建品种到板块的映射
        self.commodity_to_group: Dict[str, str] = {}
        for group, commodities in self.commodities.items():
            for comm in commodities:
                self.commodity_to_group[comm["code"]] = group
    
    def evaluate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        评估单个信号的质量
        """
        issues = []
        score = signal.confidence
        
        # 检查置信度阈值
        min_conf = self.thresholds.get("min_confidence", 0.7)
        if signal.confidence < min_conf:
            issues.append(f"置信度 {signal.confidence:.2f} 低于阈值 {min_conf}")
            score *= 0.5
        
        # 检查政策驱动
        if signal.policy_driver:
            if signal.policy_driver.confidence < self.thresholds.get("min_policy_relevance", 0.6):
                issues.append("政策相关度不足")
                score *= 0.8
        else:
            issues.append("缺少政策驱动分析")
            score *= 0.7
        
        # 检查基本面驱动
        if signal.fundamental_driver:
            if signal.fundamental_driver.score < self.thresholds.get("min_fundamental_score", 0.6):
                issues.append("基本面得分不足")
                score *= 0.8
        else:
            issues.append("缺少基本面驱动分析")
            score *= 0.7
        
        # 检查持仓周期
        max_hold = self.thresholds.get("max_holding_days", 30)
        if signal.holding_period_days > max_hold:
            issues.append(f"建议持仓 {signal.holding_period_days} 天超过阈值 {max_hold}")
        
        # 检查逻辑完整性
        if not signal.entry_conditions:
            issues.append("缺少入场条件")
        if not signal.stop_loss_logic:
            issues.append("缺少止损逻辑")
        
        # 检查高风险信号的特殊处理
        if signal.risk_level.value == "HIGH":
            if signal.position_sizing == "重仓":
                issues.append("高风险信号建议重仓，存在矛盾")
                score *= 0.8
        
        return {
            "signal_id": signal.id,
            "commodity": signal.commodity_code,
            "original_confidence": signal.confidence,
            "evaluated_score": score,
            "passed": score >= min_conf and len([i for i in issues if "缺少" not in i]) <= 2,
            "issues": issues,
            "recommendation": "PASS" if score >= min_conf else "FILTER",
        }
    
    def check_sector_exposure(
        self,
        signals: List[TradingSignal],
        max_per_sector: int = 2,
    ) -> List[TradingSignal]:
        """
        检查板块集中度，限制同板块信号数量
        """
        sector_count: Dict[str, int] = defaultdict(int)
        sector_direction: Dict[str, Direction] = {}
        filtered = []
        
        for signal in signals:
            group = self.commodity_to_group.get(signal.commodity_code, "other")
            
            # 检查同板块反向信号
            if group in sector_direction:
                if sector_direction[group] != signal.direction:
                    logger.warning(
                        f"板块 {group} 存在反向信号: "
                        f"{signal.commodity_code} {signal.direction.value} vs "
                        f"现有 {sector_direction[group].value}"
                    )
                    # 保留置信度更高的
                    continue
            
            if sector_count[group] >= max_per_sector:
                logger.info(f"板块 {group} 信号数量已达上限 {max_per_sector}，跳过 {signal.commodity_code}")
                continue
            
            sector_count[group] += 1
            sector_direction[group] = signal.direction
            filtered.append(signal)
        
        return filtered
    
    def detect_conflicts(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """
        检测信号间的冲突
        """
        conflicts = []
        
        # 检查产业链上下游反向信号
        for i, s1 in enumerate(signals):
            for s2 in signals[i+1:]:
                # 同品种双向（不应该出现）
                if s1.commodity_code == s2.commodity_code and s1.direction != s2.direction:
                    conflicts.append({
                        "type": "SAME_COMMODITY_REVERSE",
                        "signals": [s1.id, s2.id],
                        "commodity": s1.commodity_code,
                        "description": f"同品种 {s1.commodity_code} 存在双向信号",
                    })
                
                # 同板块反向
                g1 = self.commodity_to_group.get(s1.commodity_code)
                g2 = self.commodity_to_group.get(s2.commodity_code)
                if g1 and g1 == g2 and s1.direction != s2.direction:
                    conflicts.append({
                        "type": "SECTOR_REVERSE",
                        "signals": [s1.id, s2.id],
                        "sector": g1,
                        "description": f"板块 {g1} 内 {s1.commodity_code}({s1.direction.value}) 与 {s2.commodity_code}({s2.direction.value}) 反向",
                    })
        
        return conflicts
    
    def filter_signals(
        self,
        signals: List[TradingSignal],
        apply_sector_limit: bool = True,
    ) -> List[TradingSignal]:
        """
        综合过滤信号
        """
        # 第一步：质量评估
        evaluated = [self.evaluate_signal(s) for s in signals]
        passed = [s for s, e in zip(signals, evaluated) if e["passed"]]
        
        for ev in evaluated:
            if not ev["passed"]:
                logger.info(f"信号过滤: {ev['signal_id']} - {ev['issues']}")
        
        # 第二步：板块风控
        if apply_sector_limit:
            passed = self.check_sector_exposure(passed)
        
        # 第三步：冲突检测（仅记录，不自动过滤）
        conflicts = self.detect_conflicts(passed)
        if conflicts:
            logger.warning(f"检测到 {len(conflicts)} 个信号冲突")
            for c in conflicts:
                logger.warning(f"  - {c['type']}: {c['description']}")
        
        return passed
