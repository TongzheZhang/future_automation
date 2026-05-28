"""
测试 Alpha派 投研顾问模块 (data/collectors/alpha_pai_research.py)
"""

import pytest
from unittest.mock import MagicMock, patch
from data.collectors.alpha_pai_research import (
    AlphaPaiResearchAdvisor,
    SignalValidationResult,
)


class TestSignalValidationResult:
    """测试信号验证结果数据模型"""

    def test_creation(self):
        result = SignalValidationResult(
            verdict="赞同",
            assessment="逻辑通顺",
            risk_reminders="注意止损",
            confidence_adjustment=0,
        )
        assert result.verdict == "赞同"
        assert result.confidence_adjustment == 0

    def test_verdict_values(self):
        for v in ["赞同", "质疑", "补充"]:
            result = SignalValidationResult(
                verdict=v, assessment="", risk_reminders="", confidence_adjustment=0
            )
            assert result.verdict == v


class TestAlphaPaiResearchAdvisorInit:
    """测试 Alpha派 投研顾问初始化"""

    def test_init(self):
        advisor = AlphaPaiResearchAdvisor()
        assert advisor.collector is not None


class TestValidateSignalParsing:
    """测试信号验证结果的解析逻辑"""

    @pytest.fixture
    def advisor(self):
        return AlphaPaiResearchAdvisor()

    @pytest.mark.asyncio
    async def test_parse_verdict_agree(self, advisor):
        """测试解析 '赞同' verdict"""
        mock_answer = (
            "**verdict**: 赞同\n"
            "**评估**: 逻辑通顺\n"
            "**风险提醒**: 注意波动\n"
            "**置信度调整建议**: +1"
        )
        with patch.object(
            advisor, "_async_qa", return_value=mock_answer
        ):
            result = await advisor.validate_signal(
                commodity="RB", direction="LONG", entry_price=3000,
                stop_loss=2950, target=3100, core_logic="测试逻辑", confidence=8,
            )
            assert result.verdict == "赞同"
            assert result.confidence_adjustment == 1

    @pytest.mark.asyncio
    async def test_parse_verdict_challenge(self, advisor):
        """测试解析 '质疑' verdict"""
        mock_answer = (
            "**verdict**: 质疑\n"
            "**评估**: 逻辑有漏洞\n"
            "**风险提醒**: 库存超预期累库\n"
            "**置信度调整建议**: -2"
        )
        with patch.object(
            advisor, "_async_qa", return_value=mock_answer
        ):
            result = await advisor.validate_signal(
                commodity="RB", direction="LONG", entry_price=3000,
                stop_loss=2950, target=3100, core_logic="测试逻辑", confidence=8,
            )
            assert result.verdict == "质疑"
            assert result.confidence_adjustment == -2

    @pytest.mark.asyncio
    async def test_parse_fallback_on_empty(self, advisor):
        """测试空回答时的容错处理"""
        with patch.object(advisor, "_async_qa", return_value=""):
            result = await advisor.validate_signal(
                commodity="RB", direction="LONG", entry_price=3000,
                stop_loss=2950, target=3100, core_logic="测试逻辑", confidence=8,
            )
            assert result.verdict == "补充"
            assert result.confidence_adjustment == 0

    @pytest.mark.asyncio
    async def test_parse_no_verdict_marker(self, advisor):
        """测试无 verdict 标记时的容错"""
        mock_answer = "这是一段没有结构化标记的自由文本。"
        with patch.object(advisor, "_async_qa", return_value=mock_answer):
            result = await advisor.validate_signal(
                commodity="RB", direction="LONG", entry_price=3000,
                stop_loss=2950, target=3100, core_logic="测试逻辑", confidence=8,
            )
            assert result.verdict == "补充"
            assert result.assessment == mock_answer


class TestPremarketBriefing:
    """测试盘前简报"""

    @pytest.mark.asyncio
    async def test_briefing_with_news(self):
        advisor = AlphaPaiResearchAdvisor()
        mock_answer = "今日最强驱动因素是资金流动。"
        with patch.object(advisor, "_async_qa", return_value=mock_answer):
            result = await advisor.get_premarket_briefing(
                overnight_news="美联储降息",
                market_moves="原油涨2%",
            )
            assert result == mock_answer

    @pytest.mark.asyncio
    async def test_briefing_empty_on_failure(self):
        advisor = AlphaPaiResearchAdvisor()
        with patch.object(advisor, "_async_qa", return_value=""):
            result = await advisor.get_premarket_briefing()
            assert result == ""


class TestMissedOpportunities:
    """测试 Missed Opportunities 分析"""

    @pytest.mark.asyncio
    async def test_with_performances(self):
        advisor = AlphaPaiResearchAdvisor()
        scan_results = [
            {"code": "RB", "name": "螺纹钢", "open": 3000, "close": 3050,
             "change_pct": 1.5, "amplitude_pct": 2.0, "direction_reason": "观望"},
            {"code": "CU", "name": "铜", "open": 70000, "close": 69500,
             "change_pct": -0.7, "amplitude_pct": 1.5, "direction_reason": "观望"},
        ]
        mock_answer = "螺纹钢出现了明显机会但被错过。"
        with patch.object(advisor, "_async_qa", return_value=mock_answer):
            result = await advisor.analyze_missed_opportunities(
                scan_results=scan_results,
                traded_codes=set(),
            )
            assert result == mock_answer

    @pytest.mark.asyncio
    async def test_empty_when_all_traded(self):
        """当所有品种都已交易时，返回空"""
        advisor = AlphaPaiResearchAdvisor()
        scan_results = [
            {"code": "RB", "name": "螺纹钢", "open": 3000, "close": 3050,
             "change_pct": 1.5, "amplitude_pct": 2.0, "direction_reason": "观望"},
        ]
        with patch.object(advisor, "_async_qa", return_value=""):
            result = await advisor.analyze_missed_opportunities(
                scan_results=scan_results,
                traded_codes={"RB"},
            )
            assert result == ""


class TestWeeklyExploration:
    """测试每周策略探索"""

    @pytest.mark.asyncio
    async def test_exploration(self):
        advisor = AlphaPaiResearchAdvisor()
        mock_answer = "下周重点关注螺纹钢和铜。"
        with patch.object(advisor, "_async_qa", return_value=mock_answer):
            result = await advisor.weekly_exploration(
                week_trades_summary="本周交易2笔",
                hot_topics="原油大涨",
            )
            assert result == mock_answer


class TestExpertDiscuss:
    """测试专家讨论"""

    @pytest.mark.asyncio
    async def test_expert_discuss(self):
        advisor = AlphaPaiResearchAdvisor()
        mock_answer = "这是一个专业讨论回答。"
        with patch.object(
            advisor, "_async_expert_discuss", return_value=mock_answer
        ):
            result = await advisor.expert_discuss(topic="测试话题")
            assert result == mock_answer
