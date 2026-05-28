"""
端到端测试：策略自我进化完整流程
- 模拟复盘数据
- 提取认知
- 合并认知库
- 更新 prompt
- 更新框架文档
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from intraday.models import (
    CognitionItem, CognitionLibrary, DailyReview,
    IntradayTrade, IntradaySignal, Direction, TradeStatus,
)
from intraday.evolution import (
    load_cognition_library, save_cognition_library,
    merge_cognitions, get_evolved_system_prompt,
    get_dynamic_confidence_threshold,
)


@pytest.fixture
def mock_review():
    """构造一个模拟复盘（RB SHORT 盈利）"""
    trade = IntradayTrade(
        date="2026-05-25",
        commodity="RB",
        direction=Direction.SHORT,
        signal_entry=3186.0,
        signal_stop=3205.0,
        signal_target=3140.0,
        confidence=8,
        core_logic="隔夜利空+跳空低开，空头主动开仓",
        actual_entry=3186.0,
        actual_exit=3146.0,
        day_high=3195.0,
        day_low=3140.0,
        day_close=3146.0,
        pnl=400.0,
        status=TradeStatus.WIN,
    )
    
    review = DailyReview(
        date="2026-05-25",
        signals=[],
        trades=[trade],
    )
    review.compute_stats()
    review.review_summary = (
        "本次判断准确。隔夜利空消息导致跳空低开，空头动能充足。"
        "教训：跳空>1.5%且伴随利空确认时，应提高置信度，果断入场。"
    )
    return review


class TestEvolutionE2E:
    def test_full_evolution_pipeline_no_llm(self, tmp_path, monkeypatch, mock_review):
        """
        端到端测试（跳过 LLM 提取，直接注入认知）
        验证：认知合并 → prompt 更新 → 动态阈值
        """
        # Mock 认知库路径
        cognition_file = tmp_path / "cognition_library.json"
        monkeypatch.setattr("intraday.evolution.COGNITION_FILE", cognition_file)
        
        # 1. 初始认知库为空
        lib = load_cognition_library()
        assert len(lib.items) == 0
        
        # 2. 模拟从复盘提取的认知（跳过 LLM）
        extracted = [
            CognitionItem(
                id="COG-20260525-01",
                lesson="跳空>1.5%且伴随利空确认时，应提高置信度果断入场",
                category="signal_filter",
                confidence=8,
                affected_commodities=["RB"],
                source_trade_date="2026-05-25",
                verification_count=1,
                win_count=1,
            ),
            CognitionItem(
                id="COG-20260525-02",
                lesson="尾盘30分钟优先锁定盈利，避免尾盘波动",
                category="exit_timing",
                confidence=7,
                affected_commodities=["RB"],
                source_trade_date="2026-05-25",
                verification_count=1,
                win_count=1,
            ),
        ]
        mock_review.extracted_cognitions = extracted
        
        # 3. 合并认知
        lib = merge_cognitions(lib, extracted)
        save_cognition_library(lib)
        
        # 4. 验证认知库
        loaded = load_cognition_library()
        assert len(loaded.items) == 2
        assert loaded.items[0].category == "signal_filter"
        
        # 5. 验证 prompt 追加文本（要求普适性，测试数据需满足条件）
        # 手动设置普适性条件使认知能进入 prompt
        for item in loaded.items:
            item.cross_commodity_count = max(3, item.cross_commodity_count)
            item.cross_date_count = max(3, item.cross_date_count)
            item.uniqueness_score = min(5, item.uniqueness_score)
        loaded.evolved_prompt_additions = loaded.rebuild_prompt_additions()
        prompt_additions = loaded.evolved_prompt_additions
        assert "系统进化经验规则" in prompt_additions
        assert "跳空>1.5%" in prompt_additions
        assert "尾盘30分钟" in prompt_additions
        
        # 6. 验证 strategy prompt 拼接
        base_prompt = "你是一个只做日内T+0的期货交易员"
        evolved = get_evolved_system_prompt(base_prompt, prompt_additions)
        assert base_prompt in evolved
        assert "系统进化经验规则" in evolved
        
        # 7. 验证动态阈值（1胜0败，样本不足，保持默认）
        threshold = get_dynamic_confidence_threshold(loaded)
        assert threshold == 7  # 样本不足5次
        
        # 8. 模拟多次验证后阈值变化
        for i in range(10):
            loaded.items[0].record_verification(True)  # 持续胜利
        save_cognition_library(loaded)
        
        loaded2 = load_cognition_library()
        threshold2 = get_dynamic_confidence_threshold(loaded2)
        assert threshold2 == 6  # 高胜率降低阈值
    
    def test_cognition_merge_similar(self, tmp_path, monkeypatch):
        """测试相似认知的合并"""
        cognition_file = tmp_path / "cognition_library.json"
        monkeypatch.setattr("intraday.evolution.COGNITION_FILE", cognition_file)
        
        # 先存一条认知
        lib = CognitionLibrary(items=[
            CognitionItem(
                id="OLD-1",
                lesson="跳空>2%时追高风险大，偏向观望",
                category="signal_filter",
                confidence=7,
                verification_count=3,
                win_count=2,
            )
        ])
        save_cognition_library(lib)
        
        # 再提取一条几乎相同的认知
        new_cogs = [
            CognitionItem(
                id="NEW-1",
                lesson="跳空>2%时追高风险大，偏向观望",
                category="signal_filter",
                confidence=8,
                verification_count=1,
                win_count=1,
            )
        ]
        
        loaded = load_cognition_library()
        merged = merge_cognitions(loaded, new_cogs)
        save_cognition_library(merged)
        
        final = load_cognition_library()
        assert len(final.items) == 1
        assert final.items[0].verification_count == 4  # 3+1
        assert final.items[0].confidence == 8  # 取更高
    
    def test_framework_update(self, tmp_path, monkeypatch):
        """测试框架文档更新"""
        # 创建临时框架文档
        framework_file = tmp_path / "research_framework.md"
        framework_file.write_text("# Test Framework\n\n## 八、日内交易认知积累\n\n*empty*\n", encoding="utf-8")
        monkeypatch.setattr(
            "intraday.evolution.Path",
            lambda *args: tmp_path if str(args[0]) == "docs" else Path(*args)
        )
        # 简化：直接验证 update_framework_with_cognitions 的输入输出逻辑
        lib = CognitionLibrary(items=[
            CognitionItem(
                id="1", lesson="规则A", category="signal_filter", confidence=8,
                verification_count=5, status="validated"
            ),
            CognitionItem(
                id="2", lesson="规则B", category="entry_timing", confidence=7,
                verification_count=3, status="pending"
            ),
        ])
        
        # 直接测试函数，传入临时文件路径
        import asyncio
        from intraday.evolution import update_framework_with_cognitions
        asyncio.run(update_framework_with_cognitions(lib, framework_path=framework_file))
        
        updated = framework_file.read_text(encoding="utf-8")
        assert "规则A" in updated
        assert "规则B" in updated
        assert "上次更新" in updated
