"""
测试策略自我进化系统
"""

import json
import tempfile
import shutil
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
    get_cognition_prompt_additions,
)


@pytest.fixture
def temp_cognition_file(tmp_path, monkeypatch):
    """临时认知库文件"""
    cognition_file = tmp_path / "cognition_library.json"
    monkeypatch.setattr(
        "intraday.evolution.COGNITION_FILE", cognition_file
    )
    return cognition_file


class TestCognitionItem:
    def test_record_verification_win(self):
        item = CognitionItem(
            id="COG-001",
            lesson="跳空>2%时追高风险大",
            category="signal_filter",
            confidence=7,
        )
        item.record_verification(True)
        assert item.verification_count == 1
        assert item.win_count == 1
        assert item.status == "pending"  # 验证次数不够

    def test_record_verification_validated(self):
        item = CognitionItem(
            id="COG-001",
            lesson="跳空>2%时追高风险大",
            category="signal_filter",
            confidence=7,
            verification_count=3,
            win_count=2,
        )
        item.record_verification(True)  # 3胜/4次 -> 75%
        assert item.status == "validated"
        assert item.confidence == 8  # min(10, 7+1)

    def test_to_prompt_rule(self):
        item = CognitionItem(
            id="COG-001",
            lesson="跳空>2%时追高风险大",
            category="signal_filter",
            confidence=8,
            verification_count=5,
            status="validated",
        )
        rule = item.to_prompt_rule()
        assert "✅" in rule
        assert "跳空>2%时追高风险大" in rule
        assert "可靠度:8/10" in rule


class TestCognitionLibrary:
    def test_load_empty(self, temp_cognition_file):
        lib = load_cognition_library()
        assert lib.version == 1
        assert len(lib.items) == 0

    def test_save_and_load(self, temp_cognition_file):
        lib = CognitionLibrary(
            items=[
                CognitionItem(
                    id="COG-001",
                    lesson="跳空>2%时追高风险大",
                    category="signal_filter",
                    confidence=8,
                )
            ]
        )
        save_cognition_library(lib)
        
        loaded = load_cognition_library()
        assert len(loaded.items) == 1
        assert loaded.items[0].lesson == "跳空>2%时追高风险大"

    def test_get_validated_items(self):
        lib = CognitionLibrary(items=[
            CognitionItem(id="1", lesson="高置信度", category="general", confidence=8),
            CognitionItem(id="2", lesson="低置信度", category="general", confidence=5),
            CognitionItem(id="3", lesson="已证伪", category="general", confidence=8, status="invalidated"),
        ])
        valid = lib.get_validated_items(min_confidence=7)
        assert len(valid) == 1
        assert valid[0].id == "1"

    def test_rebuild_prompt_additions(self):
        lib = CognitionLibrary(items=[
            CognitionItem(id="1", lesson="规则A", category="signal_filter", confidence=8, cross_commodity_count=3, cross_date_count=3, uniqueness_score=3),
            CognitionItem(id="2", lesson="规则B", category="entry_timing", confidence=9, cross_commodity_count=5, cross_date_count=4, uniqueness_score=2),
        ])
        text = lib.rebuild_prompt_additions()
        assert "系统进化经验规则" in text
        assert "规则A" in text
        assert "规则B" in text


class TestMergeCognitions:
    def test_merge_new_cognition(self):
        lib = CognitionLibrary()
        new_cogs = [
            CognitionItem(id="NEW-1", lesson="新规则", category="general", confidence=7),
        ]
        result = merge_cognitions(lib, new_cogs)
        assert len(result.items) == 1
        assert result.items[0].lesson == "新规则"

    def test_merge_similar_cognition(self):
        lib = CognitionLibrary(items=[
            CognitionItem(id="OLD-1", lesson="跳空>2%时追高风险大", category="signal_filter", confidence=7, verification_count=2),
        ])
        new_cogs = [
            CognitionItem(id="NEW-1", lesson="跳空>2%时追高风险大", category="signal_filter", confidence=8, verification_count=1),
        ]
        result = merge_cognitions(lib, new_cogs)
        assert len(result.items) == 1
        assert result.items[0].verification_count == 3
        assert result.items[0].confidence == 8


class TestGetEvolvedSystemPrompt:
    def test_no_additions(self):
        base = "Base prompt"
        result = get_evolved_system_prompt(base)
        assert result == base

    def test_with_additions(self):
        base = "Base prompt"
        additions = "\n【系统进化经验规则】\n1. 规则A"
        result = get_evolved_system_prompt(base, additions)
        assert "Base prompt" in result
        assert "规则A" in result


class TestDynamicConfidenceThreshold:
    def test_default(self):
        lib = CognitionLibrary()
        assert get_dynamic_confidence_threshold(lib) == 7

    def test_high_win_rate_lowers_threshold(self):
        lib = CognitionLibrary(items=[
            CognitionItem(id="1", lesson="规则", category="general", confidence=8, verification_count=10, win_count=7, loss_count=3),
        ])
        assert get_dynamic_confidence_threshold(lib, default=7) == 6

    def test_low_win_rate_raises_threshold(self):
        lib = CognitionLibrary(items=[
            CognitionItem(id="1", lesson="规则", category="general", confidence=8, verification_count=10, win_count=3, loss_count=7),
        ])
        assert get_dynamic_confidence_threshold(lib, default=7) == 8


class TestIntradaySignalShouldTrade:
    def test_default_threshold(self):
        sig = IntradaySignal(date="2026-05-26", commodity="RB", direction=Direction.LONG, confidence=8)
        assert sig.should_trade() is True

    def test_low_confidence(self):
        sig = IntradaySignal(date="2026-05-26", commodity="RB", direction=Direction.LONG, confidence=5)
        assert sig.should_trade() is False

    def test_dynamic_threshold(self):
        sig = IntradaySignal(date="2026-05-26", commodity="RB", direction=Direction.LONG, confidence=6)
        assert sig.should_trade(min_confidence=6) is True
        assert sig.should_trade(min_confidence=7) is False
