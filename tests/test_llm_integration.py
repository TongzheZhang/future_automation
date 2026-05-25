"""
测试 LLM 集成模块
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from research.llm_integration import LLMClient, LLMResponse


@pytest.mark.asyncio
async def test_llm_client_init():
    """测试 LLMClient 初始化"""
    # 配置文件中已配置 API Key，应能正常初始化
    client = LLMClient()
    assert client.api_key is not None
    assert len(client.api_key) > 0


@pytest.mark.asyncio
async def test_analyze_policy_text_mock():
    """测试政策文本分析（mock）"""
    mock_response = LLMResponse(
        content=json.dumps({
            "policy_level": "部委",
            "policy_type": "供给侧改革",
            "is_direction_change": False,
            "direct_impacts": [
                {
                    "commodity": "螺纹钢",
                    "direction": "利多",
                    "mechanism": "限产导致供应收缩",
                    "strength": "强",
                    "time_horizon": "中期",
                }
            ],
            "confidence": 0.85,
        }),
        model="test-model",
        usage={},
        raw_response={},
    )
    
    with patch("research.llm_integration.LLMClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_response
        
        client = LLMClient.__new__(LLMClient)
        client.api_key = "test"
        client.base_url = "https://test.com"
        client.default_model = "test-model"
        client.fallback_model = "test-fallback"
        client.models = {}
        client._session = None
        
        result = await client.analyze_policy_text(
            policy_title="测试政策",
            policy_content="测试内容",
            related_commodities=["螺纹钢"],
        )
        
        assert result["policy_level"] == "部委"
        assert result["confidence"] == 0.85
