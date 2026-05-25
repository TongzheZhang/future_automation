"""
LLM 集成模块 — 统一调用 OpenRouter API
支持多模型路由、重试、日志记录
"""

import os
import json
import logging
import asyncio
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp
import yaml

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """从文本中提取 JSON，支持 markdown 代码块"""
    text = text.strip()
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 markdown 代码块中的 JSON
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if pattern.startswith(r'```') else match.group())
            except json.JSONDecodeError:
                continue
    
    raise json.JSONDecodeError("无法从文本中提取 JSON", text, 0)


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]


class LLMClient:
    """OpenRouter LLM 客户端"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        llm_cfg = cfg.get("llm", {})
        self.api_key = llm_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key 未配置。请在 config/settings.yaml 或环境变量 OPENROUTER_API_KEY 中设置。")
        
        self.base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self.models = llm_cfg.get("models", {})
        self.default_model = llm_cfg.get("default_model", "anthropic/claude-3.5-sonnet")
        self.fallback_model = llm_cfg.get("fallback_model", "deepseek/deepseek-chat")
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "Policy-Fundamental-Futures-Research",
                },
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        retry: int = 2,
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: OpenAI 格式消息列表 [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称，None 则使用默认模型
            temperature: 温度
            max_tokens: 最大 token 数
            retry: 失败重试次数
        """
        target_model = model or self.default_model
        session = await self._get_session()
        
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        last_error = None
        for attempt in range(retry + 1):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"OpenRouter API 错误 {resp.status}: {text}")
                    
                    data = await resp.json()
                    
                    if "error" in data:
                        raise RuntimeError(f"OpenRouter 返回错误: {data['error']}")
                    
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    logger.info(
                        f"LLM 调用成功 | model={target_model} | "
                        f"prompt_tokens={usage.get('prompt_tokens', 0)} | "
                        f"completion_tokens={usage.get('completion_tokens', 0)}"
                    )
                    
                    return LLMResponse(
                        content=content,
                        model=target_model,
                        usage=usage,
                        raw_response=data,
                    )
            
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{retry + 1}): {e}")
                if attempt < retry:
                    await asyncio.sleep(2 ** attempt)
                    # 切换到 fallback model
                    if attempt == retry - 1 and target_model != self.fallback_model:
                        target_model = self.fallback_model
                        payload["model"] = target_model
                        logger.info(f"切换到 fallback 模型: {target_model}")
        
        raise RuntimeError(f"LLM 调用最终失败: {last_error}")
    
    async def analyze_policy_text(
        self,
        policy_title: str,
        policy_content: str,
        related_commodities: List[str],
    ) -> Dict[str, Any]:
        """
        使用 LLM 分析政策文本，提取对期货品种的影响
        """
        model = self.models.get("policy_analysis", self.default_model)
        
        system_prompt = """你是一位资深的政策分析师，专注于分析中国各部委政策对大宗商品期货市场的影响。
请对给定的政策进行结构化分析，输出 JSON 格式。"""
        
        user_prompt = f"""政策标题：{policy_title}

政策内容：
{policy_content}

需要关注的期货品种：{', '.join(related_commodities)}

请输出以下 JSON 格式分析结果：
{{
    "policy_level": "政策层级（国务院/部委/地方）",
    "policy_type": "政策类型（供给侧改革/需求刺激/环保/进出口/价格调控/其他）",
    "is_direction_change": true/false,
    "direction_change_desc": "如果是转向，描述从什么转向什么",
    "direct_impacts": [
        {{
            "commodity": "品种",
            "direction": "利多/利空/中性",
            "mechanism": "影响机制（50字以内）",
            "strength": "影响强度（强/中/弱）",
            "time_horizon": "影响周期（短期<1月/中期1-3月/长期>3月）"
        }}
    ],
    "transmission_chain": [
        "产业链传导路径的每一步"
    ],
    "market_expectation": "市场是否已充分预期该政策（已充分/部分/未预期）",
    "confidence": 0.0-1.0,
    "key_quotes": ["政策中的关键原文引用"],
    "risks": ["该分析可能存在的风险或不确定性"]
}}

确保 JSON 格式正确，不要添加 markdown 代码块标记。"""
        
        response = await self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.2,
        )
        
        try:
            return extract_json_from_text(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"LLM 返回非 JSON 内容: {response.content[:500]}")
            raise
    
    async def synthesize_signal(
        self,
        policy_analysis: Dict[str, Any],
        fundamental_analysis: Dict[str, Any],
        chain_analysis: Dict[str, Any],
        commodity: str,
    ) -> Dict[str, Any]:
        """
        综合多维度分析，生成交易信号评估
        """
        model = self.models.get("signal_synthesis", self.default_model)
        
        system_prompt = """你是一位资深的期货交易员和宏观分析师。你的任务是根据政策分析、基本面分析和产业链分析，综合评估一个期货交易机会。
请输出结构化的信号评估。"""
        
        user_prompt = f"""品种：{commodity}

【政策分析】
{json.dumps(policy_analysis, ensure_ascii=False, indent=2)}

【基本面分析】
{json.dumps(fundamental_analysis, ensure_ascii=False, indent=2)}

【产业链分析】
{json.dumps(chain_analysis, ensure_ascii=False, indent=2)}

请输出以下 JSON 格式：
{{
    "direction": "LONG/SHORT/NEUTRAL",
    "confidence": 0.0-1.0,
    "conviction_level": "高/中/低",
    "core_logic": "核心交易逻辑（100字以内）",
    "entry_conditions": ["入场条件"],
    "stop_loss_logic": "止损逻辑",
    "target_logic": "目标逻辑",
    "holding_period_days": 预估持仓天数,
    "risk_factors": ["风险因素"],
    "position_sizing": "建议仓位（轻仓/适中/重仓）",
    "required_confirmations": ["还需要哪些确认信息"]
}}

确保 JSON 格式正确。"""
        
        response = await self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.2,
        )
        
        try:
            return extract_json_from_text(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"信号综合返回非 JSON: {response.content[:500]}")
            raise
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# 便捷函数
async def get_llm_client() -> LLMClient:
    return LLMClient()
