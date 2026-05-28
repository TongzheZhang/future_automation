"""
Alpha 派数据采集器封装
提供程序化调用 Alpha 派 API 的接口
"""

import os
import sys
import json
import logging
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

# 添加 skill 脚本路径
SKILL_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "skills", "alphapai-research", "scripts", "alphapai_client.py"
)

logger = logging.getLogger(__name__)


@dataclass
class AlphaPaiQAResult:
    """Alpha 派问答结果"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    raw_json: Dict[str, Any]


@dataclass
class AlphaPaiRecallResult:
    """Alpha 派数据检索结果"""
    query: str
    documents: List[Dict[str, Any]]
    raw_json: Dict[str, Any]


class AlphaPaiCollector:
    """Alpha 派数据采集器"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.script_path = SKILL_SCRIPT
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"Alpha 派客户端脚本未找到: {self.script_path}")
        
        # 若未传入 api_key，显式读取 config.json
        if api_key is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(self.script_path)), "config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                api_key = config.get("api_key")
        
        self.api_key = api_key
    
    def _run_cli(self, args: List[str], capture_json: bool = True) -> Dict[str, Any]:
        """运行 Alpha 派 CLI"""
        cmd = [sys.executable, self.script_path] + args
        if capture_json and "--json" not in args:
            cmd.append("--json")
        
        logger.debug(f"运行 Alpha 派 CLI: {' '.join(cmd)}")
        
        # 将 api_key 通过环境变量注入，确保子进程在任何环境下都能找到配置
        env = os.environ.copy()
        if self.api_key:
            env["ALPHAPAI_API_KEY"] = self.api_key
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        
        if result.returncode != 0:
            logger.error(f"Alpha 派 CLI 错误: {result.stderr}")
            raise RuntimeError(f"Alpha 派调用失败: {result.stderr}")
        
        if capture_json:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                logger.error(f"Alpha 派返回非 JSON: {result.stdout[:500]}")
                raise
        
        return {"raw_output": result.stdout}
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            result = self._run_cli(["hello"], capture_json=False)
            output = result.get("raw_output", "")
            if any(k in output for k in ("hello", "success", "连接正常", "正常")):
                logger.info(f"Alpha 派健康检查通过")
                return True
            logger.warning(f"Alpha 派健康检查异常: {result}")
            return False
        except Exception as e:
            logger.error(f"Alpha 派健康检查失败: {e}")
            return False
    
    def qa(
        self,
        question: str,
        mode: str = "Think",
        web_search: bool = False,
        deep_reasoning: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AlphaPaiQAResult:
        """
        投研知识问答
        
        Args:
            question: 问题
            mode: Flash 或 Think
            web_search: 是否联网搜索
            deep_reasoning: 是否深度推理
            start_date: 数据筛选开始日期 YYYY-MM-DD
            end_date: 数据筛选结束日期 YYYY-MM-DD
        """
        args = [
            "qa",
            "--question", question,
            "--mode", mode,
        ]
        if web_search:
            args.append("--web-search")
        if deep_reasoning:
            args.append("--deep-reasoning")
        if start_date:
            args.extend(["--start", start_date])
        if end_date:
            args.extend(["--end", end_date])
        
        data = self._run_cli(args)
        
        return AlphaPaiQAResult(
            question=question,
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            raw_json=data,
        )
    
    def recall(
        self,
        query: str,
        doc_types: Optional[List[str]] = None,
        no_cutoff: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AlphaPaiRecallResult:
        """
        投研数据检索（RAG 原始数据）
        
        Args:
            query: 查询问题
            doc_types: 数据类型列表，如 ["report", "roadShow", "comment"]
            no_cutoff: 返回截断前完整内容
            start_date: 开始日期
            end_date: 结束日期
        """
        args = [
            "recall",
            "--query", query,
        ]
        if doc_types:
            args.extend(["--type", ",".join(doc_types)])
        if no_cutoff:
            args.append("--no-cutoff")
        if start_date:
            args.extend(["--start", start_date])
        if end_date:
            args.extend(["--end", end_date])
        
        data = self._run_cli(args)
        
        return AlphaPaiRecallResult(
            query=query,
            documents=data.get("documents", []),
            raw_json=data,
        )
    
    def industry_one_page(
        self,
        industry: str,
    ) -> str:
        """
        行业一页纸（mode 11）
        """
        args = [
            "agent",
            "--mode", "11",
            "--question", f"{industry}的行业一页纸",
            "--industry", industry,
        ]
        # agent 模式输出的是 markdown 文本，不使用 --json
        result = self._run_cli(args, capture_json=False)
        return result.get("raw_output", "")
    
    def investment_logic(
        self,
        stock_code: str,
        stock_name: str,
    ) -> str:
        """
        投资逻辑梳理（mode 7）
        """
        args = [
            "agent",
            "--mode", "7",
            "--question", f"{stock_name}（{stock_code}）的投资逻辑",
            "--stock", f"{stock_code}:{stock_name}",
        ]
        result = self._run_cli(args, capture_json=False)
        return result.get("raw_output", "")
    
    def search_images(
        self,
        query: str,
        topk: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜图表
        """
        args = [
            "image",
            "--query", query,
            "--topk", str(topk),
        ]
        if start_date:
            args.extend(["--start", start_date])
        if end_date:
            args.extend(["--end", end_date])
        
        data = self._run_cli(args)
        return data.get("images", [])
    
    def watchlist(self) -> List[Dict[str, Any]]:
        """
        查看自选股列表
        """
        data = self._run_cli(["watchlist"])
        return data.get("groups", [])
    
    def get_fundamental_data(
        self,
        keywords: List[str],
        doc_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        days_back: int = 30,
        max_docs: Optional[int] = None,
        max_chars_per_doc: Optional[int] = None,
    ) -> str:
        """
        通过 recall 获取品种基本面原始数据，完整返回原始内容。
        默认不截断；如需控制长度，请在调用方通过 max_docs / max_chars_per_doc 指定。
        """
        if doc_types is None:
            doc_types = ["report", "roadShow", "comment", "ann"]

        if not start_date:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        query = " ".join(keywords[:3])
        try:
            result = self.recall(
                query=query,
                doc_types=doc_types,
                start_date=start_date,
            )

            if not result.documents:
                return "暂无相关数据"

            lines = []
            for i, doc in enumerate(result.documents, 1):
                if max_docs is not None and i > max_docs:
                    break
                title = doc.get("title", "")
                content = doc.get("content", "")
                if max_chars_per_doc is not None:
                    content = content[:max_chars_per_doc]
                doc_type = doc.get("recallType", "")
                pub_date = doc.get("publishDate", "")
                lines.append(f"[{i}] [{doc_type}] {title} ({pub_date})")
                if content:
                    lines.append(f"    {content}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Alpha 派基本面数据获取失败: {e}")
            return "数据获取失败"
    
    def expert_discuss(
        self,
        topic: str,
        context: str = "",
    ) -> str:
        """
        以专家身份进行讨论（智能问答模式）
        适用于复盘讨论、策略优化等深度分析场景
        """
        question = topic
        if context:
            question = f"{context}\n\n问题: {topic}"
        
        try:
            result = self.qa(
                question=question,
                mode="Think",
                start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            )
            return result.answer
        except Exception as e:
            logger.error(f"Alpha 派专家讨论失败: {e}")
            return "讨论失败"
    
    def industry_one_page(self, industry: str) -> str:
        """行业一页纸（mode 11）"""
        try:
            return self.agent_industry_one_page(industry)
        except Exception as e:
            logger.error(f"行业分析失败: {e}")
            return ""
    
    def agent_industry_one_page(self, industry: str) -> str:
        """Agent mode 11 - 行业一页纸"""
        args = [
            "agent",
            "--mode", "11",
            "--question", f"{industry}的行业一页纸",
            "--industry", industry,
        ]
        result = self._run_cli(args, capture_json=False)
        return result.get("raw_output", "")


# 便捷函数
def get_alpha_pai_collector() -> AlphaPaiCollector:
    return AlphaPaiCollector()
