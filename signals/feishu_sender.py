"""
飞书消息推送模块

通过飞书 API 直接发送消息和文件（不依赖 OpenClaw CLI）
"""
import json
import os
import time
from pathlib import Path
from loguru import logger

import requests

# ============ 飞书应用配置 ============
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "cli_a948f7ed3bb91cb6")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "gmYGv4mEN7t9212su0ycQcbL24aJBAMH")
FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


def _load_sender_config():
    """加载推送配置（从 settings.yaml）"""
    cfg = {}
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        import yaml
        try:
            with open(config_path) as f:
                settings = yaml.safe_load(f)
            cfg = settings.get("notify", {}) if settings else {}
        except Exception as e:
            logger.warning(f"读取 settings.yaml 失败: {e}")
    return cfg


_sender_cfg = _load_sender_config()

FEISHU_OPEN_ID = os.getenv(
    "FEISHU_OPEN_ID",
    _sender_cfg.get("feishu_open_id", "ou_5c92160942520c8fe9af332b6e255e75"),
)

MAX_RETRIES = int(os.getenv("FEISHU_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("FEISHU_RETRY_DELAY", "2.0"))

# token 缓存
_token_cache = {"token": None, "expires_at": 0}


def _get_tenant_token() -> str:
    """获取飞书 tenant_access_token（带缓存）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
    resp = requests.post(
        url,
        json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"获取飞书 token 失败: {data}")
    _token_cache["token"] = data["tenant_access_token"]
    _token_cache["expires_at"] = now + data.get("expire", 7200)
    return _token_cache["token"]


def _truncate_text(text: str, max_bytes: int = 29000) -> str:
    """截断文本到飞书限制（约 30KB）"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    # 确保不截断多字节字符
    return truncated.decode("utf-8", errors="ignore") + "\n\n...（消息过长已截断）"


def send_feishu_message(message: str, target: str = None) -> bool:
    """
    通过飞书 API 发送文本消息

    Args:
        message: 消息内容（纯文本）
        target: 目标用户 open_id（默认从配置读取）

    Returns:
        bool: 是否发送成功
    """
    if target is None:
        target = FEISHU_OPEN_ID

    message = _truncate_text(message)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            token = _get_tenant_token()
            url = f"{FEISHU_API_BASE}/im/v1/messages?receive_id_type=open_id"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {
                "receive_id": target,
                "msg_type": "text",
                "content": json.dumps({"text": message}),
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            data = resp.json()
            if data.get("code") == 0:
                logger.info("  飞书消息发送成功")
                return True
            else:
                logger.warning(
                    f"  发送失败 (尝试 {attempt}/{MAX_RETRIES}): code={data.get('code')} msg={data.get('msg')}"
                )
                # token 可能过期，清除缓存重试
                _token_cache["token"] = None
        except Exception as e:
            logger.warning(f"  发送异常 (尝试 {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    logger.error(f"  飞书消息发送失败，已重试 {MAX_RETRIES} 次")
    return False


def send_signal_report(report_text: str, target: str = None) -> bool:
    """发送信号报告到飞书（同名兼容接口）"""
    return send_feishu_message(report_text, target)


def send_file_via_feishu(file_path: str, caption: str = "", target: str = None) -> bool:
    """
    通过飞书 API 发送文件消息

    Args:
        file_path: 文件路径
        caption: 文件说明
        target: 目标用户 open_id

    Returns:
        bool: 是否发送成功
    """
    if target is None:
        target = FEISHU_OPEN_ID

    src = Path(file_path)
    if not src.exists():
        logger.error(f"  文件不存在: {file_path}")
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            token = _get_tenant_token()

            # 1. 上传文件
            file_name = src.name
            file_size = src.stat().st_size
            ext = src.suffix.lower().lstrip(".")
            mime_map = {
                "pdf": "application/pdf",
                "doc": "application/msword",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "xls": "application/vnd.ms-excel",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "csv": "text/csv",
                "txt": "text/plain",
                "zip": "application/zip",
            }
            mime = mime_map.get(ext, "application/octet-stream")

            with open(src, "rb") as f:
                upload_headers = {"Authorization": f"Bearer {token}"}
                upload_data = {
                    "file_name": file_name,
                    "file_size": str(file_size),
                    "file_type": "stream",
                }
                files = {"file": (file_name, f, mime)}
                upload_resp = requests.post(
                    f"{FEISHU_API_BASE}/im/v1/files",
                    headers=upload_headers,
                    data=upload_data,
                    files=files,
                    timeout=30,
                )
            upload_data = upload_resp.json()
            if upload_data.get("code") != 0:
                logger.warning(f"  文件上传失败 (尝试 {attempt}): {upload_data}")
                _token_cache["token"] = None
                continue

            file_key = upload_data["data"]["file_key"]

            # 2. 发送文件消息
            send_url = f"{FEISHU_API_BASE}/im/v1/messages?receive_id_type=open_id"
            send_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            send_payload = {
                "receive_id": target,
                "msg_type": "file",
                "content": json.dumps({"file_key": file_key}),
            }
            send_resp = requests.post(send_url, headers=send_headers, json=send_payload, timeout=10)
            send_data = send_resp.json()

            if send_data.get("code") == 0:
                logger.info(f"  文件发送成功: {file_name}")
                # 如果还有 caption，再发一条文本消息
                if caption:
                    send_feishu_message(f"📎 {caption}", target)
                return True
            else:
                logger.warning(f"  文件发送失败 (尝试 {attempt}): {send_data}")

        except Exception as e:
            logger.warning(f"  文件发送异常 (尝试 {attempt}/{MAX_RETRIES}): {e}")
            _token_cache["token"] = None

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    logger.error(f"  文件发送失败，已重试 {MAX_RETRIES} 次")
    return False


if __name__ == "__main__":
    send_feishu_message("🧪 Spark系统飞书通道已修复 (API直连模式) - %s" % time.strftime("%Y-%m-%d %H:%M"))
