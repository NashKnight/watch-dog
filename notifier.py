#!/usr/bin/env python3
"""飞书自定义机器人通知封装 (interactive 卡片)。

- 红色卡片: 异常告警 (Failed / Killed / 卡死无进度)
- 蓝色卡片: 进度汇报 / 启动
- 绿色卡片: 训练完成 / 从异常恢复

webhook 获取优先级:
    显式传入 webhook > 环境变量 FEISHU_WEBHOOK > config.json 里的 feishu_webhook
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


LEVEL_TEMPLATE = {
    "alert": "red",
    "warn": "orange",
    "info": "blue",
    "ok": "green",
}


class FeishuNotifier:
    def __init__(self, webhook: Optional[str] = None, logger=None, timeout: int = 15):
        self.webhook = webhook or os.environ.get("FEISHU_WEBHOOK")
        self.logger = logger
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.webhook)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)
        else:
            print(msg)

    def send(self, title: str, content: str, level: str = "info") -> bool:
        """发送一张 markdown 内容卡片。level ∈ {alert,warn,info,ok}。"""
        if not self.enabled:
            self._log("[notifier] 未配置 webhook, 跳过发送 (仅本地记录)。")
            return False
        if requests is None:
            self._log("[notifier] 缺少 requests 库, 无法发送。")
            return False

        template = LEVEL_TEMPLATE.get(level, "blue")
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": template,
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": content}}
                ],
            },
        }
        try:
            resp = requests.post(self.webhook, json=payload, timeout=self.timeout)
            body = {}
            try:
                body = resp.json()
            except Exception:
                pass
            # 自定义机器人成功时 code==0 (或 StatusCode==0 老格式)
            code = body.get("code", body.get("StatusCode", 0))
            if code not in (0, None):
                self._log(f"[notifier] 飞书发送失败: {resp.status_code} {resp.text[:300]}")
                return False
            return True
        except Exception as e:  # noqa: BLE001
            self._log(f"[notifier] 请求飞书接口异常: {e}")
            return False


def load_webhook_from_config(config_path: str) -> Optional[str]:
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("feishu_webhook")
        except Exception:
            return None
    return None
