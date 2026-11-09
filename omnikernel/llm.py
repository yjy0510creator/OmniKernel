"""LLM 客户端：OpenAI 兼容接口。未配置环境变量时 available()=False，
上层 Agent 自动退化为纯模板模式——保证无 LLM 环境（如离线评测机）结果仍可复现。"""
from __future__ import annotations

import json
import os
import urllib.request


class LLMClient:
    def __init__(self):
        self.base = os.environ.get("OMNIKERNEL_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("OMNIKERNEL_LLM_API_KEY", "")
        self.model = os.environ.get("OMNIKERNEL_LLM_MODEL", "")

    def available(self) -> bool:
        return bool(self.base and self.model)

    def chat(self, system: str, user: str, temperature: float = 0.2,
             timeout: int = 120) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def extract_code(reply: str) -> str:
        """从模型回复中提取 ``` 围栏内代码；无围栏则原样返回。"""
        out, in_fence, fence = [], False, ""
        for line in reply.splitlines():
            s = line.strip()
            if not in_fence and s.startswith("```"):
                in_fence, fence = True, s
                continue
            if in_fence and s == "```" or (in_fence and s.startswith("```") and s != fence):
                in_fence = False
                continue
            out.append(line)
        text = "\n".join(out).strip()
        return text if text else reply.strip()
