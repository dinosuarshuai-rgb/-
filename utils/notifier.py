from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _extract_summary(report_md: str) -> Tuple[str, str, str]:
    signal = _find_first(report_md, r"交易信号：\*\*(.*?)\*\*") or ""
    regime = _find_first(report_md, r"宏观模式\s*\(Regime\)：\*\*(.*?)\*\*") or ""
    confidence = _find_first(report_md, r"确定性信心度：\*\*(.*?)\*\*") or ""
    return signal.strip(), regime.strip(), confidence.strip()


def _extract_top3_intel(report_md: str) -> List[str]:
    m = re.search(r"^## 情报详情\s*$", report_md, flags=re.MULTILINE)
    if not m:
        return []
    tail = report_md[m.end() :]
    end = re.search(r"^##\s+", tail, flags=re.MULTILINE)
    block = tail[: end.start()] if end else tail
    lines = []
    for ln in block.splitlines():
        ln = ln.strip()
        if ln.startswith("- "):
            lines.append(ln[2:].strip())
        if len(lines) >= 3:
            break
    return lines


def _find_first(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1)


def send_wechat(report_md: str) -> bool:
    sendkey = (os.getenv("SERVERCHAN_SENDKEY") or os.getenv("WECOM_WEBHOOK_URL") or "").strip()
    if not sendkey:
        return False

    signal, regime, confidence = _extract_summary(report_md)
    top3 = _extract_top3_intel(report_md)

    title = "MacroEvent_Quant_V1 哨兵告警"
    parts = []
    if signal or regime or confidence:
        parts.append(f"- 交易信号：{signal or 'N/A'}")
        parts.append(f"- 宏观模式：{regime or 'N/A'}")
        parts.append(f"- 确定性信心度：{confidence or 'N/A'}")
    if top3:
        parts.append("")
        parts.append("TOP3 情报：")
        parts.extend([f"- {x}" for x in top3])
    desp = "\n".join(parts) if parts else report_md[:1500]

    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    payload = urlencode({"title": title, "desp": desp}).encode("utf-8")
    req = Request(url=url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    try:
        with urlopen(req, timeout=20) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False
