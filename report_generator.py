from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.macro_agent import MacroIntel, NewsItem
from analyzer import MarketSnapshot
from models import FinalDecision, GeoSignal


def _fmt_dt(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _confidence(geo_signal: GeoSignal) -> float:
    if geo_signal.status == "verified":
        return 0.85
    if geo_signal.status == "confirmed":
        return 0.65
    return 0.35


def _score_rows(decision: FinalDecision) -> List[List[str]]:
    d: Dict[str, Any] = decision.factor_details or {}
    rows: List[List[str]] = []
    def row(name: str, score_key: str, weight_key: str, desc: str) -> List[str]:
        score = float(d.get(score_key, 0.0))
        weight = float(d.get(weight_key, 0.0))
        contrib = score * weight
        return [name, f"{score:.2f}", f"{weight:.2f}", f"{contrib:.2f}", desc]

    rows.append(row("地缘 (Level 5)", "geo_score", "geo_weight", "地缘战争/突发危机"))
    rows.append(row("原油", "oil_score", "oil_weight", "宏观因子（异动可插队）"))
    rows.append(row("美元指数", "dxy_score", "dxy_weight", "宏观因子（异动可插队）"))
    rows.append(row("美债", "bond_score", "bond_weight", "宏观因子（10Y）"))
    rows.append(row("BTC ETF 资金流", "etf_score", "etf_weight", "净流入/流出强度"))
    rows.append(row("市场情绪", "sentiment_score", "sentiment_weight", "资金费率/多空比"))
    rows.append(["总分 (FinalScore)", "-", "-", f"{float(d.get('total_score', 0.0)):.2f}", "加权合成总分"])
    return rows


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _render_intel_list(items: List[NewsItem], limit: int = 6) -> str:
    if not items:
        return "- 无"
    lines = []
    for i in items[:limit]:
        title = i.translated_title or i.title
        age = _age_label(i)
        bang = "!" if age["hours"] is not None and age["hours"] >= 1.0 else ""
        suffix = f"（{age['label']}）{bang}"
        lines.append(f"- {i.source}: {title} ({i.url}) {suffix}")
    return "\n".join(lines)


def _age_label(item: NewsItem) -> Dict[str, Any]:
    if item.timestamp_origin == "fetched":
        return {"label": "刚刚抓取", "hours": 0.0}
    if item.published_at is None:
        return {"label": "未知", "hours": None}
    ts = item.published_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - ts
    minutes = max(0, int(delta.total_seconds() // 60))
    if minutes < 60:
        return {"label": f"{minutes} 分钟前", "hours": minutes / 60.0}
    hours = minutes / 60.0
    if hours < 24:
        return {"label": f"{hours:.1f} 小时前", "hours": hours}
    days = hours / 24.0
    return {"label": f"{days:.1f} 天前", "hours": hours}


def _stars(weight: float) -> str:
    if weight >= 1.2:
        return "★★★★★"
    if weight >= 1.0:
        return "★★★★"
    if weight >= 0.6:
        return "★★★"
    if weight >= 0.5:
        return "★★"
    return "★"


def generate_markdown_report(
    *,
    query: str,
    geo_signal: GeoSignal,
    decision: FinalDecision,
    market: MarketSnapshot,
    intel: MacroIntel,
    generated_at: datetime,
    warnings: Optional[List[str]] = None,
) -> str:
    confidence = _confidence(geo_signal)
    d: Dict[str, Any] = decision.factor_details or {}
    etf_flow_usd = float(d.get("etf_net_inflow_usd", 0.0))
    etf_flow_m = etf_flow_usd / 1_000_000.0
    score_table = _render_table(
        ["因子", "分值", "权重", "贡献", "说明"],
        _score_rows(decision),
    )

    lines: List[str] = []
    lines.append("# 💎 MacroEvent_Quant_V1 中文研报")
    lines.append("")
    lines.append(f"- 生成时间：{_fmt_dt(generated_at)}")
    lines.append(f"- 查询主题：{query}")
    lines.append("")

    lines.append("## 结论摘要")
    lines.append("")
    lines.append(f"- 宏观模式 (Regime)：**{decision.regime}**")
    lines.append(f"- 交易信号：**{decision.signal}**")
    lines.append(f"- 三层验证后确定性信心度：**{confidence:.0%}**")
    lines.append("")

    lines.append("## 核心逻辑评分表")
    lines.append("")
    lines.append(score_table)
    lines.append("")

    lines.append("## 因子优先级仪表盘")
    lines.append("")
    lines.append(
        _render_table(
            ["因子", "权重", "优先级"],
            [
                ["地缘", f"{float(d.get('geo_weight', 0.0)):.2f}", _stars(float(d.get("geo_weight", 0.0)))],
                ["原油", f"{float(d.get('oil_weight', 0.0)):.2f}", _stars(float(d.get("oil_weight", 0.0)))],
                ["美元", f"{float(d.get('dxy_weight', 0.0)):.2f}", _stars(float(d.get("dxy_weight", 0.0)))],
                ["美债", f"{float(d.get('bond_weight', 0.0)):.2f}", _stars(float(d.get("bond_weight", 0.0)))],
                ["ETF", f"{float(d.get('etf_weight', 0.0)):.2f}", _stars(float(d.get("etf_weight", 0.0)))],
                ["情绪", f"{float(d.get('sentiment_weight', 0.0)):.2f}", _stars(float(d.get("sentiment_weight", 0.0)))],
            ],
        )
    )
    lines.append("")

    lines.append("## 市场快照（实时）")
    lines.append("")
    lines.append(
        _render_table(
            ["指标", "数值"],
            [
                ["原油日内涨跌(%)", f"{market.oil_price_change_pct:.2f}"],
                ["美元指数日内涨跌(%)", f"{market.dxy_change_pct:.2f}"],
                ["10年期美债收益率(%)", f"{market.tnx_yield_pct:.2f}"],
                ["10年期美债收益率日内涨跌(%)", f"{market.tnx_change_pct:.2f}"],
                ["昨日 BTC ETF 总净流入(US$m)", f"{etf_flow_m:.1f}"],
                ["BTC/USDT 现价", f"{market.price:.4f}"],
                ["4h MA20", f"{market.ma20_4h:.4f}"],
                ["1d MA20", f"{market.ma20_1d:.4f}"],
            ],
        )
    )
    lines.append("")

    lines.append("## 清算热力分析")
    lines.append("")
    lines.append(
        _render_table(
            ["指标", "数值", "含义"],
            [
                ["资金费率", f"{float(d.get('funding_rate', 0.0)):.6f}", "越高代表多头拥挤，利空更易被放大"],
                ["多空账户比", f"{float(d.get('long_short_ratio', 1.0)):.3f}", ">1 代表多头占优"],
                ["价格低于双 MA20", "是" if bool(d.get("is_below_ma20", False)) else "否", "4h 与 1d 均线过滤"],
            ],
        )
    )
    lines.append("")

    lines.append("## 地缘信号")
    lines.append("")
    lines.append(
        _render_table(
            ["字段", "数值"],
            [
                ["状态", geo_signal.status],
                ["层级", str(geo_signal.tier)],
                ["评分", f"{geo_signal.score:.2f}"],
                ["来源", geo_signal.source],
                ["时间戳", _fmt_dt(geo_signal.timestamp)],
                ["理由/摘要", geo_signal.reason],
            ],
        )
    )
    lines.append("")

    lines.append("## 情报详情")
    lines.append("")
    lines.append("### 第一层（雷达层：Twitter/X 传闻）")
    lines.append(_render_intel_list(intel.tier1))
    lines.append("")
    lines.append("### 第二层（决策层：Reuters / Bloomberg 核心源）")
    lines.append(_render_intel_list(intel.tier2))
    lines.append("")
    lines.append("### 第三层（验证层：AP / Al Jazeera + 油价异动）")
    lines.append(_render_intel_list(intel.tier3))
    lines.append("")

    lines.append("## 背景参考（已硬剔除：发布时间 > 12 小时，不参与评分）")
    lines.append("")
    lines.append("### 第一层背景")
    lines.append(_render_intel_list(intel.tier1_background))
    lines.append("")
    lines.append("### 第二层背景")
    lines.append(_render_intel_list(intel.tier2_background))
    lines.append("")
    lines.append("### 第三层背景")
    lines.append(_render_intel_list(intel.tier3_background))
    lines.append("")

    if warnings:
        lines.append("## 运行告警")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)
