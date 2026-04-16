from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import os
import time
import sys

from agents.macro_agent import MacroAgent, MacroAgentConfig, load_dotenv
from analyzer import DecisionContext, MacroDecisionEngine
from models import MacroFactors
from report_generator import generate_markdown_report
from utils.notifier import send_wechat


STATE_PATH = ".sentinel_state.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception:
        return


def _contains_high_value_keywords(titles: list[str]) -> bool:
    keywords = ["attack", "war", "blockade", "hormuz", "sanction"]
    for t in titles:
        lt = t.lower()
        if any(k in lt for k in keywords):
            return True
    return False


def _collect_titles(intel) -> list[str]:
    out: list[str] = []
    for tier in [intel.tier3, intel.tier2, intel.tier1]:
        for i in tier:
            out.append(i.title)
    return out


def fast_scan(query: str) -> tuple[bool, dict]:
    engine = MacroDecisionEngine()
    oil_change = 0.0
    dxy_change = 0.0
    try:
        oil_change, dxy_change = engine.fetch_macro_quick()
    except Exception:
        pass

    agent = MacroAgent(config=MacroAgentConfig(enable_deepseek=False))
    intel = agent.fast_scan(query=query, now=_utcnow())
    titles = _collect_titles(intel)
    keyword_trigger = _contains_high_value_keywords(titles)
    macro_trigger = abs(oil_change) > 2.5 or abs(dxy_change) > 2.5

    state = _load_state()
    last_full = state.get("last_full_analysis_utc")
    time_trigger = True
    if isinstance(last_full, str):
        try:
            last_dt = datetime.fromisoformat(last_full.replace("Z", "+00:00"))
            time_trigger = (_utcnow() - last_dt).total_seconds() >= 4 * 3600
        except Exception:
            time_trigger = True

    triggered = keyword_trigger or macro_trigger or time_trigger
    meta = {
        "keyword_trigger": keyword_trigger,
        "macro_trigger": macro_trigger,
        "time_trigger": time_trigger,
        "oil_change_pct": oil_change,
        "dxy_change_pct": dxy_change,
    }
    return triggered, meta


def full_analysis(query: str, push: bool) -> str:
    engine = MacroDecisionEngine()
    market = engine.build_market_snapshot()

    agent = MacroAgent(config=MacroAgentConfig(enable_deepseek=True))
    geo_signal, _, intel = agent.run_with_intel(
        query=query,
        macro=MacroFactors(
            oil_price=market.oil_price_change_pct,
            dxy=market.dxy_change_pct,
            etf_flow=0.0,
        ),
        now=_utcnow(),
    )

    decision = engine.analyze(DecisionContext(geo_signal=geo_signal, market=market))
    report = generate_markdown_report(
        query=query,
        geo_signal=geo_signal,
        decision=decision,
        market=market,
        intel=intel,
        generated_at=_utcnow(),
        warnings=engine.last_warnings,
    )

    if push:
        send_wechat(report)
    return report


def sentinel_once(query: str, push: bool) -> None:
    triggered, meta = fast_scan(query)
    if not triggered:
        print(json.dumps({"triggered": False, "meta": meta}, ensure_ascii=False))
        return

    report = full_analysis(query, push=False)
    if push:
        send_wechat(report)

    state = _load_state()
    state["last_full_analysis_utc"] = _utcnow().isoformat().replace("+00:00", "Z")
    state["last_trigger"] = meta
    _save_state(state)

    print(report)


def sentinel_loop(query: str, interval_sec: int, push: bool) -> None:
    while True:
        triggered, meta = fast_scan(query)
        if triggered:
            report = full_analysis(query, push=push)
            print(report)
            state = _load_state()
            state["last_full_analysis_utc"] = _utcnow().isoformat().replace("+00:00", "Z")
            state["last_trigger"] = meta
            _save_state(state)
        else:
            print(
                f"[fast_scan] no trigger | oil={meta['oil_change_pct']:.2f}% dxy={meta['dxy_change_pct']:.2f}%",
                file=sys.stderr,
            )
        time.sleep(max(5, int(interval_sec)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast_scan", "full_analysis", "sentinel"], default="full_analysis")
    parser.add_argument("--query", default="Iran geopolitical news latest escalation sanctions oil shipping")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")

    if args.mode == "fast_scan":
        triggered, meta = fast_scan(args.query)
        print(json.dumps({"triggered": triggered, "meta": meta}, ensure_ascii=False))
        return

    if args.mode == "sentinel":
        if args.once or os.getenv("GITHUB_ACTIONS") == "true":
            sentinel_once(args.query, push=args.push)
        else:
            sentinel_loop(args.query, interval_sec=args.interval, push=args.push)
        return

    try:
        report = full_analysis(args.query, push=args.push)
        print(report)
    except RuntimeError as exc:
        print(f"[错误] 全量分析失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
