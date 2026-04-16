from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from models import FinalDecision, GeoSignal, MacroFactors


class NewsItem(BaseModel):
    title: str
    translated_title: Optional[str] = None
    url: str
    source: str
    published_at: Optional[datetime] = None
    fetched_at: Optional[datetime] = None
    timestamp_origin: Literal["published", "inferred", "fetched"] = "fetched"
    content: Optional[str] = None


class MacroIntel(BaseModel):
    tier1: List[NewsItem] = Field(default_factory=list)
    tier2: List[NewsItem] = Field(default_factory=list)
    tier3: List[NewsItem] = Field(default_factory=list)
    tier1_background: List[NewsItem] = Field(default_factory=list)
    tier2_background: List[NewsItem] = Field(default_factory=list)
    tier3_background: List[NewsItem] = Field(default_factory=list)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    raw = _read_text(dotenv_path)
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hours_passed(published_at: Optional[datetime], now: datetime) -> Optional[float]:
    if published_at is None:
        return None
    age = now - _to_utc(published_at)
    return max(0.0, age.total_seconds() / 3600.0)


def half_life_weight(
    *, published_at: Optional[datetime], now: datetime, initial_weight: float = 1.0, half_life_hours: float = 3.0
) -> float:
    hours = _hours_passed(published_at, now)
    if hours is None:
        return initial_weight
    return float(initial_weight) * (0.5 ** (hours / float(half_life_hours)))


def _parse_datetime_maybe(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v = v / 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if re.fullmatch(r"\d{10,13}", s):
        try:
            v = float(s)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(s)
        return _to_utc(dt)
    except ValueError:
        return None


def _infer_datetime_from_url(url: str) -> Optional[datetime]:
    patterns = [
        r"/(20\d{2})-(\d{2})-(\d{2})/",
        r"-(20\d{2})-(\d{2})-(\d{2})(?:/|$)",
        r"/(20\d{2})/(\d{1,2})/(\d{1,2})/",
    ]
    for p in patterns:
        m = re.search(p, url)
        if not m:
            continue
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d, tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _looks_speculative(title: str, url: str, content: Optional[str]) -> bool:
    text = " ".join([title, url, content or ""]).lower()
    triggers = [
        "opinion",
        "analysis",
        "column",
        "commentary",
        "editorial",
        "rumor",
        "rumour",
        "speculation",
        "unconfirmed",
    ]
    return any(t in text for t in triggers)


def _heuristic_geo_score(text: str) -> float:
    t = text.lower()
    negative = [
        "attack",
        "strike",
        "missile",
        "war",
        "sanction",
        "blockade",
        "invasion",
        "explosion",
        "hostage",
        "killed",
    ]
    positive = [
        "ceasefire",
        "deal",
        "agreement",
        "talks",
        "de-escalation",
        "truce",
        "resume exports",
    ]
    score = 0.0
    score -= 0.4 * sum(1 for w in negative if w in t)
    score += 0.4 * sum(1 for w in positive if w in t)
    return _clamp(score, -2.0, 2.0)


@dataclass(frozen=True)
class TavilySearchParams:
    query: str
    topic: Literal["general", "news"] = "news"
    search_depth: Literal["basic", "advanced"] = "basic"
    max_results: int = 5
    time_range: Optional[str] = None
    include_domains: Optional[List[str]] = None


class TavilyClient:
    def __init__(self, api_key: str, base_url: str = "https://api.tavily.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def search(self, params: TavilySearchParams) -> List[NewsItem]:
        fetched_at = _utcnow()
        url = f"{self._base_url}/search"
        payload: Dict[str, Any] = {
            "query": params.query,
            "topic": params.topic,
            "search_depth": params.search_depth,
            "max_results": params.max_results,
            "include_raw_content": False,
            "include_answer": False,
        }
        if params.time_range:
            payload["time_range"] = params.time_range
        if params.include_domains:
            payload["include_domains"] = params.include_domains
        req = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []

        items: List[NewsItem] = []
        for r in data.get("results", []) or []:
            title = (r.get("title") or "").strip()
            link = (r.get("url") or "").strip()
            content = r.get("content")
            published_at = _parse_datetime_maybe(
                r.get("published_date")
                or r.get("published_time")
                or r.get("published_at")
                or r.get("date")
            )
            timestamp_origin: Literal["published", "inferred", "fetched"] = "fetched"
            if published_at is not None:
                timestamp_origin = "published"
            if published_at is None and link:
                published_at = _infer_datetime_from_url(link)
                if published_at is not None:
                    timestamp_origin = "inferred"
            if not title or not link:
                continue
            items.append(
                NewsItem(
                    title=title,
                    url=link,
                    source=self._guess_source_from_url(link),
                    published_at=published_at,
                    fetched_at=fetched_at,
                    timestamp_origin=timestamp_origin,
                    content=content,
                )
            )
        return items

    @staticmethod
    def _guess_source_from_url(url: str) -> str:
        m = re.search(r"https?://([^/]+)/", url)
        if not m:
            return "unknown"
        host = m.group(1).lower()
        host = host.replace("www.", "")
        return host


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def filter_speculative(self, items: Sequence[NewsItem]) -> List[NewsItem]:
        if not items:
            return []

        annotated = self.annotate_items(items)
        if annotated is None:
            keep: List[NewsItem] = []
            for item in items:
                if _looks_speculative(item.title, item.url, item.content):
                    continue
                keep.append(item)
            return keep

        keep = []
        for item, meta in zip(items, annotated):
            translated_title = meta.get("translated_title") or meta.get("translatedTitle")
            if isinstance(translated_title, str) and translated_title.strip():
                item.translated_title = translated_title.strip()
            inferred_time = meta.get("published_at") or meta.get("publishedAt")
            inferred_dt = _parse_datetime_maybe(inferred_time)
            if inferred_dt is not None and item.published_at is None:
                item.published_at = inferred_dt
                item.timestamp_origin = "inferred"
            if meta.get("speculative") is True:
                continue
            keep.append(item)
        return keep

    def annotate_items(self, items: Sequence[NewsItem]) -> Optional[List[Dict[str, Any]]]:
        prompt_items = [
            {
                "title": i.title,
                "url": i.url,
                "source": i.source,
                "content": (i.content or "")[:600],
            }
            for i in items
        ]
        prompt = (
            "你是系统化交易系统的新闻过滤与标题翻译器。\n"
            "任务：\n"
            "1) 若新闻为猜测性/观点性内容（如 Opinion / Analysis / Column / Editorial）或明显带有推测语气，则标记 speculative=true，否则 speculative=false。\n"
            "2) 请将每条新闻的原始标题翻译成简练、专业的中文金融术语，并作为 translated_title 字段返回。\n"
            "3) 尝试从标题/摘要/链接中推断发布时间（例如 'April 16'、'Thursday' 等时间词）。若能推断，请输出 published_at (ISO8601, UTC, 例如 2026-04-16T08:30:00Z)；若无法推断则输出 null。\n"
            "输出必须为严格 JSON，格式：{\"items\":[{\"speculative\":true/false,\"translated_title\":\"...\",\"published_at\":string|null}, ...]}，items 长度必须与输入一致。\n"
            f"输入：{json.dumps(prompt_items, ensure_ascii=False)}"
        )
        text = self._chat(prompt)
        if text is None:
            return None
        try:
            parsed = self._load_json(text)
            if not isinstance(parsed, dict):
                return None
            rows = parsed.get("items", [])
            if not isinstance(rows, list) or len(rows) != len(items):
                return None
            out: List[Dict[str, Any]] = []
            for r in rows:
                if not isinstance(r, dict):
                    return None
                out.append(r)
            return out
        except Exception:
            return None

    def score_confirmed_signal(self, items: Sequence[NewsItem]) -> Tuple[float, str, List[str]]:
        if not items:
            return 0.0, "", []

        subset = list(items[:6])
        context_items = [{"source": i.source, "title": i.title, "url": i.url} for i in subset]
        prompt = (
            "你是地缘政治新闻信号分析师。\n"
            "请基于下列新闻对全球风险偏好冲击进行打分，分值严格限定在 [-2, +2]。\n"
            "负值代表风险偏好下降（冲突升级、封锁、制裁加码、军事行动、供应中断等），正值代表风险偏好上升（外交斡旋、停火、缓和、协议达成等）。\n"
            "中文摘要要求：请用专业、简练的中文表述打分理由（reason），尽量使用“封锁”“制裁”“外交斡旋”“升级”“缓和”等术语。\n"
            "翻译要求：请将每条新闻的原始标题翻译成简练、专业的中文金融术语，并作为 translated_titles 数组返回（顺序与输入一致）。\n"
            "输出必须为严格 JSON，格式：{\"score\":number,\"reason\":\"中文...\",\"translated_titles\":[\"...\", ...]}。\n"
            f"输入：{json.dumps(context_items, ensure_ascii=False)}"
        )
        text = self._chat(prompt)
        if text is None:
            joined = "\n".join(f"{i.source}: {i.title}" for i in subset)
            return _heuristic_geo_score(joined), "", []
        try:
            parsed = self._load_json(text)
            if not isinstance(parsed, dict):
                raise ValueError("bad-shape")
            obj = parsed
            score = _clamp(float(obj["score"]), -2.0, 2.0)
            reason = str(obj.get("reason", "")).strip()
            translated_titles = obj.get("translated_titles", [])
            if not isinstance(translated_titles, list) or len(translated_titles) != len(subset):
                translated_titles = []
            translated_titles_out: List[str] = []
            for t in translated_titles:
                if isinstance(t, str) and t.strip():
                    translated_titles_out.append(t.strip())
                else:
                    translated_titles_out.append("")
            return score, reason, translated_titles_out
        except Exception:
            joined = "\n".join(f"{i.source}: {i.title}" for i in subset)
            return _heuristic_geo_score(joined), "", []

    @staticmethod
    def _load_json(text: str) -> Optional[object]:
        s = text.strip()
        if not s:
            return None
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass

        obj_start = s.find("{")
        obj_end = s.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                return json.loads(s[obj_start : obj_end + 1])
            except Exception:
                return None

        arr_start = s.find("[")
        arr_end = s.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                return json.loads(s[arr_start : arr_end + 1])
            except Exception:
                return None

        return None

    def _chat(self, user_prompt: str) -> Optional[str]:
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": "deepseek-chat",
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a precise JSON-only assistant."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        req = Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return None

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None


class MacroAgentConfig(BaseModel):
    dotenv_path: str = ".env"
    max_results_per_tier: int = Field(default=5, ge=1, le=20)
    fetch_last_hours: float = Field(default=24.0, gt=0)
    scoring_cutoff_hours: float = Field(default=12.0, gt=0)
    half_life_hours: float = Field(default=3.0, gt=0)
    enable_deepseek: bool = True
    oil_jump_threshold: float = 1.0
    verified_multiplier: float = 1.2


class MacroAgent:
    def __init__(
        self,
        config: Optional[MacroAgentConfig] = None,
        tavily: Optional[TavilyClient] = None,
        deepseek: Optional[DeepSeekClient] = None,
    ) -> None:
        self.config = config or MacroAgentConfig()
        load_dotenv(self.config.dotenv_path)

        tavily_key = os.getenv("TAVILY_API_KEY") or ""
        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or ""

        self.tavily = tavily or (TavilyClient(tavily_key) if tavily_key else None)
        self.deepseek = deepseek or (DeepSeekClient(deepseek_key) if deepseek_key else None)

    def run(self, query: str, macro: MacroFactors, now: Optional[datetime] = None) -> Tuple[GeoSignal, FinalDecision]:
        geo_signal, decision, _ = self.run_with_intel(query=query, macro=macro, now=now)
        return geo_signal, decision

    def run_with_intel(
        self, query: str, macro: MacroFactors, now: Optional[datetime] = None
    ) -> Tuple[GeoSignal, FinalDecision, MacroIntel]:
        now = _to_utc(now) if now else _utcnow()

        tier1_items = self._tier1_radar(query)
        tier2_items = self._tier2_core(query)
        tier3_items = self._tier3_verify(query)

        tier1_items = self._filter_speculative(tier1_items)
        tier2_items = self._filter_speculative(tier2_items)
        tier3_items = self._filter_speculative(tier3_items)

        self._fill_missing_timestamps(tier1_items)
        self._fill_missing_timestamps(tier2_items)
        self._fill_missing_timestamps(tier3_items)

        tier1_scoring, tier1_background = self._split_scoring_background(tier1_items, now)
        tier2_scoring, tier2_background = self._split_scoring_background(tier2_items, now)
        tier3_scoring, tier3_background = self._split_scoring_background(tier3_items, now)

        geo_signal = self._build_geo_signal(
            now=now,
            macro=macro,
            tier1_items=tier1_scoring,
            tier2_items=tier2_scoring,
            tier3_items=tier3_scoring,
            tier2_background=tier2_background,
        )

        decision = self._build_final_decision(geo_signal=geo_signal, macro=macro)
        intel = MacroIntel(
            tier1=tier1_scoring,
            tier2=tier2_scoring,
            tier3=tier3_scoring,
            tier1_background=tier1_background,
            tier2_background=tier2_background,
            tier3_background=tier3_background,
        )
        return geo_signal, decision, intel

    def _tier1_radar(self, query: str) -> List[NewsItem]:
        if not self.tavily:
            return []
        params = TavilySearchParams(
            query=query,
            topic="general",
            search_depth="basic",
            max_results=self.config.max_results_per_tier,
            time_range="day",
            include_domains=["x.com", "twitter.com"],
        )
        return self.tavily.search(params)

    def _tier2_core(self, query: str) -> List[NewsItem]:
        if not self.tavily:
            return []
        params = TavilySearchParams(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=self.config.max_results_per_tier,
            time_range="day",
            include_domains=["reuters.com", "bloomberg.com"],
        )
        return self.tavily.search(params)

    def _tier3_verify(self, query: str) -> List[NewsItem]:
        if not self.tavily:
            return []
        params = TavilySearchParams(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=self.config.max_results_per_tier,
            time_range="day",
            include_domains=["apnews.com", "aljazeera.com"],
        )
        return self.tavily.search(params)

    def _filter_speculative(self, items: List[NewsItem]) -> List[NewsItem]:
        if not items:
            return []
        if self.config.enable_deepseek and self.deepseek:
            return self.deepseek.filter_speculative(items)
        return [i for i in items if not _looks_speculative(i.title, i.url, i.content)]

    def fast_scan(self, query: str, now: Optional[datetime] = None) -> MacroIntel:
        now = _to_utc(now) if now else _utcnow()
        tier1_items = self._tier1_radar(query)
        tier2_items = self._tier2_core(query)
        tier3_items = self._tier3_verify(query)
        self._fill_missing_timestamps(tier1_items)
        self._fill_missing_timestamps(tier2_items)
        self._fill_missing_timestamps(tier3_items)
        tier1_scoring, tier1_background = self._split_scoring_background(tier1_items, now)
        tier2_scoring, tier2_background = self._split_scoring_background(tier2_items, now)
        tier3_scoring, tier3_background = self._split_scoring_background(tier3_items, now)
        return MacroIntel(
            tier1=tier1_scoring,
            tier2=tier2_scoring,
            tier3=tier3_scoring,
            tier1_background=tier1_background,
            tier2_background=tier2_background,
            tier3_background=tier3_background,
        )

    def _split_scoring_background(self, items: List[NewsItem], now: datetime) -> Tuple[List[NewsItem], List[NewsItem]]:
        scoring: List[NewsItem] = []
        background: List[NewsItem] = []
        for i in items:
            hours = _hours_passed(i.published_at, now)
            if hours is not None and hours > self.config.fetch_last_hours:
                continue
            if hours is not None and hours > self.config.scoring_cutoff_hours:
                background.append(i)
            else:
                scoring.append(i)
        return scoring, background

    @staticmethod
    def _fill_missing_timestamps(items: List[NewsItem]) -> None:
        now = _utcnow()
        for i in items:
            if i.fetched_at is None:
                i.fetched_at = now
            if i.published_at is None:
                i.published_at = _to_utc(i.fetched_at)
                i.timestamp_origin = "fetched"

    def _build_geo_signal(
        self,
        now: datetime,
        macro: MacroFactors,
        tier1_items: List[NewsItem],
        tier2_items: List[NewsItem],
        tier3_items: List[NewsItem],
        tier2_background: Optional[List[NewsItem]] = None,
    ) -> GeoSignal:
        memory_score = 0.0
        memory_reason = ""
        if tier2_background:
            bg_score, bg_reason = self._score_confirmed(tier2_background)
            if abs(bg_score) > 1.0:
                memory_score = 0.3 * bg_score
                memory_reason = bg_reason

        if tier2_items:
            score, reason = self._score_confirmed(tier2_items)
            best_ts = self._best_timestamp(tier2_items) or now
            score = score * half_life_weight(
                published_at=best_ts, now=now, initial_weight=1.0, half_life_hours=self.config.half_life_hours
            )
            if memory_score != 0.0 and (score * memory_score) < 0:
                memory_score = 0.0
            score = _clamp(score + memory_score, -2.0, 2.0)
            verified = bool(tier3_items) and (macro.oil_price > self.config.oil_jump_threshold)
            if verified:
                score = _clamp(score * self.config.verified_multiplier, -2.0, 2.0)
                headline_item = tier3_items[0]
                return GeoSignal(
                    score=score,
                    tier=3,
                    status="verified",
                    reason=reason or (headline_item.translated_title or headline_item.title),
                    translated_title=headline_item.translated_title,
                    source=headline_item.source,
                    timestamp=best_ts,
                )
            headline_item = tier2_items[0]
            return GeoSignal(
                score=_clamp(score, -2.0, 2.0),
                tier=2,
                status="confirmed",
                reason=reason or memory_reason or (headline_item.translated_title or headline_item.title),
                translated_title=headline_item.translated_title,
                source=headline_item.source,
                timestamp=best_ts,
            )

        if memory_score != 0.0 and tier2_background:
            best_ts = self._best_timestamp(tier2_background) or now
            headline_item = tier2_background[0]
            return GeoSignal(
                score=_clamp(memory_score, -2.0, 2.0),
                tier=2,
                status="confirmed",
                reason=memory_reason or (headline_item.translated_title or headline_item.title),
                translated_title=headline_item.translated_title,
                source=headline_item.source,
                timestamp=best_ts,
            )

        if tier1_items:
            best_ts = self._best_timestamp(tier1_items) or now
            headline_item = tier1_items[0]
            return GeoSignal(
                score=0.0,
                tier=1,
                status="pending",
                reason="雷达层传闻，待核心源确认",
                translated_title=headline_item.translated_title,
                source=headline_item.source,
                timestamp=best_ts,
            )

        return GeoSignal(
            score=0.0,
            tier=1,
            status="pending",
            reason="未抓取到有效新闻",
            translated_title=None,
            source="none",
            timestamp=now,
        )

    def _score_confirmed(self, items: Sequence[NewsItem]) -> Tuple[float, str]:
        if self.deepseek:
            score, reason, translated_titles = self.deepseek.score_confirmed_signal(items)
            if translated_titles:
                for item, t in zip(list(items[:6]), translated_titles):
                    if not item.translated_title and t:
                        item.translated_title = t
            return score, reason
        joined = "\n".join(f"{i.source}: {i.title}" for i in items[:6])
        return _heuristic_geo_score(joined), ""

    @staticmethod
    def _best_timestamp(items: Sequence[NewsItem]) -> Optional[datetime]:
        timestamps = [i.published_at for i in items if i.published_at is not None]
        if not timestamps:
            return None
        return max(_to_utc(t) for t in timestamps)

    @staticmethod
    def _build_final_decision(geo_signal: GeoSignal, macro: MacroFactors) -> FinalDecision:
        macro_score = _clamp(
            (-0.6 * macro.oil_price) + (-0.4 * macro.dxy) + (0.3 * macro.etf_flow),
            -2.0,
            2.0,
        )
        if macro_score > 0.5:
            regime: Literal["RISK_ON", "RISK_OFF", "NEUTRAL"] = "RISK_ON"
        elif macro_score < -0.5:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        combined = geo_signal.score + macro_score
        if geo_signal.status != "verified":
            signal: Literal["LONG", "SHORT", "NO_TRADE"] = "NO_TRADE"
        elif combined > 0.8:
            signal = "LONG"
        elif combined < -0.8:
            signal = "SHORT"
        else:
            signal = "NO_TRADE"

        return FinalDecision(
            regime=regime,
            macro_score=macro_score,
            signal=signal,
            factor_details={
                "geo_score": geo_signal.score,
                "geo_tier": geo_signal.tier,
                "geo_status": geo_signal.status,
                "combined_score": combined,
                "oil_price": macro.oil_price,
                "dxy": macro.dxy,
                "etf_flow": macro.etf_flow,
            },
        )
