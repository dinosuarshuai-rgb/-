from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Dict, List, Literal, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from datetime import datetime, timezone

from models import FinalDecision, GeoSignal


@dataclass(frozen=True)
class MarketSnapshot:
    oil_price_change_pct: float
    dxy_change_pct: float
    tnx_yield_pct: float
    tnx_change_pct: float
    etf_net_inflow_usd: float
    price: float
    ma20_4h: float
    ma20_1d: float
    funding_rate: float
    long_short_ratio: float


@dataclass(frozen=True)
class DecisionContext:
    geo_signal: GeoSignal
    market: MarketSnapshot


class MacroDecisionEngine:
    def __init__(self) -> None:
        self.last_warnings: List[str] = []

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _stooq_last_prev(symbol: str) -> Tuple[float, float]:
        import requests

        url = f"https://stooq.com/q/?s={symbol}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = resp.text

        sym = symbol.lower()
        m_prev = re.search(rf"id=aq_{re.escape(sym)}_p>([0-9]+(?:\.[0-9]+)?)<", text)
        m_last = (
            re.search(rf"id=aq_{re.escape(sym)}_c3>([0-9]+(?:\.[0-9]+)?)<", text)
            or re.search(rf"id=aq_{re.escape(sym)}_c2\\|3>([0-9]+(?:\.[0-9]+)?)<", text)
            or re.search(rf"id=aq_{re.escape(sym)}_c2>([0-9]+(?:\.[0-9]+)?)<", text)
        )
        if not m_last or not m_prev:
            raise RuntimeError(f"Stooq 解析失败：{symbol} 找不到 last/prev 字段。")
        return float(m_last.group(1)), float(m_prev.group(1))

    def _fetch_btc_realtime_and_ma(self) -> Tuple[float, float, float]:
        import requests

        okx_url = "https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT"
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    okx_url,
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                )
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data")
                last = data[0].get("last") if isinstance(data, list) and data else None
                if last is None:
                    raise RuntimeError("OKX 返回数据缺少 last 字段。")
                last_price = float(last)
                return last_price, last_price, last_price
            except Exception as exc:
                last_error = exc
                time.sleep(0.8 * (attempt + 1))

        try:
            # 备用源：CoinGecko 简单价格接口
            cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            resp = requests.get(cg_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            payload = resp.json()
            last_price = float(payload["bitcoin"]["usd"])
            self.last_warnings.append("OKX 价格拉取失败，已切换 CoinGecko 备用源。")
            return last_price, last_price, last_price
        except Exception:
            mock = os.getenv("MOCK_BTC_PRICE", "").strip()
            if mock:
                try:
                    p = float(mock)
                    if p > 0:
                        self.last_warnings.append("BTC 价格使用 MOCK_BTC_PRICE。")
                        return p, p, p
                except Exception:
                    pass
            raise RuntimeError(f"BTC 实时价格获取失败（OKX/CoinGecko/Mock）: {last_error}") from last_error

    def _fetch_macro_changes(self) -> Tuple[float, float]:
        oil_last, oil_prev = self._stooq_last_prev("CL.F")
        dxy_last, dxy_prev = self._stooq_last_prev("DX.F")
        oil_change = 0.0 if oil_prev == 0 else ((oil_last - oil_prev) / oil_prev) * 100.0
        dxy_change = 0.0 if dxy_prev == 0 else ((dxy_last - dxy_prev) / dxy_prev) * 100.0
        return float(oil_change), float(dxy_change)

    def _fetch_tnx(self) -> Tuple[float, float]:
        last, prev = self._stooq_last_prev("10YUSY.B")
        change_pct = 0.0 if prev == 0 else ((last - prev) / prev) * 100.0
        return float(last), float(change_pct)

    @staticmethod
    def _parse_usm_token(token: str) -> Optional[float]:
        t = token.strip()
        if not t:
            return None
        if t == "-":
            return None
        is_negative = t.startswith("(") and t.endswith(")")
        if is_negative:
            t = t[1:-1]
        t = t.replace(",", "")
        try:
            v = float(t)
        except ValueError:
            return None
        if is_negative:
            v = -v
        return v

    def _fetch_btc_etf_net_inflow_usd(self) -> float:
        try:
            req = Request(
                url="https://farside.co.uk/btc/",
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://farside.co.uk/",
                },
                method="GET",
            )
            with urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"抓取 BTC ETF 资金流失败（Farside Investors）：{exc}") from exc

        dates = list(re.finditer(r"\b\d{2}\s+[A-Za-z]{3}\s+\d{4}\b", html))
        if not dates:
            raise RuntimeError("抓取 BTC ETF 资金流失败：无法解析日期行。")

        for i in range(len(dates) - 1, -1, -1):
            start = dates[i].start()
            end = dates[i + 1].start() if i + 1 < len(dates) else len(html)
            row = html[start:end]

            boundary = re.search(r"\bTotal\b", row)
            if boundary:
                row = row[: boundary.start()]

            tokens = re.findall(r"\(\d[\d,]*\.\d+\)|\d[\d,]*\.\d+|\b-\b", row)
            if not tokens:
                continue
            last_token = tokens[-1].strip()
            if last_token == "-":
                continue
            total_usm = self._parse_usm_token(last_token)
            if total_usm is None:
                continue
            return float(total_usm) * 1_000_000.0

        raise RuntimeError("抓取 BTC ETF 资金流失败：未找到有效的最新交易日 Total 值。")

    @staticmethod
    def _parse_mock_etf_flow(value: str) -> Optional[float]:
        s = value.strip()
        if not s:
            return None
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)([mM])?$", s)
        if not m:
            return None
        v = float(m.group(1))
        if m.group(2):
            return v * 1_000_000.0
        if abs(v) <= 10000:
            return v * 1_000_000.0
        return v

    @staticmethod
    def _fetch_liquidation_heat() -> Tuple[float, float]:
        funding_rate = 0.0
        long_short_ratio = 1.0

        try:
            with urlopen(
                "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
                timeout=15,
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                funding_rate = float(payload.get("lastFundingRate", 0.0))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            funding_rate = 0.0

        try:
            with urlopen(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                "?symbol=BTCUSDT&period=5m&limit=1",
                timeout=15,
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, list) and payload:
                    long_short_ratio = float(payload[0].get("longShortRatio", 1.0))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            long_short_ratio = 1.0

        return funding_rate, long_short_ratio

    def build_market_snapshot(
        self,
        *,
        oil_price_change_pct: Optional[float] = None,
        dxy_change_pct: Optional[float] = None,
        tnx_yield_pct: Optional[float] = None,
        tnx_change_pct: Optional[float] = None,
        etf_net_inflow_usd: Optional[float] = None,
        price: Optional[float] = None,
        ma20_4h: Optional[float] = None,
        ma20_1d: Optional[float] = None,
        funding_rate: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
    ) -> MarketSnapshot:
        self.last_warnings = []

        px = price
        ma4 = ma20_4h
        ma1 = ma20_1d
        if px is None or ma4 is None or ma1 is None:
            try:
                px, ma4, ma1 = self._fetch_btc_realtime_and_ma()
            except Exception as exc:
                self.last_warnings.append(str(exc))
                px = self._safe_float(px, 0.0)
                ma4 = self._safe_float(ma4, px)
                ma1 = self._safe_float(ma1, px)

        oil = oil_price_change_pct
        dxy = dxy_change_pct
        if oil is None or dxy is None:
            try:
                oil, dxy = self._fetch_macro_changes()
            except Exception as exc:
                self.last_warnings.append(str(exc))
                oil = self._safe_float(oil, 0.0)
                dxy = self._safe_float(dxy, 0.0)

        tnx_yield = tnx_yield_pct
        tnx_change = tnx_change_pct
        if tnx_yield is None or tnx_change is None:
            try:
                tnx_yield, tnx_change = self._fetch_tnx()
            except Exception as exc:
                self.last_warnings.append(str(exc))
                tnx_yield = self._safe_float(tnx_yield, 0.0)
                tnx_change = self._safe_float(tnx_change, 0.0)

        etf_flow = etf_net_inflow_usd
        if etf_flow is None:
            try:
                etf_flow = self._fetch_btc_etf_net_inflow_usd()
            except Exception as exc:
                mock = self._parse_mock_etf_flow(os.getenv("MOCK_ETF_FLOW", ""))
                if mock is None:
                    self.last_warnings.append(str(exc))
                    etf_flow = 0.0
                else:
                    self.last_warnings.append("BTC ETF 资金流使用 MOCK_ETF_FLOW。")
                    etf_flow = mock

        fr = funding_rate
        lsr = long_short_ratio
        if fr is None or lsr is None:
            fr, lsr = self._fetch_liquidation_heat()

        return MarketSnapshot(
            oil_price_change_pct=self._safe_float(oil, 0.0),
            dxy_change_pct=self._safe_float(dxy, 0.0),
            tnx_yield_pct=self._safe_float(tnx_yield, 0.0),
            tnx_change_pct=self._safe_float(tnx_change, 0.0),
            etf_net_inflow_usd=self._safe_float(etf_flow, 0.0),
            price=self._safe_float(px, 0.0),
            ma20_4h=self._safe_float(ma4, 0.0),
            ma20_1d=self._safe_float(ma1, 0.0),
            funding_rate=self._safe_float(fr, 0.0),
            long_short_ratio=self._safe_float(lsr, 1.0),
        )

    def fetch_macro_quick(self) -> Tuple[float, float]:
        oil, dxy = self._fetch_macro_changes()
        return float(oil), float(dxy)

    @staticmethod
    def _score_oil(oil_price_change_pct: float) -> float:
        if oil_price_change_pct >= 2.0:
            return -1.5
        if oil_price_change_pct >= 1.0:
            return -1.0
        if oil_price_change_pct <= -1.0:
            return 0.6
        return 0.0

    @staticmethod
    def _score_dxy(dxy_change_pct: float) -> float:
        if dxy_change_pct >= 0.7:
            return -1.0
        if dxy_change_pct >= 0.3:
            return -0.6
        if dxy_change_pct <= -0.3:
            return 0.5
        return 0.0

    @staticmethod
    def _score_bonds(tnx_change_pct: float) -> float:
        if tnx_change_pct >= 1.5:
            return -1.0
        if tnx_change_pct <= -1.5:
            return 1.0
        return 0.0

    @staticmethod
    def _score_etf(etf_net_inflow_usd: float) -> float:
        if etf_net_inflow_usd >= 100_000_000.0:
            return 1.5
        if etf_net_inflow_usd <= -100_000_000.0:
            return -1.5
        return 0.0

    @staticmethod
    def _score_sentiment(funding_rate: float, long_short_ratio: float) -> float:
        score = 0.0
        if funding_rate >= 0.0008:
            score -= 1.2
        elif funding_rate >= 0.0005:
            score -= 0.8
        elif funding_rate <= -0.0005:
            score += 0.4

        if long_short_ratio >= 1.8:
            score -= 0.8
        elif long_short_ratio >= 1.4:
            score -= 0.4
        elif long_short_ratio <= 0.8:
            score += 0.2

        return max(-2.0, min(2.0, score))

    @staticmethod
    def _weight_macro_factor(change_pct: float) -> float:
        if abs(change_pct) > 2.5:
            return 1.2
        return 0.6

    @staticmethod
    def _weight_etf(etf_net_inflow_usd: float) -> float:
        amt = abs(etf_net_inflow_usd)
        if amt < 100_000_000.0:
            return 0.5
        if amt > 300_000_000.0:
            return 1.0
        return 0.8

    def analyze(self, ctx: DecisionContext) -> FinalDecision:
        geo_score_raw = float(ctx.geo_signal.score)
        alignment = self._price_alignment_adjust(
            geo_score=geo_score_raw,
            geo_signal=ctx.geo_signal,
        )
        geo_score = float(alignment["geo_score_adjusted"])
        oil_score = self._score_oil(ctx.market.oil_price_change_pct)
        dxy_score = self._score_dxy(ctx.market.dxy_change_pct)
        bond_score = self._score_bonds(ctx.market.tnx_change_pct)
        etf_score = self._score_etf(ctx.market.etf_net_inflow_usd)
        sentiment_score = self._score_sentiment(ctx.market.funding_rate, ctx.market.long_short_ratio)

        geo_w = 1.5
        oil_w = self._weight_macro_factor(ctx.market.oil_price_change_pct)
        dxy_w = self._weight_macro_factor(ctx.market.dxy_change_pct)
        bond_w = 0.6
        etf_w = self._weight_etf(ctx.market.etf_net_inflow_usd)
        sentiment_w = 0.4

        emergency_mode = abs(geo_score) > 1.8
        if emergency_mode:
            etf_w *= 0.5
            sentiment_w *= 0.5

        total_score = (
            (geo_score * geo_w)
            + (oil_score * oil_w)
            + (dxy_score * dxy_w)
            + (bond_score * bond_w)
            + (etf_score * etf_w)
            + (sentiment_score * sentiment_w)
        )

        is_below_ma20 = (ctx.market.price < ctx.market.ma20_4h) and (ctx.market.price < ctx.market.ma20_1d)
        if total_score <= -2.0 and is_below_ma20:
            signal: Literal["LONG", "SHORT", "NO_TRADE"] = "SHORT"
        else:
            signal = "LONG"

        if total_score >= 1.0:
            regime: Literal["RISK_ON", "RISK_OFF", "NEUTRAL"] = "RISK_ON"
        elif total_score <= -1.0:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        return FinalDecision(
            regime=regime,
            macro_score=float(total_score),
            signal=signal,
            factor_details={
                "geo_score_raw": geo_score_raw,
                "geo_score": geo_score,
                "oil_score": oil_score,
                "dxy_score": dxy_score,
                "bond_score": bond_score,
                "etf_score": etf_score,
                "sentiment_score": sentiment_score,
                "geo_weight": geo_w,
                "oil_weight": oil_w,
                "dxy_weight": dxy_w,
                "bond_weight": bond_w,
                "etf_weight": etf_w,
                "sentiment_weight": sentiment_w,
                "emergency_mode": emergency_mode,
                "total_score": total_score,
                "geo_price_aligned": alignment["applied"],
                "geo_price_move_pct_2h": alignment["move_pct_2h"],
                "geo_price_move_directional_pct_2h": alignment["directional_move_pct_2h"],
                "price": ctx.market.price,
                "ma20_4h": ctx.market.ma20_4h,
                "ma20_1d": ctx.market.ma20_1d,
                "oil_price_change_pct": ctx.market.oil_price_change_pct,
                "dxy_change_pct": ctx.market.dxy_change_pct,
                "tnx_yield_pct": ctx.market.tnx_yield_pct,
                "tnx_change_pct": ctx.market.tnx_change_pct,
                "etf_net_inflow_usd": ctx.market.etf_net_inflow_usd,
                "funding_rate": ctx.market.funding_rate,
                "long_short_ratio": ctx.market.long_short_ratio,
                "is_below_ma20": is_below_ma20,
                "geo_tier": ctx.geo_signal.tier,
                "geo_status": ctx.geo_signal.status,
            },
        )

    def _price_alignment_adjust(self, *, geo_score: float, geo_signal: GeoSignal) -> Dict[str, Union[float, bool]]:
        ts = geo_signal.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        current = datetime.now(timezone.utc)

        hours = (current - ts).total_seconds() / 3600.0
        if hours <= 0:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }
        if hours > 2.0:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }

        direction = -1.0 if geo_score < 0 else 1.0 if geo_score > 0 else 0.0
        if direction == 0.0:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }

        try:
            import requests

            url = "https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=5m&limit=48"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data") or []
        except Exception:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }

        closes: List[float] = []
        for r in rows:
            if isinstance(r, (list, tuple)) and len(r) >= 5:
                try:
                    closes.append(float(r[4]))
                except Exception:
                    continue
        if len(closes) < 2:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }
        windows = max(1, int(min(48, (hours * 60.0) / 5.0)))
        start = closes[-(windows + 1)] if len(closes) >= windows + 1 else closes[0]
        end = closes[-1]
        if start <= 0:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }
        move_pct = ((end - start) / start) * 100.0
        directional_move = direction * move_pct
        if directional_move >= 2.0:
            return {
                "geo_score_adjusted": geo_score * 0.6,
                "applied": True,
                "move_pct_2h": move_pct,
                "directional_move_pct_2h": directional_move,
            }
        return {
            "geo_score_adjusted": geo_score,
            "applied": False,
            "move_pct_2h": move_pct,
            "directional_move_pct_2h": directional_move,
        }

    @staticmethod
    def core_score_table(decision: FinalDecision) -> Dict[str, float]:
        d = decision.factor_details or {}
        keys = [
            "geo_score",
            "oil_score",
            "dxy_score",
            "bond_score",
            "etf_score",
            "sentiment_score",
            "total_score",
        ]
        out: Dict[str, float] = {}
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out
