from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
import re
from statistics import mean
from contextlib import redirect_stderr, redirect_stdout
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
    def _sma20_from_ohlcv(rows: List[List[float]]) -> float:
        closes = [float(r[4]) for r in rows if len(r) >= 5]
        if len(closes) < 20:
            raise RuntimeError("Binance OHLCV 数据不足，无法计算 MA20。")
        return float(mean(closes[-20:]))

    def _fetch_btc_realtime_and_ma(self) -> Tuple[float, float, float]:
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("未安装 ccxt，无法抓取 Binance 实时行情。请先执行: pip install ccxt") from exc

        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        try:
            ticker = exchange.fetch_ticker("BTC/USDT")
        except Exception as exc:
            raise RuntimeError(f"ccxt 抓取 Binance 现价失败：{exc}") from exc
        price = self._safe_float(ticker.get("last") or ticker.get("close"))
        if price <= 0:
            raise RuntimeError("ccxt 获取 BTC/USDT 实时价格失败。")

        try:
            rows_4h = exchange.fetch_ohlcv("BTC/USDT", timeframe="4h", limit=20)
            rows_1d = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=20)
        except Exception as exc:
            raise RuntimeError(f"ccxt 抓取 Binance K 线失败：{exc}") from exc
        ma20_4h = self._sma20_from_ohlcv(rows_4h)
        ma20_1d = self._sma20_from_ohlcv(rows_1d)
        return price, ma20_4h, ma20_1d

    @staticmethod
    def _yfinance_intraday_last_and_change(symbol: str) -> Tuple[float, float]:
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(symbol)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            daily = ticker.history(period="5d", interval="1d")
        if daily.empty:
            raise RuntimeError(f"yfinance 未返回 {symbol} 日线数据。")
        daily_close = daily["Close"].dropna()
        if daily_close.empty:
            raise RuntimeError(f"yfinance 返回 {symbol} 日线收盘价为空。")
        if len(daily_close) >= 2:
            prev_close = float(daily_close.iloc[-2])
        else:
            prev_close = float(daily_close.iloc[-1])

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            intraday = ticker.history(period="1d", interval="5m")
        if intraday.empty or intraday.get("Close") is None:
            last_price = float(daily_close.iloc[-1])
        else:
            intraday_close = intraday["Close"].dropna()
            if intraday_close.empty:
                last_price = float(daily_close.iloc[-1])
            else:
                last_price = float(intraday_close.iloc[-1])

        if prev_close == 0:
            return last_price, 0.0
        return last_price, ((last_price - prev_close) / prev_close) * 100.0

    @classmethod
    def _yfinance_intraday_change_pct(cls, symbol: str) -> float:
        _, change = cls._yfinance_intraday_last_and_change(symbol)
        return change

    def _fetch_macro_changes(self) -> Tuple[float, float]:
        try:
            oil_change = self._yfinance_intraday_change_pct("CL=F")
        except Exception as exc:
            raise RuntimeError(f"yfinance 抓取 CL=F 失败: {exc}") from exc

        dxy_symbols = ["DX-Y.NYB", "DX-Y"]
        last_error: Optional[Exception] = None
        for symbol in dxy_symbols:
            try:
                dxy_change = self._yfinance_intraday_change_pct(symbol)
                return oil_change, dxy_change
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"yfinance 抓取美元指数失败: {last_error}")

    def _fetch_tnx_from_stooq(self) -> Tuple[float, float]:
        try:
            with urlopen("https://stooq.com/q/d/l/?s=10yusy.b&i=d", timeout=20) as resp:
                csv_text = resp.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Stooq 抓取 10YUSY.B 失败: {exc}") from exc

        lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
        if len(lines) < 3:
            raise RuntimeError("Stooq 10YUSY.B 数据不足，无法计算涨跌幅。")

        header = lines[0].lower().split(",")
        try:
            close_idx = header.index("close")
        except ValueError:
            close_idx = 4

        def _close_from_line(line: str) -> float:
            parts = line.split(",")
            return float(parts[close_idx])

        last_close = _close_from_line(lines[-1])
        prev_close = _close_from_line(lines[-2])
        if prev_close == 0:
            return last_close, 0.0
        change_pct = ((last_close - prev_close) / prev_close) * 100.0
        return last_close, change_pct

    def _fetch_tnx(self) -> Tuple[float, float]:
        try:
            last, change = self._yfinance_intraday_last_and_change("^TNX")
            if last <= 0:
                raise RuntimeError("yfinance 返回 ^TNX 价格异常。")
            return last, change
        except Exception:
            try:
                last, change = self._fetch_tnx_from_stooq()
                self.last_warnings.append("10年期美债收益率：^TNX 获取失败，已切换到 Stooq(10YUSY.B)。")
                return last, change
            except Exception:
                try:
                    _, tlt_change = self._yfinance_intraday_last_and_change("TLT")
                    self.last_warnings.append("10年期美债收益率：已使用 TLT 反向波动作为代理。")
                    return 0.0, -tlt_change
                except Exception as exc:
                    raise RuntimeError(f"10年期美债收益率获取失败（^TNX/Stooq/TLT均不可用）: {exc}") from exc

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
            px, ma4, ma1 = self._fetch_btc_realtime_and_ma()

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
            import ccxt  # type: ignore
        except ImportError:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }

        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        try:
            rows = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=48)
        except Exception:
            return {
                "geo_score_adjusted": geo_score,
                "applied": False,
                "move_pct_2h": 0.0,
                "directional_move_pct_2h": 0.0,
            }

        closes = [float(r[4]) for r in rows if len(r) >= 5]
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
