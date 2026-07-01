"""
data_fetcher.py
Polygon.io API wrapper for the penny stock screener.
Free tier supports: snapshot data, previous close, ticker details, news, daily bars.

Key behaviours:
- get_gainers_snapshot() falls back to last-trading-day historical data when
  the live snapshot is empty (weekends, holidays, pre/post market).
- screen() returns a diagnostic dict if no results pass filters, so the UI
  can tell the user WHY nothing was returned instead of silently showing nothing.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


# Tickers used as fallback when snapshot is empty (well-known momentum names).
# Broad mix of small/mid caps across biotech, energy, tech, and retail —
# all tend to trade in the $2-$20 range. Real bar data is fetched for each;
# only the ticker list itself is hardcoded.
FALLBACK_WATCHLIST = [
    # Biotech / pharma — frequent news catalysts
    "ABIO", "PHIO", "BIOR", "AVGR", "CELZ", "TNXP", "ATOS", "OCGN",
    "SAVA", "VISL", "ATNF", "GFAI", "CLOV", "RDBX", "MOTS",
    # Energy / resources
    "NRGV", "TELL", "INDO", "MFIN", "GROM", "VERB",
    # Tech / software
    "DPRO", "INPX", "MMAT", "SGBX", "ILUS", "ABST", "COSM",
    # Retail / consumer
    "SNDL", "EXPR", "BBBY", "NKLA", "RIDE", "WKHS",
    # Recent high-volume momentum
    "FFIE", "MULN", "AGRX", "HCDI", "BNRG", "FTFT", "NAKD",
    "CENN", "PPSI", "TANH", "MOXC", "SMFL", "IMPP", "GTII",
]


def _last_trading_day() -> str:
    """Return the most recent weekday (skips Saturday/Sunday)."""
    d = datetime.now()
    if d.weekday() == 6:   # Sunday
        d -= timedelta(days=2)
    elif d.weekday() == 5: # Saturday
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


class PolygonFetcher:
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"apiKey": api_key}

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.BASE}{path}"
        try:
            r = self.session.get(url, params=params or {}, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "ERROR"}

    # ── API key check ──────────────────────────────────────────────────────────
    def check_api_key(self) -> dict:
        """Validate the API key by calling a lightweight endpoint."""
        data = self._get("/v2/aggs/ticker/AAPL/prev")
        if "error" in data:
            return {"valid": False, "error": data["error"]}
        if data.get("status") == "ERROR" or data.get("resultsCount", 0) == 0:
            return {"valid": False, "error": data.get("error", "No results returned")}
        return {"valid": True}

    # ── Live snapshot ──────────────────────────────────────────────────────────
    def get_gainers_snapshot(
        self,
        min_price: float = 2.00,
        max_price: float = 20.00,
    ) -> tuple[list[dict], str]:
        """
        Fetch top gainers snapshot from Polygon.
        Returns (results, source) where source is 'live' or 'historical:YYYY-MM-DD'.
        Falls back to last-trading-day historical data if snapshot is empty.
        """
        data = self._get("/v2/snapshot/locale/us/markets/stocks/gainers")
        tickers = data.get("tickers", [])

        results = self._parse_snapshot(tickers, min_price, max_price)

        if results:
            return results, "live"

        # ── Fallback: market closed or snapshot empty ──────────────────────────
        return self._get_historical_movers(min_price, max_price)

    def _parse_snapshot(
        self,
        tickers: list,
        min_price: float,
        max_price: float,
    ) -> list[dict]:
        """Parse raw snapshot ticker list into normalised dicts."""
        results = []
        for t in tickers:
            day  = t.get("day", {})
            prev = t.get("prevDay", {})
            last = t.get("lastTrade", {})
            price = last.get("p") or day.get("c") or 0
            if price <= 0 or price < min_price or price > max_price:
                continue
            prev_close = prev.get("c") or price
            pct_chg    = ((price - prev_close) / prev_close * 100) if prev_close else 0
            vol_today  = day.get("v") or 0
            vol_prev   = prev.get("v") or 1
            results.append({
                "ticker":       t.get("ticker", ""),
                "price":        round(price, 4),
                "change_pct":   round(pct_chg, 2),
                "volume_today": int(vol_today),
                "volume_prev":  int(vol_prev),
                "surge_ratio":  round(vol_today / vol_prev, 2) if vol_prev else 0,
                "vwap":         round(day.get("vw") or price, 4),
                "open":         round(day.get("o") or price, 4),
                "high":         round(day.get("h") or price, 4),
                "low":          round(day.get("l") or price, 4),
            })
        return results

    def _get_historical_movers(
        self,
        min_price: float,
        max_price: float,
    ) -> tuple[list[dict], str]:
        """
        Build a screener list from last-trading-day aggregate bars.
        Used when the live snapshot endpoint returns nothing (weekend/holiday).
        Fetches the watchlist tickers, filters by price, computes % change
        vs the prior bar, and returns with a 'historical' source label.
        """
        trade_date = _last_trading_day()
        prior_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")

        results = []
        for ticker in FALLBACK_WATCHLIST:
            try:
                data = self._get(
                    f"/v2/aggs/ticker/{ticker}/range/1/day/{prior_date}/{trade_date}",
                    {"adjusted": "true", "sort": "asc", "limit": 10},
                )
                bars = data.get("results", [])
                if len(bars) < 2:
                    continue

                today  = bars[-1]
                prev   = bars[-2]
                price  = today.get("c", 0)
                if price <= 0 or price < min_price or price > max_price:
                    continue

                prev_close = prev.get("c") or price
                pct_chg    = ((price - prev_close) / prev_close * 100) if prev_close else 0
                vol_today  = today.get("v") or 0
                vol_prev   = prev.get("v") or 1

                results.append({
                    "ticker":       ticker,
                    "price":        round(price, 4),
                    "change_pct":   round(pct_chg, 2),
                    "volume_today": int(vol_today),
                    "volume_prev":  int(vol_prev),
                    "surge_ratio":  round(vol_today / vol_prev, 2) if vol_prev else 0,
                    "vwap":         round(today.get("vw") or price, 4),
                    "open":         round(today.get("o") or price, 4),
                    "high":         round(today.get("h") or price, 4),
                    "low":          round(today.get("l") or price, 4),
                })
                time.sleep(0.1)   # stay within free-tier rate limits
            except Exception:
                continue

        # Sort by % change descending to mimic gainers order
        results.sort(key=lambda x: x["change_pct"], reverse=True)
        return results, f"historical:{trade_date}"

    # ── Ticker details ─────────────────────────────────────────────────────────
    def get_ticker_details(self, ticker: str) -> dict:
        data = self._get(f"/v3/reference/tickers/{ticker}")
        res  = data.get("results", {})
        return {
            "ticker":                         ticker,
            "name":                           res.get("name", ticker),
            "market_cap":                     res.get("market_cap"),
            "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
            "weighted_shares_outstanding":    res.get("weighted_shares_outstanding"),
            "primary_exchange":               res.get("primary_exchange", ""),
            "type":                           res.get("type", ""),
            "description":                    res.get("description", ""),
        }

    def get_previous_close(self, ticker: str) -> dict:
        data    = self._get(f"/v2/aggs/ticker/{ticker}/prev")
        results = data.get("results", [])
        if not results:
            return {}
        r = results[0]
        return {
            "open": r.get("o"), "high": r.get("h"),
            "low":  r.get("l"), "close": r.get("c"),
            "volume": r.get("v"), "vwap": r.get("vw"),
        }

    def get_agg_bars(self, ticker: str, days: int = 10) -> pd.DataFrame:
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
        data  = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": days + 10},
        )
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap",
        })
        return df[["date", "open", "high", "low", "close", "volume", "vwap"]].tail(days)

    def get_news(self, ticker: str, limit: int = 5) -> list[dict]:
        published_after = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get(
            "/v2/reference/news",
            {
                "ticker": ticker, "limit": limit,
                "published_utc.gte": published_after,
                "sort": "published_utc", "order": "desc",
            },
        )
        return [
            {
                "title":     a.get("title", ""),
                "published": a.get("published_utc", ""),
                "publisher": a.get("publisher", {}).get("name", ""),
                "url":       a.get("article_url", ""),
                "summary":   a.get("description", ""),
            }
            for a in data.get("results", [])
        ]

    def get_rsi(self, ticker: str, window: int = 14) -> float | None:
        bars = self.get_agg_bars(ticker, days=window + 5)
        if bars.empty or len(bars) < window:
            return None
        closes  = bars["close"].values
        deltas  = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains   = [max(d, 0) for d in deltas]
        losses  = [abs(min(d, 0)) for d in deltas]
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def enrich_ticker(self, row: dict, fetch_rsi: bool = True) -> dict:
        ticker      = row["ticker"]
        details     = self.get_ticker_details(ticker)
        float_shares = (
            details.get("share_class_shares_outstanding")
            or details.get("weighted_shares_outstanding")
            or 0
        )
        float_m = round(float_shares / 1_000_000, 1) if float_shares else None
        news    = self.get_news(ticker, limit=3)
        rsi     = self.get_rsi(ticker) if fetch_rsi else None
        return {
            **row,
            "name":        details.get("name", ticker),
            "exchange":    details.get("primary_exchange", ""),
            "float_m":     float_m,
            "has_news":    len(news) > 0,
            "news_count":  len(news),
            "news":        news,
            "rsi":         rsi,
            "description": details.get("description", ""),
        }

    # ── Full screening pipeline ────────────────────────────────────────────────
    def screen(
        self,
        min_price: float = 2.00,
        max_price: float = 20.00,
        min_avg_vol_k: int = 500,
        min_surge: float = 3.0,
        min_chg_pct: float = 5.0,
        max_float_m: float = 20.0,
        rsi_lo: int = 40,
        rsi_hi: int = 70,
        require_news: bool = False,
        top_n: int = 10,
        enrich: bool = True,
        on_progress=None,
    ) -> dict:
        """
        Full screening pipeline.
        Returns a dict with keys:
          df         — DataFrame of results (may be empty)
          source     — 'live' or 'historical:YYYY-MM-DD'
          candidates — how many passed the snapshot/price filter
          dropped    — dict explaining how many were dropped at each filter stage
          warning    — human-readable explanation if df is empty
        """
        # ── Step 1: snapshot (live or historical fallback) ─────────────────────
        candidates, source = self.get_gainers_snapshot(
            min_price=min_price, max_price=max_price
        )

        diagnostic = {
            "source":         source,
            "after_snapshot": len(candidates),
            "after_volume":   0,
            "after_surge":    0,
            "after_change":   0,
            "after_enrich":   0,
            "is_historical":  source.startswith("historical"),
        }

        if not candidates:
            return {
                "df":         pd.DataFrame(),
                "source":     source,
                "candidates": 0,
                "dropped":    diagnostic,
                "warning": (
                    "No stocks found in the $2–$20 price range from Polygon. "
                    "This usually means the API key is invalid or the market data "
                    "is unavailable. Check your Polygon API key in the sidebar."
                ),
            }

        # ── Step 2: basic filters (no extra API calls) ─────────────────────────
        is_historical = source.startswith("historical")

        min_avg_vol = min_avg_vol_k * 1000

        # Historical daily bars: volume and surge filters are not meaningful
        # (day-over-day volume change is very different from intraday surge).
        # Relax them automatically so we always get results for practice.
        if is_historical:
            after_vol   = candidates          # skip volume floor
            after_surge = candidates          # skip surge (not intraday)
            # Keep all stocks — sort by absolute % change so best movers surface first.
            # A flat or slightly red stock is still useful for practice analysis.
            after_chg   = sorted(candidates, key=lambda x: abs(x["change_pct"]), reverse=True)
        else:
            after_vol   = [c for c in candidates if c["volume_today"] >= min_avg_vol]
            after_surge = [c for c in after_vol   if c["surge_ratio"]  >= min_surge]
            after_chg   = [c for c in after_surge if c["change_pct"]   >= min_chg_pct]

        diagnostic["after_volume"] = len(after_vol)
        diagnostic["after_surge"]  = len(after_surge)
        diagnostic["after_change"] = len(after_chg)

        # If every candidate was dropped, explain which filter was too tight
        if not after_chg:
            tightest = ""
            if not after_vol:
                tightest = (
                    f"Min avg volume filter ({min_avg_vol_k:,}K) is too high — "
                    f"all {len(candidates)} candidates were below it. "
                    f"Try lowering to 100K or 0."
                )
            elif not after_surge:
                tightest = (
                    f"Volume surge filter ({min_surge:.1f}×) dropped all remaining candidates. "
                    f"Try lowering to 1.5× or 1×."
                )
            else:
                tightest = (
                    f"% change filter ({min_chg_pct:.0f}%) dropped all remaining candidates. "
                    f"Try lowering to 1% or 0%."
                )
            return {
                "df":         pd.DataFrame(),
                "source":     source,
                "candidates": len(candidates),
                "dropped":    diagnostic,
                "warning":    tightest,
            }

        # Rank by dollar volume before enrichment
        after_chg.sort(key=lambda x: x["price"] * x["volume_today"], reverse=True)
        top_candidates = after_chg[:min(top_n * 3, 30)]

        # ── Step 3: enrich + post-enrichment filters ───────────────────────────
        enriched = []
        for i, c in enumerate(top_candidates):
            if on_progress:
                on_progress(i, len(top_candidates), c["ticker"])
            if enrich:
                time.sleep(0.2)
                c = self.enrich_ticker(c, fetch_rsi=True)

            if enrich:
                if not is_historical:
                    # Live mode: enforce all post-enrichment filters strictly
                    if c.get("float_m") and c["float_m"] > max_float_m:
                        continue
                    if c.get("rsi") and (c["rsi"] < rsi_lo or c["rsi"] > rsi_hi):
                        continue
                    if require_news and not c.get("has_news"):
                        continue
                else:
                    # Historical mode: only hard-fail on float — RSI and news
                    # are less meaningful on weekend/closed-market data
                    if c.get("float_m") and c["float_m"] > max_float_m * 3:
                        continue

            enriched.append(c)
            if len(enriched) >= top_n:
                break

        diagnostic["after_enrich"] = len(enriched)

        if not enriched:
            return {
                "df":         pd.DataFrame(),
                "source":     source,
                "candidates": len(candidates),
                "dropped":    diagnostic,
                "warning": (
                    f"{len(after_chg)} candidates passed the basic filters but all were "
                    f"dropped by float / RSI / news filters. "
                    f"Try raising max float ({max_float_m}M), widening RSI range "
                    f"({rsi_lo}–{rsi_hi}), or turning off 'Require News Catalyst'."
                ),
            }

        df = pd.DataFrame(enriched)
        df["dollar_vol"] = df["price"] * df["volume_today"]
        df = df.sort_values("dollar_vol", ascending=False).head(top_n).reset_index(drop=True)
        df.index += 1

        return {
            "df":         df,
            "source":     source,
            "candidates": len(candidates),
            "dropped":    diagnostic,
            "warning":    None,
        }
