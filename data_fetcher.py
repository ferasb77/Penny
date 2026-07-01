"""
data_fetcher.py
Polygon.io API wrapper for the penny stock screener.

Key behaviours:
- get_gainers_snapshot() falls back to historical data when live snapshot is empty.
- Historical fallback uses Polygon's grouped daily bars endpoint (one call, all tickers)
  instead of 48 individual calls — much faster and avoids free-tier rate limits.
- screen() returns a diagnostic dict so the UI can explain empty results.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


def _last_n_trading_days(n: int = 5) -> list[str]:
    """
    Return the last N weekdays as YYYY-MM-DD strings, most recent first.
    Does not account for US holidays but skips weekends.
    """
    days = []
    d = datetime.now()
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:   # Mon–Fri
            days.append(d.strftime("%Y-%m-%d"))
    return days


def _last_trading_day() -> str:
    return _last_n_trading_days(1)[0]


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
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status": "HTTP_ERROR", "status_code": r.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "ERROR"}

    # ── API key validation ─────────────────────────────────────────────────────
    def check_api_key(self) -> dict:
        data = self._get("/v2/aggs/ticker/AAPL/prev")
        if "error" in data:
            return {"valid": False, "error": data["error"]}
        if data.get("status") == "ERROR":
            return {"valid": False, "error": data.get("error", "Unknown error")}
        # resultsCount can be 0 on holidays — that's still a valid key
        return {"valid": True}

    # ── Live snapshot ──────────────────────────────────────────────────────────
    def get_gainers_snapshot(
        self,
        min_price: float = 2.00,
        max_price: float = 20.00,
    ) -> tuple[list[dict], str]:
        """
        Try the live gainers snapshot first.
        Falls back to grouped daily bars if snapshot is empty.
        Returns (results, source_label).
        """
        data = self._get("/v2/snapshot/locale/us/markets/stocks/gainers")
        tickers = data.get("tickers", [])
        results = self._parse_snapshot(tickers, min_price, max_price)

        if results:
            return results, "live"

        # Fallback to grouped daily bars (single API call)
        return self._get_grouped_daily_movers(min_price, max_price)

    def _parse_snapshot(
        self,
        tickers: list,
        min_price: float,
        max_price: float,
    ) -> list[dict]:
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

    def _get_grouped_daily_movers(
        self,
        min_price: float,
        max_price: float,
    ) -> tuple[list[dict], str]:
        """
        Use Polygon grouped daily bars to find small-cap momentum movers.
        Aggressively excludes large-cap stocks so results match the Ross Cameron
        universe: $2-$20, genuine % move, not institutional dollar volume.
        """
        trading_days = _last_n_trading_days(5)
        trade_date   = None
        day_bars     = []

        for candidate_date in trading_days:
            data = self._get(
                f"/v2/aggs/grouped/locale/us/market/stocks/{candidate_date}",
                {"adjusted": "true", "include_otc": "false"},
            )
            if data.get("resultsCount", 0) > 0:
                day_bars   = data.get("results", [])
                trade_date = candidate_date
                break

        if not day_bars:
            return [], f"historical:{trading_days[0]}"

        # Fetch previous day for % change calculation
        idx = trading_days.index(trade_date)
        prev_bars_map = {}
        if idx + 1 < len(trading_days):
            prev_date = trading_days[idx + 1]
            prev_data = self._get(
                f"/v2/aggs/grouped/locale/us/market/stocks/{prev_date}",
                {"adjusted": "true", "include_otc": "false"},
            )
            prev_bars_map = {r["T"]: r for r in prev_data.get("results", [])}

        results = []
        for bar in day_bars:
            ticker = bar.get("T", "")
            price  = bar.get("c", 0)
            vol    = bar.get("v", 0)
            vwap   = bar.get("vw") or price

            if not ticker or price <= 0:
                continue

            # Price range filter
            if price < min_price or price > max_price:
                continue

            # Minimum volume — must be tradeable
            if vol < 200_000:
                continue

            # Dollar volume cap: over $500M = large-cap institutional stock
            # A $15 stock with 40M shares is Dominion Energy, not a momentum play
            if price * vol > 500_000_000:
                continue

            # Skip warrants, rights, preferred: look for clean 1-5 char tickers
            if len(ticker) > 5 or any(c in ticker for c in [".", "/"]):
                continue

            # % change vs previous close
            prev       = prev_bars_map.get(ticker, {})
            prev_close = prev.get("c") or price
            pct_chg    = ((price - prev_close) / prev_close * 100) if prev_close else 0

            # Must have moved at least 2% to be relevant
            if abs(pct_chg) < 2.0:
                continue

            vol_prev = prev.get("v") or 1

            results.append({
                "ticker":       ticker,
                "price":        round(price, 4),
                "change_pct":   round(pct_chg, 2),
                "volume_today": int(vol),
                "volume_prev":  int(vol_prev),
                "surge_ratio":  round(vol / vol_prev, 2) if vol_prev else 1.0,
                "vwap":         round(vwap, 4),
                "open":         round(bar.get("o") or price, 4),
                "high":         round(bar.get("h") or price, 4),
                "low":          round(bar.get("l") or price, 4),
            })

        # Sort gainers first (like the live snapshot endpoint)
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
        closes   = bars["close"].values
        deltas   = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains    = [max(d, 0) for d in deltas]
        losses   = [abs(min(d, 0)) for d in deltas]
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def enrich_ticker(self, row: dict, fetch_rsi: bool = True) -> dict:
        ticker       = row["ticker"]
        details      = self.get_ticker_details(ticker)
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
        Full screening pipeline. Returns a dict:
          df             — DataFrame of results (may be empty)
          source         — 'live' or 'historical:YYYY-MM-DD'
          candidates     — count after price filter
          dropped        — filter funnel counts
          raw_candidates — all enriched tickers (even dropped ones)
          warning        — human-readable message if df is empty
        """
        # Step 1: get candidates
        candidates, source = self.get_gainers_snapshot(
            min_price=min_price, max_price=max_price
        )
        is_historical = source.startswith("historical")

        diagnostic = {
            "source":         source,
            "after_snapshot": len(candidates),
            "after_volume":   0,
            "after_surge":    0,
            "after_change":   0,
            "after_enrich":   0,
            "is_historical":  is_historical,
        }

        if not candidates:
            return {
                "df":             pd.DataFrame(),
                "source":         source,
                "candidates":     0,
                "dropped":        diagnostic,
                "raw_candidates": [],
                "warning": (
                    "Polygon returned no stocks in the selected price range. "
                    "This can happen on public holidays or if Polygon's free tier "
                    "has a data delay. Try again in a few minutes, or check "
                    "polygon.io/dashboard for service status."
                ),
            }

        # Step 2: basic filters
        min_avg_vol = min_avg_vol_k * 1000

        if is_historical:
            # Historical grouped bars: skip surge/volume (not intraday data)
            # Keep all sorted by absolute % change
            after_vol   = candidates
            after_surge = candidates
            after_chg   = sorted(
                candidates,
                key=lambda x: abs(x["change_pct"]),
                reverse=True,
            )
        else:
            after_vol   = [c for c in candidates if c["volume_today"] >= min_avg_vol]
            after_surge = [c for c in after_vol   if c["surge_ratio"]  >= min_surge]
            after_chg   = [c for c in after_surge if c["change_pct"]   >= min_chg_pct]

        diagnostic["after_volume"] = len(after_vol)
        diagnostic["after_surge"]  = len(after_surge)
        diagnostic["after_change"] = len(after_chg)

        if not after_chg:
            if not after_vol:
                msg = (
                    f"Volume filter ({min_avg_vol_k:,}K) removed all candidates. "
                    f"Try lowering it."
                )
            elif not after_surge:
                msg = (
                    f"Surge filter ({min_surge:.1f}×) removed all candidates. "
                    f"Try lowering it."
                )
            else:
                msg = (
                    f"% Change filter ({min_chg_pct:.0f}%) removed all candidates. "
                    f"Try lowering it."
                )
            return {
                "df":             pd.DataFrame(),
                "source":         source,
                "candidates":     len(candidates),
                "dropped":        diagnostic,
                "raw_candidates": [],
                "warning":        msg,
            }

        # Take top candidates by dollar volume for enrichment
        after_chg.sort(key=lambda x: x["price"] * x["volume_today"], reverse=True)
        top_candidates = after_chg[:min(top_n * 3, 30)]

        # Step 3: enrich + post-enrichment filters
        enriched     = []
        all_enriched = []

        for i, c in enumerate(top_candidates):
            if on_progress:
                on_progress(i, len(top_candidates), c["ticker"])
            if enrich:
                time.sleep(0.2)
                c = self.enrich_ticker(c, fetch_rsi=True)

            all_enriched.append(c)

            if enrich:
                if not is_historical:
                    if c.get("float_m") and c["float_m"] > max_float_m:
                        continue
                    if c.get("rsi") and (c["rsi"] < rsi_lo or c["rsi"] > rsi_hi):
                        continue
                    if require_news and not c.get("has_news"):
                        continue
                # Historical mode: no post-enrichment filters — show everything

            enriched.append(c)
            if len(enriched) >= top_n:
                break

        diagnostic["after_enrich"] = len(enriched)

        if not enriched:
            return {
                "df":             pd.DataFrame(),
                "source":         source,
                "candidates":     len(candidates),
                "dropped":        diagnostic,
                "raw_candidates": all_enriched,
                "warning": (
                    f"{len(after_chg)} candidates passed basic filters but all were "
                    f"dropped by float/RSI/news filters. "
                    f"Try raising max float ({max_float_m}M), widening RSI range, "
                    f"or turning off 'Require News Catalyst'."
                ),
            }

        df = pd.DataFrame(enriched)
        df["dollar_vol"] = df["price"] * df["volume_today"]
        df = df.sort_values("dollar_vol", ascending=False).head(top_n).reset_index(drop=True)
        df.index += 1

        return {
            "df":             df,
            "source":         source,
            "candidates":     len(candidates),
            "dropped":        diagnostic,
            "raw_candidates": all_enriched,
            "warning":        None,
        }
