"""
data_fetcher.py
Polygon.io API wrapper for the penny stock screener.
Free tier supports: snapshot data, previous close, ticker details, news.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


class PolygonFetcher:
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"apiKey": api_key}

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.BASE}{path}"
        try:
            r = self.session.get(url, params=params or {}, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "ERROR"}

    # ------------------------------------------------------------------ #
    # Screener core: get all gainers under $1                             #
    # ------------------------------------------------------------------ #
    def get_gainers_snapshot(self, min_price: float = 2.00, max_price: float = 20.00) -> list[dict]:
        """
        Fetch top gainers snapshot from Polygon.
        Filters to stocks within min_price and max_price (Ross Cameron default: $2-$20).
        Note: Free tier has a slight delay on snapshot data.
        """
        data = self._get("/v2/snapshot/locale/us/markets/stocks/gainers")
        tickers = data.get("tickers", [])
        results = []
        for t in tickers:
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            last = t.get("lastTrade", {})
            price = last.get("p") or day.get("c") or 0
            if price <= 0 or price < min_price or price > max_price:
                continue
            prev_close = prev.get("c") or 1
            pct_chg = ((price - prev_close) / prev_close * 100) if prev_close else 0
            vol_today = day.get("v") or 0
            vol_prev = prev.get("v") or 1
            results.append({
                "ticker": t.get("ticker", ""),
                "price": round(price, 4),
                "change_pct": round(pct_chg, 2),
                "volume_today": int(vol_today),
                "volume_prev": int(vol_prev),
                "surge_ratio": round(vol_today / vol_prev, 2) if vol_prev else 0,
                "vwap": round(day.get("vw") or price, 4),
                "open": round(day.get("o") or price, 4),
                "high": round(day.get("h") or price, 4),
                "low": round(day.get("l") or price, 4),
            })
        return results

    def get_ticker_details(self, ticker: str) -> dict:
        """Get float, market cap, exchange info for a ticker."""
        data = self._get(f"/v3/reference/tickers/{ticker}")
        res = data.get("results", {})
        return {
            "ticker": ticker,
            "name": res.get("name", ticker),
            "market_cap": res.get("market_cap"),
            "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
            "weighted_shares_outstanding": res.get("weighted_shares_outstanding"),
            "primary_exchange": res.get("primary_exchange", ""),
            "type": res.get("type", ""),
            "description": res.get("description", ""),
        }

    def get_previous_close(self, ticker: str) -> dict:
        """Get previous day OHLCV for a ticker."""
        data = self._get(f"/v2/aggs/ticker/{ticker}/prev")
        results = data.get("results", [])
        if not results:
            return {}
        r = results[0]
        return {
            "open": r.get("o"),
            "high": r.get("h"),
            "low": r.get("l"),
            "close": r.get("c"),
            "volume": r.get("v"),
            "vwap": r.get("vw"),
        }

    def get_agg_bars(self, ticker: str, days: int = 10) -> pd.DataFrame:
        """Get daily OHLCV bars for the last N days."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 5)).strftime("%Y-%m-%d")
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
            {"adjusted": "true", "sort": "asc", "limit": days + 5},
        )
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
        return df[["date", "open", "high", "low", "close", "volume", "vwap"]].tail(days)

    def get_news(self, ticker: str, limit: int = 5) -> list[dict]:
        """Get latest news for a ticker."""
        published_after = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get(
            "/v2/reference/news",
            {"ticker": ticker, "limit": limit, "published_utc.gte": published_after, "sort": "published_utc", "order": "desc"},
        )
        articles = data.get("results", [])
        return [
            {
                "title": a.get("title", ""),
                "published": a.get("published_utc", ""),
                "publisher": a.get("publisher", {}).get("name", ""),
                "url": a.get("article_url", ""),
                "summary": a.get("description", ""),
            }
            for a in articles
        ]

    def get_rsi(self, ticker: str, window: int = 14) -> float | None:
        """Calculate RSI from daily bars."""
        bars = self.get_agg_bars(ticker, days=window + 5)
        if bars.empty or len(bars) < window:
            return None
        closes = bars["close"].values
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def enrich_ticker(self, row: dict, fetch_rsi: bool = True) -> dict:
        """
        Add RSI, float, and news flag to a snapshot row.
        Makes 2-3 additional API calls per ticker — use sparingly on free tier.
        """
        ticker = row["ticker"]
        details = self.get_ticker_details(ticker)
        float_shares = (
            details.get("share_class_shares_outstanding")
            or details.get("weighted_shares_outstanding")
            or 0
        )
        float_m = round(float_shares / 1_000_000, 1) if float_shares else None
        news = self.get_news(ticker, limit=3)
        rsi = self.get_rsi(ticker) if fetch_rsi else None
        return {
            **row,
            "name": details.get("name", ticker),
            "exchange": details.get("primary_exchange", ""),
            "float_m": float_m,
            "has_news": len(news) > 0,
            "news_count": len(news),
            "news": news,
            "rsi": rsi,
            "description": details.get("description", ""),
        }

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
    ) -> pd.DataFrame:
        """
        Full screening pipeline.
        Returns a DataFrame of the top N stocks ranked by dollar volume.
        """
        # Step 1: gainers snapshot
        candidates = self.get_gainers_snapshot(min_price=min_price, max_price=max_price)

        # Step 2: volume filters (no extra API calls needed)
        filtered = []
        min_avg_vol = min_avg_vol_k * 1000
        for c in candidates:
            if c["volume_today"] < min_avg_vol:
                continue
            if c["surge_ratio"] < min_surge:
                continue
            if c["change_pct"] < min_chg_pct:
                continue
            filtered.append(c)

        # Rank by dollar volume before enrichment to limit API calls
        filtered.sort(key=lambda x: x["price"] * x["volume_today"], reverse=True)
        top_candidates = filtered[:min(top_n * 3, 30)]  # enrich up to 3× to allow for RSI/float drops

        # Step 3: enrich with RSI + float + news (rate-limited)
        enriched = []
        for i, c in enumerate(top_candidates):
            if on_progress:
                on_progress(i, len(top_candidates), c["ticker"])
            if enrich:
                time.sleep(0.25)  # respect free-tier rate limit (5 req/min on some endpoints)
                c = self.enrich_ticker(c, fetch_rsi=True)

            # Apply float / RSI / news filters after enrichment
            if enrich:
                if c.get("float_m") and c["float_m"] > max_float_m:
                    continue
                if c.get("rsi"):
                    if c["rsi"] < rsi_lo or c["rsi"] > rsi_hi:
                        continue
                if require_news and not c.get("has_news"):
                    continue

            enriched.append(c)
            if len(enriched) >= top_n:
                break

        if not enriched:
            return pd.DataFrame()

        df = pd.DataFrame(enriched)
        # Dollar volume score
        df["dollar_vol"] = df["price"] * df["volume_today"]
        df = df.sort_values("dollar_vol", ascending=False).head(top_n).reset_index(drop=True)
        df.index += 1
        return df
