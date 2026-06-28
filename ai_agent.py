"""
ai_agent.py
Claude-powered AI agent implementing Ross Cameron's (Warrior Trading) day trading methodology.
Based on: 5-criteria stock selection, bull flag entries, 2:1 P/L ratio, position sizing discipline,
hot/cold market detection, and the profit-cushion-first scaling system.
"""

import anthropic
import json
import pandas as pd
from data_fetcher import PolygonFetcher


SYSTEM_PROMPT = """You are an elite day trading coach and research assistant built on the methodology of Ross Cameron (Warrior Trading). You have analyzed over $12.5 million in verified trading profits and 10+ years of daily trade data. You apply this framework rigorously to every stock you evaluate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROSS CAMERON'S CORE FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## THE 5 CRITERIA FOR STOCK SELECTION (all 5 must be met for an A-quality setup)
1. UP AT LEAST 10% (minimum 2% pre-market) — the stock must already be moving
2. RELATIVE VOLUME ≥ 5× its 50-day average — volume confirms real interest, not noise
3. NEWS CATALYST — earnings, FDA approval, partnership, clinical trial results; volume without news is a red flag (likely pump-and-dump)
4. PRICE BETWEEN $2–$20 — big percentage swings are possible; penny stocks under $1 are ultra-high risk
5. FLOAT UNDER 10 MILLION SHARES — low supply + high demand = explosive moves; lower is better

Grade every setup: A (all 5), B (4/5), C (3/5). Only trade A and B setups. C setups = unnecessary risk.

## THE BULL FLAG ENTRY PATTERN
The only pattern to look for on A/B-quality stocks:
- A strong multi-candle GREEN SQUEEZE UP (3–7 candles, increasing volume)
- A PULLBACK on lighter volume (this is the flag — sellers taking profit)
- The flag must hold at least 50% of the initial move (if it drops more, skip the trade)
- Entry: the FIRST CANDLE TO MAKE A NEW HIGH after the pullback
- Buy the moment it ticks ONE PENNY above the prior candle's high
- Stop loss: the LOW OF THE PULLBACK (never move this lower)
- Profit target: retest of the HIGH OF DAY (or prior resistance)

First pullbacks are strongest. Second pullbacks are acceptable. Third pullbacks — be cautious. Never chase.

## RISK MANAGEMENT — THE 2:1 PROFIT-TO-LOSS RULE
- NEVER take a trade where you cannot make at least 2× what you're risking
- If entry is $3.06, stop is $2.96 (10¢ risk) → profit target must be at least $3.26 (20¢)
- Before every setup ask: "Does the high of day give me 2:1?" If the target is too close to resistance, skip it
- Math truth: with 2:1 P/L ratio, you only need to be right 34% of the time to break even
- With 70% accuracy + 2:1 P/L = the formula for consistent profitability

## THE PROFIT CUSHION SYSTEM (secret to consistency — 76 consecutive green days)
This is HOW to size positions, not just how much:

PHASE 1 — START SMALL (¼ size):
- Begin every day at ONE QUARTER of your normal share size
- Stay at ¼ size until you have built a profit cushion = 25% of your daily goal
- If your daily goal is $1,000 → you need $250 in profit before sizing up
- If you never reach that cushion, stay at ¼ size ALL DAY

PHASE 2 — SIZE UP (full size):
- Once the cushion is built, increase to your full normal share size
- On hot market days, you can add to winning positions (see below)
- If you give back the cushion → immediately drop back to ¼ size or stop

PHASE 3 — ADD TO WINNERS (not losers):
- When a trade is working and you're up, DOUBLE the position
- Move stop to BREAKEVEN once you add (worst case: flat, not a loss)
- Never add to a losing position — cut losers ruthlessly
- This is what produces $6,000 trades vs $2,000 trades on the same setup

DAILY MAX LOSS = your daily profit goal (e.g., goal $1,000 → max loss $1,000)
If no trade in 30 minutes → call it, walk away. Don't force trades.

## HOT vs COLD MARKET DETECTION
Before sizing up, assess market temperature:
HOT MARKET signals: multiple A-quality setups appearing, volume is surging, first pullbacks resolving cleanly, stocks holding VWAP, news flow is active
COLD MARKET signals: stocks fading after initial squeeze, few setups meet all 5 criteria, VWAP not holding, spreads are wide, volume drying up

In a cold market: stay at ¼ size, reduce quality threshold requirements, walk away early
In a hot market: size up aggressively, add to winners, trade longer

## ACCURACY + P/L RATIO RELATIONSHIP
These numbers must both be strong:
- Accuracy 70%+ with 2:1 P/L → highly profitable
- Accuracy 50% with 2:1 P/L → breakeven (still viable)
- Accuracy 50% with 1:1 P/L → losing money
- Accuracy 46% (red day average) → stop trading immediately, it's a cold day

When accuracy is dropping during a session → it's a signal the market has gone cold → reduce size, tighten quality standards, or stop

## POSITION SIZING SCALING PATH (for new traders)
Start: 10–20 shares (prove the concept, make mistakes cheaply)
Week 1-2: 10 shares → 20 → 30 → 40 → 50
Month 1: Build to 100–160 shares
After 1,000 profitable trades at small size: step up to 500–1,000 shares
After 2,000 trades: step up to 1,500–5,000 shares
Important: you CANNOT scale a strategy up indefinitely (liquidity ceiling exists); you CAN always scale it down

## THE NEGATIVE FEEDBACK LOOP (avoid at all costs)
Loss → emotional pain → revenge trading → lower quality trades → bigger losses → emotional spiral
BREAK IT BY: accepting the small loss, resetting to ¼ size, tightening your quality filter back to A-only

## EXIT INDICATORS
- Primary exit: retest of high of day (profit target)
- Secondary exit: if a candle shows a long upper wick (dragonfly doji type) → buyers and sellers in conflict → consider reducing position
- Time-based exit: if the move hasn't resolved in the expected direction within 2–3 minutes → reevaluate
- Stop hit: exit immediately, no hoping, no averaging down
- Average winner hold time: ~3 minutes. Average loser: ~2 minutes. Get in, get out.

## WHAT TO AVOID
- Trading stocks that don't meet 5 criteria (Ford Motor Company going sideways = no opportunity)
- Holding losers and hoping — this kills accounts
- Trading during emotional states (frustration, FOMO, greed, revenge)
- Adding to losing positions to reduce cost basis
- Trading C-quality setups in a cold market
- Overstaying your welcome (if you've made your goal, protect it)
- Stocks under $1 (allowed in this screener for educational purposes but EXTREME risk: manipulation, pump-and-dump, wide spreads, halts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ROLE AS ANALYST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For every stock you analyze:
1. GRADE IT: Score it A/B/C against the 5 criteria. State which criteria it meets and which it fails.
2. IDENTIFY THE PATTERN: Is there a bull flag forming? Where is the entry trigger? Where is the stop? Where is the target? Does the distance give 2:1?
3. ASSESS MARKET TEMP: Based on the screener data, is today hot or cold? What does that mean for sizing?
4. FLAG THE RISKS: Pump-and-dump signals, no catalyst, too wide a spread, float too large, extended too far from entry
5. POSITION SIZING CONTEXT: Based on the user's implied experience level, suggest appropriate share size (remind beginners to start at ¼ size)
6. GIVE A VERDICT: A-setup worth watching, B-setup with caveats, or C-setup to skip

Be direct and fast — traders need answers in seconds, not paragraphs. Lead with the grade and the key numbers. Explain second.

CRITICAL DISCLAIMER: You are NOT a licensed financial advisor. This analysis is for educational and informational purposes only. Day trading is extremely risky and most retail traders lose money. Always paper trade first. Never risk money you cannot afford to lose. The stocks on this screener (under $1) carry extreme additional risks including manipulation, halts, and illiquidity.
"""

TOOLS = [
    {
        "name": "get_stock_details",
        "description": "Fetch full details for a specific ticker: price, volume surge ratio, RSI, float, news catalyst, and VWAP. Used to grade a setup against the 5 criteria and identify bull flag entry levels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. ABIO"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_news",
        "description": "Fetch latest news/catalyst for a ticker from the last 48 hours. News catalyst is criteria #3 — a stock moving without news is likely a pump-and-dump. Use this to verify the catalyst before grading a setup.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_tickers",
        "description": "Compare two or more tickers side by side on all 5 criteria: % change, relative volume, news flag, price range fit, float size. Use to rank setups and identify which is the A-quality vs B/C.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to compare"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "get_price_bars",
        "description": "Get recent OHLCV bars to identify bull flag patterns: the squeeze up, the pullback, and the entry trigger candle. Also used to check if the stock is holding 50% of its initial move (key bull flag validity test) and to calculate the 2:1 profit-to-loss ratio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {
                    "type": "integer",
                    "description": "Number of days of bars (use 1-2 for intraday pattern analysis, up to 10 for trend context)",
                    "default": 5
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "assess_market_temperature",
        "description": "Assess whether today is a HOT or COLD market based on screener results. Hot = size up, trade longer. Cold = stay at quarter size, walk away early. Analyzes the current screener data to give a market temperature verdict and recommended daily approach.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tickers currently on the screener to analyze as a group"
                }
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "calculate_trade_levels",
        "description": "Calculate entry, stop loss, and profit target for a bull flag setup. Verifies the 2:1 profit-to-loss ratio. Input the current price, pullback low (stop), and high of day (target). Returns whether the setup meets the 2:1 minimum and the exact share size risk for common position sizes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "entry_price": {"type": "number", "description": "Your planned entry price (first candle to make new high)"},
                "stop_price": {"type": "number", "description": "Stop loss = low of the pullback"},
                "target_price": {"type": "number", "description": "Profit target = high of day or prior resistance"},
                "share_sizes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Share sizes to calculate risk/reward for (e.g. [100, 500, 1000, 5000])",
                    "default": [100, 500, 1000, 5000]
                }
            },
            "required": ["ticker", "entry_price", "stop_price", "target_price"]
        }
    }
]


class TradingAgent:
    def __init__(self, anthropic_api_key: str, polygon_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.fetcher = PolygonFetcher(api_key=polygon_api_key)
        self.conversation: list[dict] = []

    def _score_criteria(self, stock: dict) -> dict:
        """Score a stock against Ross Cameron's 5 criteria. Returns grades and details."""
        criteria = {}

        # 1. % Change (need 10%+ ideally, 2%+ minimum)
        chg = stock.get("change_pct", 0)
        criteria["pct_change"] = {
            "value": chg,
            "pass": chg >= 10,
            "partial": 2 <= chg < 10,
            "label": f"+{chg:.1f}%",
            "note": "A-grade needs 10%+" if chg < 10 else "✓ Strong move"
        }

        # 2. Relative volume (need 5×+)
        surge = stock.get("surge_ratio", 0)
        criteria["relative_volume"] = {
            "value": surge,
            "pass": surge >= 5,
            "partial": 2 <= surge < 5,
            "label": f"{surge:.1f}× avg volume",
            "note": "Need 5× minimum" if surge < 5 else "✓ High relative volume"
        }

        # 3. News catalyst
        has_news = stock.get("has_news", False)
        criteria["news_catalyst"] = {
            "value": has_news,
            "pass": has_news,
            "partial": False,
            "label": "Has catalyst" if has_news else "NO NEWS FOUND",
            "note": "✓ Catalyst confirmed" if has_news else "⚠ No catalyst = pump-and-dump risk"
        }

        # 4. Price range $2–$20 (these are sub-$1 stocks so this will usually fail)
        price = stock.get("price", 0)
        in_range = 2.0 <= price <= 20.0
        sub_dollar = price < 1.0
        criteria["price_range"] = {
            "value": price,
            "pass": in_range,
            "partial": False,
            "label": f"${price:.3f}",
            "note": "⚠ Sub-$1: extreme risk (Ross trades $2–$20)" if sub_dollar else
                    "✓ In ideal range" if in_range else "Outside $2–$20 range"
        }

        # 5. Float under 10M shares
        float_m = stock.get("float_m")
        if float_m is not None:
            criteria["float"] = {
                "value": float_m,
                "pass": float_m <= 10,
                "partial": 10 < float_m <= 20,
                "label": f"{float_m}M shares",
                "note": "✓ Low float" if float_m <= 10 else
                        "Acceptable but higher" if float_m <= 20 else "Too large for big moves"
            }
        else:
            criteria["float"] = {
                "value": None,
                "pass": False,
                "partial": True,
                "label": "Unknown",
                "note": "Float data unavailable"
            }

        # Grade: A = 5/5, B = 4/5, C = 3/5
        passes = sum(1 for c in criteria.values() if c["pass"])
        partials = sum(1 for c in criteria.values() if c.get("partial"))
        effective = passes + (partials * 0.5)

        if passes >= 5:
            grade = "A"
        elif passes >= 4 or effective >= 4:
            grade = "B"
        elif passes >= 3 or effective >= 3:
            grade = "C"
        else:
            grade = "D"

        return {"grade": grade, "passes": passes, "criteria": criteria}

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a JSON string."""
        try:
            if tool_name == "get_stock_details":
                ticker = tool_input["ticker"].upper()
                snapshot = self.fetcher.get_gainers_snapshot(max_price=999)
                row = next((s for s in snapshot if s["ticker"] == ticker), {"ticker": ticker, "price": None})
                enriched = self.fetcher.enrich_ticker(row, fetch_rsi=True)
                # Add 5-criteria scoring
                score = self._score_criteria(enriched)
                enriched["setup_grade"] = score["grade"]
                enriched["criteria_passes"] = score["passes"]
                enriched["criteria_detail"] = score["criteria"]
                return json.dumps(enriched, default=str)

            elif tool_name == "get_stock_news":
                ticker = tool_input["ticker"].upper()
                news = self.fetcher.get_news(ticker, limit=5)
                catalyst_found = len(news) > 0
                catalyst_types = []
                for article in news:
                    title = article.get("title", "").lower()
                    if any(w in title for w in ["fda", "approval", "cleared", "approved"]):
                        catalyst_types.append("FDA/Regulatory")
                    elif any(w in title for w in ["earnings", "revenue", "profit", "quarter"]):
                        catalyst_types.append("Earnings")
                    elif any(w in title for w in ["trial", "clinical", "phase", "data"]):
                        catalyst_types.append("Clinical/Trial")
                    elif any(w in title for w in ["partnership", "agreement", "contract", "deal"]):
                        catalyst_types.append("Partnership/Deal")
                    elif any(w in title for w in ["merger", "acquisition", "buyout"]):
                        catalyst_types.append("M&A")
                    else:
                        catalyst_types.append("General News")
                return json.dumps({
                    "ticker": ticker,
                    "catalyst_found": catalyst_found,
                    "catalyst_types": list(set(catalyst_types)),
                    "ross_verdict": "CRITERIA #3 MET — valid catalyst" if catalyst_found else
                                   "CRITERIA #3 FAILED — no news = likely pump-and-dump, avoid",
                    "articles": news
                }, default=str)

            elif tool_name == "compare_tickers":
                tickers = [t.upper() for t in tool_input["tickers"]]
                snapshot = self.fetcher.get_gainers_snapshot(max_price=999)
                snap_map = {s["ticker"]: s for s in snapshot}
                results = []
                for t in tickers:
                    row = snap_map.get(t, {"ticker": t, "price": 0, "change_pct": 0,
                                           "surge_ratio": 0, "volume_today": 0})
                    rsi = self.fetcher.get_rsi(t)
                    details = self.fetcher.get_ticker_details(t)
                    news = self.fetcher.get_news(t, limit=2)
                    float_m = None
                    fo = details.get("share_class_shares_outstanding") or details.get("weighted_shares_outstanding")
                    if fo:
                        float_m = round(fo / 1_000_000, 1)
                    enriched = {**row, "rsi": rsi, "float_m": float_m,
                                "has_news": len(news) > 0, "name": details.get("name", t)}
                    score = self._score_criteria(enriched)
                    enriched["setup_grade"] = score["grade"]
                    enriched["criteria_passes"] = score["passes"]
                    results.append(enriched)

                # Sort by grade then dollar volume
                grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
                results.sort(key=lambda x: (grade_order.get(x["setup_grade"], 4),
                                            -(x.get("price", 0) * x.get("volume_today", 0))))
                return json.dumps({
                    "comparison": results,
                    "top_pick": results[0]["ticker"] if results else None,
                    "ranking_note": "Ranked by setup grade (A→D), then dollar volume"
                }, default=str)

            elif tool_name == "get_price_bars":
                ticker = tool_input["ticker"].upper()
                days = min(int(tool_input.get("days", 5)), 30)
                df = self.fetcher.get_agg_bars(ticker, days=days)
                if df.empty:
                    return json.dumps({"error": "No price data available"})

                bars = df.to_dict(orient="records")

                # Detect bull flag pattern in the bars
                pattern_notes = []
                if len(bars) >= 3:
                    closes = [b["close"] for b in bars]
                    highs = [b["high"] for b in bars]
                    lows = [b["low"] for b in bars]
                    volumes = [b["volume"] for b in bars]

                    recent_high = max(highs[-5:]) if len(highs) >= 5 else max(highs)
                    recent_low = min(lows[-3:]) if len(lows) >= 3 else min(lows)
                    initial_move = recent_high - closes[-5] if len(closes) >= 5 else 0
                    pullback = recent_high - recent_low
                    pullback_pct = (pullback / recent_high * 100) if recent_high > 0 else 0

                    if pullback_pct <= 50:
                        pattern_notes.append(f"Holding >{100-pullback_pct:.0f}% of move — flag structure valid")
                    else:
                        pattern_notes.append(f"⚠ Pulled back {pullback_pct:.0f}% — exceeds 50% retrace, flag may be broken")

                    if len(volumes) >= 2 and volumes[-1] < volumes[-2]:
                        pattern_notes.append("Volume lighter on pullback — healthy flag formation")
                    else:
                        pattern_notes.append("Volume not declining on pullback — less clean flag")

                return json.dumps({
                    "ticker": ticker,
                    "bars": bars,
                    "pattern_analysis": pattern_notes,
                    "key_levels": {
                        "high_of_recent_bars": max(b["high"] for b in bars[-5:]) if len(bars) >= 5 else None,
                        "low_of_recent_bars": min(b["low"] for b in bars[-3:]) if len(bars) >= 3 else None,
                        "vwap_last": bars[-1].get("vwap") if bars else None
                    }
                }, default=str)

            elif tool_name == "assess_market_temperature":
                tickers = [t.upper() for t in tool_input["tickers"]]
                snapshot = self.fetcher.get_gainers_snapshot(max_price=999)
                snap_map = {s["ticker"]: s for s in snapshot}

                scores = []
                for t in tickers[:5]:  # limit API calls
                    row = snap_map.get(t, {"ticker": t})
                    score = self._score_criteria(row)
                    scores.append({"ticker": t, "grade": score["grade"], "passes": score["passes"]})

                a_count = sum(1 for s in scores if s["grade"] == "A")
                b_count = sum(1 for s in scores if s["grade"] == "B")
                avg_surge = sum(
                    snap_map.get(t, {}).get("surge_ratio", 0) for t in tickers
                ) / max(len(tickers), 1)
                avg_chg = sum(
                    snap_map.get(t, {}).get("change_pct", 0) for t in tickers
                ) / max(len(tickers), 1)

                if a_count >= 2 and avg_surge >= 5:
                    temp = "HOT"
                    advice = ("Multiple A-quality setups present. Size up to FULL position after building your 25% cushion. "
                              "Add to winners. Trade longer. This is the day to maximize.")
                elif a_count >= 1 or b_count >= 2:
                    temp = "WARM"
                    advice = ("Some quality setups but not a roaring market. Build your cushion first with ¼ size. "
                              "Size up selectively. Focus only on A/B setups. Don't force trades.")
                else:
                    temp = "COLD"
                    advice = ("Few quality setups. Stay at ¼ size ALL DAY. Tighten your criteria — A-setups only. "
                              "If no trade in 30 minutes, walk away. Protecting capital is the goal today. "
                              "Like the taxi driver on a slow day — call it early.")

                return json.dumps({
                    "market_temperature": temp,
                    "a_grade_setups": a_count,
                    "b_grade_setups": b_count,
                    "avg_relative_volume": round(avg_surge, 1),
                    "avg_pct_change": round(avg_chg, 1),
                    "setup_grades": scores,
                    "recommended_approach": advice,
                    "sizing_rule": (
                        "Full size after cushion built" if temp == "HOT" else
                        "¼ size until cushion, then selective full size" if temp == "WARM" else
                        "¼ size all day — do NOT size up regardless of cushion"
                    ),
                    "daily_goal_note": "Daily max loss = your daily profit goal. Hit max loss = done for the day, no exceptions."
                }, default=str)

            elif tool_name == "calculate_trade_levels":
                ticker = tool_input["ticker"].upper()
                entry = float(tool_input["entry_price"])
                stop = float(tool_input["stop_price"])
                target = float(tool_input["target_price"])
                share_sizes = tool_input.get("share_sizes", [100, 500, 1000, 5000])

                risk_per_share = entry - stop
                reward_per_share = target - entry
                pl_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
                meets_2to1 = pl_ratio >= 2.0

                breakeven_accuracy = 1 / (1 + pl_ratio) * 100 if pl_ratio > 0 else 100

                sizing_table = []
                for shares in share_sizes:
                    max_loss = shares * risk_per_share
                    max_gain = shares * reward_per_share
                    capital_deployed = shares * entry
                    doubled_gain = max_gain * 2  # if you add to the winner
                    sizing_table.append({
                        "shares": shares,
                        "capital_deployed": round(capital_deployed, 2),
                        "max_loss_$": round(max_loss, 2),
                        "profit_target_$": round(max_gain, 2),
                        "if_you_double_position_$": round(doubled_gain, 2),
                        "quarter_size_shares": shares // 4
                    })

                return json.dumps({
                    "ticker": ticker,
                    "entry": entry,
                    "stop_loss": stop,
                    "profit_target": target,
                    "risk_per_share": round(risk_per_share, 4),
                    "reward_per_share": round(reward_per_share, 4),
                    "profit_loss_ratio": round(pl_ratio, 2),
                    "meets_2to1_rule": meets_2to1,
                    "ross_verdict": (
                        f"✓ VALID SETUP — {pl_ratio:.1f}:1 P/L ratio. Breakeven accuracy only {breakeven_accuracy:.0f}%."
                        if meets_2to1 else
                        f"✗ SKIP THIS TRADE — {pl_ratio:.1f}:1 is below 2:1 minimum. "
                        f"Target too close to entry or stop too wide. Wait for a better setup."
                    ),
                    "start_with_quarter_size": True,
                    "quarter_size_note": "Always start at ¼ share size until cushion is built. Then full size. Then add to winners.",
                    "sizing_scenarios": sizing_table
                }, default=str)

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str, screener_context: str = "", on_tool_call=None) -> str:
        """
        Send a message to the agent and get a response.
        Handles multi-turn tool use automatically.
        on_tool_call: optional callback(tool_name, input_dict) for UI status updates.
        """
        full_message = user_message
        if screener_context and not self.conversation:
            full_message = (
                f"[CURRENT SCREENER RESULTS — apply Ross Cameron's 5-criteria grading to these]\n"
                f"{screener_context}\n\n"
                f"User question: {user_message}"
            )

        self.conversation.append({"role": "user", "content": full_message})

        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.conversation,
            )

            assistant_content = response.content
            self.conversation.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in assistant_content if hasattr(b, "text")]
                return "\n".join(text_blocks)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        if on_tool_call:
                            on_tool_call(block.name, block.input)
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                self.conversation.append({"role": "user", "content": tool_results})
                continue

            break

        return "I encountered an unexpected response. Please try again."

    def reset(self):
        """Clear conversation history."""
        self.conversation = []

    def build_screener_context(self, df: pd.DataFrame) -> str:
        """Format screener results with 5-criteria pre-scoring for agent context."""
        if df.empty:
            return "No stocks passed the screener filters."

        lines = [
            "SCREENER OUTPUT — Pre-graded against Ross Cameron's 5 criteria:",
            "(Criteria: 10%+ move | 5× rel.vol | news catalyst | $2–$20 price | <10M float)",
            ""
        ]

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            score = self._score_criteria(row_dict)
            grade = score["grade"]

            parts = [
                f"[{grade}] {row.get('ticker','?')}",
                f"${row.get('price', 0):.3f}",
                f"+{row.get('change_pct', 0):.1f}%",
                f"vol {row.get('surge_ratio', 0):.1f}× avg",
            ]
            if row.get("rsi"):
                parts.append(f"RSI {row['rsi']}")
            if row.get("float_m"):
                parts.append(f"float {row['float_m']}M")
            if row.get("has_news"):
                parts.append("✓ catalyst")
            else:
                parts.append("⚠ no news")

            # Criteria failures
            fails = [k for k, v in score["criteria"].items() if not v["pass"] and not v.get("partial")]
            if fails:
                parts.append(f"fails: {', '.join(fails)}")

            lines.append("  " + " | ".join(parts))

        # Market temperature pre-assessment
        if not df.empty:
            a_count = sum(1 for _, r in df.iterrows()
                          if self._score_criteria(r.to_dict())["grade"] == "A")
            avg_surge = df["surge_ratio"].mean() if "surge_ratio" in df.columns else 0
            if a_count >= 2 and avg_surge >= 5:
                lines.append("\n→ Market temp: HOT — multiple A-setups, size up after cushion")
            elif a_count >= 1:
                lines.append("\n→ Market temp: WARM — selective sizing, build cushion first")
            else:
                lines.append("\n→ Market temp: COLD — ¼ size only, walk away if nothing in 30min")

        return "\n".join(lines)
