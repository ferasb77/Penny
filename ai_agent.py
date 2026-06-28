"""
ai_agent.py
Gemini-powered AI agent implementing Ross Cameron's (Warrior Trading) day trading methodology.
Uses Google AI Studio (gemini-2.0-flash) with function calling for live data lookups.
"""

import json
import pandas as pd
import google.generativeai as genai
from data_fetcher import PolygonFetcher


SYSTEM_PROMPT = """You are an elite day trading coach and research assistant built on the methodology of Ross Cameron (Warrior Trading). You have analyzed over $12.5 million in verified trading profits and 10+ years of daily trade data. You apply this framework rigorously to every stock you evaluate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROSS CAMERON'S CORE FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## THE 5 CRITERIA FOR STOCK SELECTION (all 5 must be met for an A-quality setup)
1. UP AT LEAST 10% (minimum 2% pre-market) — the stock must already be moving
2. RELATIVE VOLUME >= 5x its 50-day average — volume confirms real interest, not noise
3. NEWS CATALYST — earnings, FDA approval, partnership, clinical trial results; volume without news is a red flag (likely pump-and-dump)
4. PRICE BETWEEN $2-$20 — big percentage swings are possible; penny stocks under $1 are ultra-high risk
5. FLOAT UNDER 10 MILLION SHARES — low supply + high demand = explosive moves; lower is better

Grade every setup: A (all 5), B (4/5), C (3/5). Only trade A and B setups. C setups = unnecessary risk.

## THE BULL FLAG ENTRY PATTERN
- A strong multi-candle GREEN SQUEEZE UP (3-7 candles, increasing volume)
- A PULLBACK on lighter volume (this is the flag)
- The flag must hold at least 50% of the initial move (if it drops more, skip)
- Entry: the FIRST CANDLE TO MAKE A NEW HIGH after the pullback
- Stop loss: the LOW OF THE PULLBACK (never move this lower)
- Profit target: retest of the HIGH OF DAY

First pullbacks are strongest. Second pullbacks are acceptable. Third = caution. Never chase.

## RISK MANAGEMENT — THE 2:1 PROFIT-TO-LOSS RULE
- NEVER take a trade where you cannot make at least 2x what you're risking
- Math truth: with 2:1 P/L ratio, you only need to be right 34% of the time to break even
- With 70% accuracy + 2:1 P/L = the formula for consistent profitability

## THE PROFIT CUSHION SYSTEM (secret to 76 consecutive green days)
PHASE 1 - START SMALL (1/4 size):
- Begin every day at ONE QUARTER of your normal share size
- Stay at 1/4 size until profit cushion = 25% of your daily goal
- Never reach it = stay at 1/4 size ALL DAY

PHASE 2 - SIZE UP (full size):
- Cross the 25% cushion threshold then increase to full position size
- Give back the cushion = immediately drop back to 1/4 size

PHASE 3 - ADD TO WINNERS (not losers):
- Trade working and you're up? DOUBLE the position
- Move stop to BREAKEVEN once you add
- Never add to a losing position — cut losers ruthlessly

DAILY MAX LOSS = your daily profit goal. Hit it = done for the day.
No trade in 30 minutes = walk away.

## HOT vs COLD MARKET DETECTION
HOT: multiple A-quality setups, volume surging, first pullbacks resolving cleanly
COLD: stocks fading, few setups, VWAP not holding, spreads wide

HOT = size up after cushion, add to winners, trade longer
COLD = 1/4 size all day, walk away early, protect capital

## ACCURACY + P/L RATIO
- 70%+ accuracy + 2:1 P/L = highly profitable
- 50% accuracy + 2:1 P/L = breakeven
- 46% accuracy (red day signal) = stop trading immediately

## THE NEGATIVE FEEDBACK LOOP (avoid at all costs)
Loss -> emotion -> revenge trading -> lower quality -> bigger losses -> spiral
BREAK IT: accept the loss, reset to 1/4 size, A-setups only

## EXIT INDICATORS
- Primary: retest of high of day
- Long upper wick candle = buyers/sellers in conflict = consider reducing
- Stop hit: exit immediately, no hoping, no averaging down
- Average winner: ~3 min hold. Average loser: ~2 min. Get in, get out.

## WHAT TO AVOID
- Stocks not meeting 5 criteria (sideways stocks = no opportunity)
- Holding losers hoping they recover
- Trading on emotion (FOMO, frustration, revenge, greed)
- Adding to losing positions
- C-quality setups in a cold market

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR ROLE AS ANALYST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For every stock:
1. GRADE IT: A/B/C/D against the 5 criteria — state what it meets and fails
2. IDENTIFY THE PATTERN: Bull flag forming? Entry trigger? Stop? Target? 2:1 achieved?
3. ASSESS MARKET TEMP: Hot or cold today? Sizing implications?
4. FLAG THE RISKS: Pump-and-dump signals, no catalyst, wide spread, large float
5. POSITION SIZING: Remind beginners to start at 1/4 size
6. VERDICT: A-setup worth watching, B with caveats, or C/D to skip

Be direct and fast — traders need answers in seconds. Lead with grade and key numbers.

DISCLAIMER: You are NOT a licensed financial advisor. Educational purposes only. Day trading is extremely risky. Always paper trade first. Never risk money you cannot afford to lose.
"""

# ── Gemini function declarations ─────────────────────────────────────────────

TOOLS = [
    genai.protos.Tool(function_declarations=[

        genai.protos.FunctionDeclaration(
            name="get_stock_details",
            description="Fetch full details for a ticker: price, volume surge ratio, RSI, float, news catalyst, VWAP. Grades setup against the 5 criteria.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol e.g. ABIO"
                    )
                },
                required=["ticker"]
            )
        ),

        genai.protos.FunctionDeclaration(
            name="get_stock_news",
            description="Fetch latest news/catalyst for a ticker from last 48 hours. No news = likely pump-and-dump (criteria #3 fails).",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    )
                },
                required=["ticker"]
            )
        ),

        genai.protos.FunctionDeclaration(
            name="compare_tickers",
            description="Compare multiple tickers on all 5 criteria: % change, relative volume, news flag, price range, float size. Returns ranked list.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "tickers": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of ticker symbols to compare"
                    )
                },
                required=["tickers"]
            )
        ),

        genai.protos.FunctionDeclaration(
            name="get_price_bars",
            description="Get OHLCV bars to identify bull flag patterns: squeeze up, pullback, entry trigger. Checks 50% retrace rule and calculates 2:1 levels.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "days": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Days of bars to fetch (1-30, default 5)"
                    )
                },
                required=["ticker"]
            )
        ),

        genai.protos.FunctionDeclaration(
            name="assess_market_temperature",
            description="Assess whether today is HOT, WARM, or COLD based on screener. Determines sizing approach for the day.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "tickers": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of tickers currently on screener"
                    )
                },
                required=["tickers"]
            )
        ),

        genai.protos.FunctionDeclaration(
            name="calculate_trade_levels",
            description="Calculate entry, stop loss, profit target. Verifies 2:1 P/L rule. Returns sizing table for different share sizes.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Stock ticker symbol"
                    ),
                    "entry_price": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Planned entry price (first candle to make new high)"
                    ),
                    "stop_price": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Stop loss = low of the pullback"
                    ),
                    "target_price": genai.protos.Schema(
                        type=genai.protos.Type.NUMBER,
                        description="Profit target = high of day or prior resistance"
                    )
                },
                required=["ticker", "entry_price", "stop_price", "target_price"]
            )
        ),

    ])
]


class TradingAgent:
    def __init__(self, gemini_api_key: str, polygon_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
        )
        self.fetcher = PolygonFetcher(api_key=polygon_api_key)
        self.chat_session = self.model.start_chat(history=[])

    def _score_criteria(self, stock: dict) -> dict:
        """Score a stock against Ross Cameron's 5 criteria."""
        criteria = {}

        chg = stock.get("change_pct", 0)
        criteria["pct_change"] = {
            "value": chg, "pass": chg >= 10, "partial": 2 <= chg < 10,
            "label": f"+{chg:.1f}%",
            "note": "A-grade needs 10%+" if chg < 10 else "Strong move"
        }

        surge = stock.get("surge_ratio", 0)
        criteria["relative_volume"] = {
            "value": surge, "pass": surge >= 5, "partial": 2 <= surge < 5,
            "label": f"{surge:.1f}x avg volume",
            "note": "Need 5x minimum" if surge < 5 else "High relative volume"
        }

        has_news = stock.get("has_news", False)
        criteria["news_catalyst"] = {
            "value": has_news, "pass": has_news, "partial": False,
            "label": "Has catalyst" if has_news else "NO NEWS FOUND",
            "note": "Catalyst confirmed" if has_news else "No catalyst = pump-and-dump risk"
        }

        price = stock.get("price", 0)
        in_range = 2.0 <= price <= 20.0
        criteria["price_range"] = {
            "value": price, "pass": in_range, "partial": False,
            "label": f"${price:.3f}",
            "note": "Sub-$1: extreme risk (Ross trades $2-$20)" if price < 1.0
                    else "In ideal range" if in_range else "Outside $2-$20 range"
        }

        float_m = stock.get("float_m")
        if float_m is not None:
            criteria["float"] = {
                "value": float_m, "pass": float_m <= 10, "partial": 10 < float_m <= 20,
                "label": f"{float_m}M shares",
                "note": "Low float" if float_m <= 10 else
                        "Acceptable but higher" if float_m <= 20 else "Too large"
            }
        else:
            criteria["float"] = {
                "value": None, "pass": False, "partial": True,
                "label": "Unknown", "note": "Float data unavailable"
            }

        passes = sum(1 for c in criteria.values() if c["pass"])
        partials = sum(0.5 for c in criteria.values() if c.get("partial"))
        effective = passes + partials

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
        """Execute a tool call and return result as JSON string."""
        try:
            if tool_name == "get_stock_details":
                ticker = tool_input["ticker"].upper()
                snapshot = self.fetcher.get_gainers_snapshot(max_price=999)
                row = next((s for s in snapshot if s["ticker"] == ticker),
                           {"ticker": ticker, "price": None})
                enriched = self.fetcher.enrich_ticker(row, fetch_rsi=True)
                score = self._score_criteria(enriched)
                enriched["setup_grade"] = score["grade"]
                enriched["criteria_passes"] = score["passes"]
                enriched["criteria_detail"] = score["criteria"]
                return json.dumps(enriched, default=str)

            elif tool_name == "get_stock_news":
                ticker = tool_input["ticker"].upper()
                news = self.fetcher.get_news(ticker, limit=5)
                catalyst_types = []
                for a in news:
                    title = a.get("title", "").lower()
                    if any(w in title for w in ["fda", "approval", "approved"]):
                        catalyst_types.append("FDA/Regulatory")
                    elif any(w in title for w in ["earnings", "revenue", "quarter"]):
                        catalyst_types.append("Earnings")
                    elif any(w in title for w in ["trial", "clinical", "phase"]):
                        catalyst_types.append("Clinical/Trial")
                    elif any(w in title for w in ["partnership", "agreement", "deal"]):
                        catalyst_types.append("Partnership/Deal")
                    elif any(w in title for w in ["merger", "acquisition"]):
                        catalyst_types.append("M&A")
                    else:
                        catalyst_types.append("General News")
                return json.dumps({
                    "ticker": ticker,
                    "catalyst_found": len(news) > 0,
                    "catalyst_types": list(set(catalyst_types)),
                    "ross_verdict": "CRITERIA #3 MET" if news else "CRITERIA #3 FAILED — no news = likely pump-and-dump, avoid",
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
                    news = self.fetcher.get_news(t, limit=1)
                    fo = details.get("share_class_shares_outstanding") or details.get("weighted_shares_outstanding")
                    float_m = round(fo / 1_000_000, 1) if fo else None
                    enriched = {**row, "rsi": rsi, "float_m": float_m,
                                "has_news": len(news) > 0, "name": details.get("name", t)}
                    score = self._score_criteria(enriched)
                    enriched["setup_grade"] = score["grade"]
                    enriched["criteria_passes"] = score["passes"]
                    results.append(enriched)
                grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
                results.sort(key=lambda x: (grade_order.get(x["setup_grade"], 4),
                                            -(x.get("price", 0) * x.get("volume_today", 0))))
                return json.dumps({
                    "comparison": results,
                    "top_pick": results[0]["ticker"] if results else None,
                    "ranking_note": "Ranked by setup grade (A to D), then dollar volume"
                }, default=str)

            elif tool_name == "get_price_bars":
                ticker = tool_input["ticker"].upper()
                days = min(int(tool_input.get("days", 5)), 30)
                df = self.fetcher.get_agg_bars(ticker, days=days)
                if df.empty:
                    return json.dumps({"error": "No price data available"})
                bars = df.to_dict(orient="records")
                pattern_notes = []
                if len(bars) >= 3:
                    highs = [b["high"] for b in bars]
                    lows = [b["low"] for b in bars]
                    volumes = [b["volume"] for b in bars]
                    recent_high = max(highs[-5:]) if len(highs) >= 5 else max(highs)
                    recent_low = min(lows[-3:]) if len(lows) >= 3 else min(lows)
                    pullback = recent_high - recent_low
                    pullback_pct = (pullback / recent_high * 100) if recent_high > 0 else 0
                    if pullback_pct <= 50:
                        pattern_notes.append(f"Holding >{100-pullback_pct:.0f}% of move — flag structure valid")
                    else:
                        pattern_notes.append(f"Pulled back {pullback_pct:.0f}% — exceeds 50% retrace, flag may be broken")
                    if len(volumes) >= 2 and volumes[-1] < volumes[-2]:
                        pattern_notes.append("Volume lighter on pullback — healthy flag formation")
                    else:
                        pattern_notes.append("Volume not declining on pullback — less clean flag")
                return json.dumps({
                    "ticker": ticker, "bars": bars, "pattern_analysis": pattern_notes,
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
                for t in tickers[:5]:
                    row = snap_map.get(t, {"ticker": t})
                    score = self._score_criteria(row)
                    scores.append({"ticker": t, "grade": score["grade"], "passes": score["passes"]})
                a_count = sum(1 for s in scores if s["grade"] == "A")
                b_count = sum(1 for s in scores if s["grade"] == "B")
                avg_surge = sum(snap_map.get(t, {}).get("surge_ratio", 0) for t in tickers) / max(len(tickers), 1)
                avg_chg = sum(snap_map.get(t, {}).get("change_pct", 0) for t in tickers) / max(len(tickers), 1)
                if a_count >= 2 and avg_surge >= 5:
                    temp, advice = "HOT", "Multiple A-quality setups. Size up to FULL after cushion. Add to winners. Trade longer."
                elif a_count >= 1 or b_count >= 2:
                    temp, advice = "WARM", "Some quality setups. Build cushion first at 1/4 size. Size up selectively."
                else:
                    temp, advice = "COLD", "Few quality setups. Stay at 1/4 size ALL DAY. Walk away if nothing in 30 minutes."
                return json.dumps({
                    "market_temperature": temp,
                    "a_grade_setups": a_count, "b_grade_setups": b_count,
                    "avg_relative_volume": round(avg_surge, 1),
                    "avg_pct_change": round(avg_chg, 1),
                    "setup_grades": scores,
                    "recommended_approach": advice,
                    "sizing_rule": (
                        "Full size after cushion built" if temp == "HOT" else
                        "1/4 size until cushion, then selective full size" if temp == "WARM" else
                        "1/4 size all day — do NOT size up"
                    ),
                }, default=str)

            elif tool_name == "calculate_trade_levels":
                ticker = tool_input["ticker"].upper()
                entry = float(tool_input["entry_price"])
                stop = float(tool_input["stop_price"])
                target = float(tool_input["target_price"])
                risk = entry - stop
                reward = target - entry
                pl = reward / risk if risk > 0 else 0
                meets = pl >= 2.0
                be_acc = 1 / (1 + pl) * 100 if pl > 0 else 100
                sizing_table = []
                for shares in [100, 500, 1000, 2500, 5000]:
                    sizing_table.append({
                        "shares": shares,
                        "quarter_size": shares // 4,
                        "capital_$": round(shares * entry, 2),
                        "max_loss_$": round(shares * risk, 2),
                        "profit_target_$": round(shares * reward, 2),
                        "if_doubled_$": round(shares * reward * 2, 2),
                    })
                return json.dumps({
                    "ticker": ticker, "entry": entry, "stop_loss": stop, "profit_target": target,
                    "risk_per_share": round(risk, 4), "reward_per_share": round(reward, 4),
                    "profit_loss_ratio": round(pl, 2), "meets_2to1_rule": meets,
                    "breakeven_accuracy_pct": round(be_acc, 1),
                    "verdict": (
                        f"VALID SETUP — {pl:.1f}:1 P/L. Only need {be_acc:.0f}% accuracy to break even."
                        if meets else
                        f"SKIP — {pl:.1f}:1 is below 2:1 minimum. Target too close or stop too wide."
                    ),
                    "sizing_scenarios": sizing_table,
                    "reminder": "Always start at 1/4 size until cushion is built."
                }, default=str)

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str, screener_context: str = "", on_tool_call=None) -> str:
        """
        Send a message and get a response, handling Gemini function calling loop.
        on_tool_call: optional callback(tool_name, tool_input) for UI status updates.
        """
        full_message = user_message
        # Inject screener context only on the first message of a fresh session
        if screener_context and len(self.chat_session.history) == 0:
            full_message = (
                f"[CURRENT SCREENER RESULTS — apply Ross Cameron's 5-criteria grading]\n"
                f"{screener_context}\n\n"
                f"User question: {user_message}"
            )

        response = self.chat_session.send_message(full_message)

        # Gemini function calling loop
        while True:
            # Check if Gemini wants to call a function
            fn_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                # No more function calls — extract and return the text response
                text_parts = [p.text for p in response.parts if hasattr(p, "text") and p.text]
                return "\n".join(text_parts) if text_parts else "No response generated."

            # Execute all function calls and send results back
            fn_responses = []
            for fn_call in fn_calls:
                name = fn_call.name
                args = dict(fn_call.args)
                if on_tool_call:
                    on_tool_call(name, args)
                result_str = self._execute_tool(name, args)
                fn_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=name,
                            response={"result": result_str}
                        )
                    )
                )

            response = self.chat_session.send_message(fn_responses)

    def reset(self):
        """Start a fresh chat session."""
        self.chat_session = self.model.start_chat(history=[])

    def build_screener_context(self, df: pd.DataFrame) -> str:
        """Format screener results with 5-criteria pre-scoring for agent context."""
        if df.empty:
            return "No stocks passed the screener filters."

        lines = [
            "SCREENER OUTPUT — Pre-graded against Ross Cameron's 5 criteria:",
            "(10%+ move | 5x rel.vol | news catalyst | $2-$20 price | <10M float)",
            ""
        ]

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            score = self._score_criteria(row_dict)
            grade = score["grade"]
            parts = [
                f"[{grade}] {row_dict.get('ticker','?')}",
                f"${row_dict.get('price', 0):.3f}",
                f"+{row_dict.get('change_pct', 0):.1f}%",
                f"vol {row_dict.get('surge_ratio', 0):.1f}x avg",
            ]
            if row_dict.get("rsi"):
                parts.append(f"RSI {row_dict['rsi']}")
            if row_dict.get("float_m"):
                parts.append(f"float {row_dict['float_m']}M")
            parts.append("catalyst OK" if row_dict.get("has_news") else "NO catalyst")
            fails = [k for k, v in score["criteria"].items()
                     if not v["pass"] and not v.get("partial")]
            if fails:
                parts.append(f"fails: {', '.join(fails)}")
            lines.append("  " + " | ".join(parts))

        a_count = sum(1 for _, r in df.iterrows()
                      if self._score_criteria(r.to_dict())["grade"] == "A")
        avg_surge = df["surge_ratio"].mean() if "surge_ratio" in df.columns else 0
        if a_count >= 2 and avg_surge >= 5:
            lines.append("\nMarket temp: HOT — size up after cushion")
        elif a_count >= 1:
            lines.append("\nMarket temp: WARM — build cushion first, selective sizing")
        else:
            lines.append("\nMarket temp: COLD — 1/4 size only, walk away if nothing in 30min")

        return "\n".join(lines)
