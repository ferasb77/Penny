# Day Trade Agent 📈
**Ross Cameron (Warrior Trading) methodology · Polygon.io live data · Claude AI**

Built on the exact framework from Ross Cameron's Warrior Trading system:
5-criteria stock selection, bull flag entries with 2:1 P/L, profit-cushion position sizing, and hot/cold market detection.

---

## Quick Start

### 1. Clone & install
```bash
cd penny_screener
pip install -r requirements.txt
```

### 2. Get your API keys (both are free to start)

**Polygon.io** (live market data):
- Sign up at https://polygon.io/dashboard/signup
- Free tier gives you: snapshot data, previous close, ticker details, news, daily bars
- Note: free tier has a 15-min delay on real-time data

**Anthropic** (AI agent):
- Get a key at https://console.anthropic.com
- The agent uses `claude-sonnet-4-6`

### 3. Set up environment
```bash
cp .env.example .env
# Edit .env and add your keys
```

### 4. Run
```bash
streamlit run app.py
```

---

## What's Built In

### Ross Cameron's 5 Criteria (graded on every stock)
| # | Criteria | A-Grade | B-Grade |
|---|----------|---------|---------|
| 1 | % Move from prev close | 10%+ | 2–10% |
| 2 | Relative Volume vs 50d avg | 5×+ | 2–5× |
| 3 | News Catalyst | Required | Required |
| 4 | Price Range | $2–$20 | $1–$25 |
| 5 | Float size | <10M shares | 10–20M |

### The Profit Cushion System
- Start every day at **¼ position size**
- Build a 25% cushion of your daily goal before sizing up
- Add to winners (not losers), move stop to breakeven when adding
- Max loss = daily goal. Hit it → done for the day

### Bull Flag Entry Logic
- Squeeze up → pullback (light volume, holds 50%+ of move) → first candle to new high = entry
- Stop = low of pullback
- Target = high of day
- **Never take a trade below 2:1 P/L ratio**

### Hot/Cold Market Detection
- HOT 🔥 — multiple A-setups, 5×+ surge: size up, add to winners, trade longer
- WARM 🌤 — some quality setups: build cushion first, selective sizing
- COLD ❄️ — few setups, stocks fading: ¼ size all day, walk away early

### AI Agent Tools
The Claude agent can:
- `get_stock_details` — full data + 5-criteria grade for any ticker
- `get_stock_news` — catalyst verification (no news = pump risk)
- `compare_tickers` — rank multiple tickers by setup grade
- `get_price_bars` — OHLCV analysis + bull flag pattern detection
- `assess_market_temperature` — HOT/WARM/COLD verdict + sizing advice
- `calculate_trade_levels` — entry/stop/target with 2:1 verification + sizing table

---

## File Structure
```
penny_screener/
├── app.py              # Streamlit UI (4 tabs: Screener, Chart, AI Agent, Framework)
├── ai_agent.py         # Claude agent with Ross Cameron methodology
├── data_fetcher.py     # Polygon.io API wrapper
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚠️ Disclaimer
This tool is for **educational purposes only**. Day trading is extremely risky.
Most retail traders lose money. This is not financial advice.
Always practice in a simulator before trading real money.
Never risk money you cannot afford to lose.

The stocks screened (under $1.00) carry **extreme additional risk** including
manipulation, pump-and-dump schemes, halts, and extreme illiquidity.
Ross Cameron himself trades stocks in the **$2–$20 range** — the sub-$1 filter
here is for educational exploration only.
