"""
app.py — Penny Stock Day Trading Agent
Built on Ross Cameron's (Warrior Trading) methodology:
• 5-criteria stock selection
• Bull flag entries with 2:1 P/L
• Profit-cushion position sizing system
• Hot/cold market detection
Live data: Polygon.io | AI analysis: Gemini (Google AI Studio)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Pull keys from Streamlit secrets (cloud) or .env (local)
def _secret(key: str) -> str:
    try:
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

from data_fetcher import PolygonFetcher  # noqa: E402
from ai_agent import TradingAgent  # noqa: E402

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Day Trade Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f0f0f; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.grade-A { background:#0d3d1e; color:#4ade80; padding:3px 10px;
           border-radius:4px; font-weight:700; font-size:13px; }
.grade-B { background:#1a3a00; color:#a3e635; padding:3px 10px;
           border-radius:4px; font-weight:700; font-size:13px; }
.grade-C { background:#3d2a00; color:#fbbf24; padding:3px 10px;
           border-radius:4px; font-weight:700; font-size:13px; }
.grade-D { background:#3d0d0d; color:#f87171; padding:3px 10px;
           border-radius:4px; font-weight:700; font-size:13px; }
.metric-hot  { color:#4ade80; font-weight:700; font-size:16px; }
.metric-warm { color:#fbbf24; font-weight:700; font-size:16px; }
.metric-cold { color:#f87171; font-weight:700; font-size:16px; }
.chat-user { background:#1e293b; border-radius:10px; padding:12px 16px;
             margin:8px 0; border-left:3px solid #3b82f6; }
.chat-agent { background:#0f1f0f; border-radius:10px; padding:12px 16px;
              margin:8px 0; border-left:3px solid #22c55e; }
.rule-box { background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
            padding:12px 16px; margin:4px 0; font-size:13px; }
.stButton>button { width:100%; }
div[data-testid="metric-container"] { background:#111827; border-radius:8px;
    padding:12px; border:1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "screener_df": pd.DataFrame(),
        "chat_history": [],
        "agent": None,
        "last_scan_time": None,
        "screener_source": "",
        "screener_dropped": {},
        "polygon_key": _secret("POLYGON_API_KEY"),
        "anthropic_key": _secret("GEMINI_API_KEY"),
        "scanning": False,
        "selected_ticker": None,
        "daily_goal": 1000,
        "current_pnl": 0.0,
        "trades_today": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_fetcher():
    if not st.session_state.polygon_key:
        return None
    return PolygonFetcher(api_key=st.session_state.polygon_key)

def get_agent():
    if st.session_state.agent is None:
        if st.session_state.polygon_key and st.session_state.anthropic_key:
            st.session_state.agent = TradingAgent(
                gemini_api_key=st.session_state.anthropic_key,
                polygon_api_key=st.session_state.polygon_key,
            )
    return st.session_state.agent

def grade_badge(grade):
    return f'<span class="grade-{grade}">{grade}</span>'

def criteria_score(row: dict) -> tuple[str, int]:
    """Quick 5-criteria grade for display."""
    passes = 0
    if row.get("change_pct", 0) >= 10:
        passes += 1
    elif row.get("change_pct", 0) >= 2:
        passes += 0.5
    if row.get("surge_ratio", 0) >= 5:
        passes += 1
    elif row.get("surge_ratio", 0) >= 2:
        passes += 0.5
    if row.get("has_news"):
        passes += 1
    price = row.get("price", 0)
    if 2 <= price <= 20:
        passes += 1
    float_m = row.get("float_m")
    if float_m and float_m <= 10:
        passes += 1
    elif float_m and float_m <= 20:
        passes += 0.5

    passes = int(passes)
    if passes >= 5:
        return "A", passes
    elif passes >= 4:
        return "B", passes
    elif passes >= 3:
        return "C", passes
    return "D", passes

def cushion_pct():
    if st.session_state.daily_goal <= 0:
        return 0
    return min(100, (st.session_state.current_pnl / st.session_state.daily_goal) * 100)

def market_temp_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "COLD"
    a_count = sum(1 for _, r in df.iterrows() if criteria_score(r.to_dict())[0] == "A")
    avg_surge = df["surge_ratio"].mean() if "surge_ratio" in df.columns else 0
    if a_count >= 2 and avg_surge >= 5:
        return "HOT"
    if a_count >= 1:
        return "WARM"
    return "COLD"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    with st.expander("🔑 API Keys", expanded=not st.session_state.polygon_key):
        pk = st.text_input("Polygon.io API Key", value=st.session_state.polygon_key,
                           type="password", help="Free tier: polygon.io/dashboard/signup")
        ak = st.text_input("Google AI Studio API Key", value=st.session_state.anthropic_key,
                           type="password", help="aistudio.google.com → Get API key")
        if st.button("Save Keys"):
            st.session_state.polygon_key = pk
            st.session_state.anthropic_key = ak
            st.session_state.agent = None  # reset agent with new keys
            st.success("Keys saved!")

    st.markdown("---")
    st.markdown("## 🎯 Ross Cameron's 5 Criteria")
    st.markdown("""
    <div class="rule-box">
    <b>1. % Move</b> — Up 10%+ (min 2% pre-mkt)<br>
    <b>2. Rel. Volume</b> — 5× 50-day avg+<br>
    <b>3. News Catalyst</b> — Required<br>
    <b>4. Price Range</b> — $2–$20<br>
    <b>5. Float</b> — Under 10M shares
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 Screener Filters")

    min_price, max_price = st.select_slider(
        "Price Range ($) — Ross ideal: $2–$20",
        options=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50],
        value=(2, 20),
    )
    min_vol_k = st.slider("Min Avg Volume (K)", 100, 2000, 500, 100)
    min_surge = st.slider("Min Volume Surge (×)", 1.5, 10.0, 3.0, 0.5)
    min_chg = st.slider("Min % Change", 1, 50, 5, 1)
    max_float = st.slider("Max Float (M shares)", 1, 100, 20, 1)
    rsi_lo, rsi_hi = st.select_slider("RSI Range", options=list(range(10, 91, 5)),
                                       value=(40, 70))
    require_news = st.toggle("Require News Catalyst", value=False,
                              help="Ross says: no catalyst = likely pump-and-dump")

    st.markdown("---")
    st.markdown("## 💰 Daily P&L Tracker")
    st.session_state.daily_goal = st.number_input("Daily Goal ($)", 100, 50000, 1000, 100)
    pnl_input = st.number_input("Current P&L ($)", -10000, 50000,
                                  int(st.session_state.current_pnl), 50)
    st.session_state.current_pnl = float(pnl_input)
    st.session_state.trades_today = st.number_input("Trades Today", 0, 200,
                                                      st.session_state.trades_today, 1)

    # Cushion gauge
    cushion = cushion_pct()
    cushion_built = cushion >= 25
    color = "#4ade80" if cushion_built else "#fbbf24" if cushion > 0 else "#f87171"
    st.markdown(f"""
    **Cushion Progress:** {cushion:.0f}% of daily goal
    <div style="background:#1f2937;border-radius:6px;height:12px;margin:4px 0">
      <div style="background:{color};width:{min(cushion,100):.0f}%;height:12px;border-radius:6px;transition:width .3s"></div>
    </div>
    <small>{'✅ Cushion built → size up to FULL' if cushion_built else '⚠️ Stay at ¼ size until 25% cushion'}</small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Reset Agent Memory"):
        agent = get_agent()
        if agent:
            agent.reset()
            st.session_state.chat_history = []
            st.success("Agent memory cleared")


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("# 📈 Day Trade Agent")
st.caption("Ross Cameron methodology · Polygon.io live data · Gemini AI analysis")

tab_screener, tab_chart, tab_chat, tab_guide = st.tabs(
    ["🔍 Screener", "📊 Chart", "🤖 AI Agent", "📚 Framework"]
)


# ═══════════════════════════════════════════════
# TAB 1: SCREENER
# ═══════════════════════════════════════════════
with tab_screener:
    col_btn, col_time, col_temp = st.columns([2, 3, 2])

    with col_btn:
        run_scan = st.button("▶ Run Live Scan", type="primary", use_container_width=True)

    with col_time:
        if st.session_state.last_scan_time:
            src = getattr(st.session_state, "screener_source", "")
            src_label = " · 📡 Live" if src == "live" else f" · 🕐 Historical ({src.split(':')[1] if ':' in src else src})"
            st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')} · {len(st.session_state.screener_df)} results{src_label}")

    with col_temp:
        temp = market_temp_from_df(st.session_state.screener_df)
        temp_class = {"HOT": "metric-hot", "WARM": "metric-warm", "COLD": "metric-cold"}[temp]
        temp_emoji = {"HOT": "🔥", "WARM": "🌤", "COLD": "❄️"}[temp]
        st.markdown(f'Market: <span class="{temp_class}">{temp_emoji} {temp}</span>',
                    unsafe_allow_html=True)

    if run_scan:
        fetcher = get_fetcher()
        if not fetcher:
            st.error("Add your Polygon.io API key in the sidebar first.")
        else:
            # Validate API key first
            key_check = fetcher.check_api_key()
            if not key_check["valid"]:
                st.error(
                    f"❌ Polygon API key error: {key_check.get('error', 'invalid key')}. "
                    f"Check your key at polygon.io/dashboard."
                )
            else:
                progress_bar = st.progress(0, text="Starting scan…")
                status_text = st.empty()

                def on_progress(i, total, ticker):
                    pct = int((i / max(total, 1)) * 100)
                    progress_bar.progress(pct, text=f"Enriching {ticker} ({i+1}/{total})…")
                    status_text.caption(f"Fetching RSI + float + news for {ticker}…")

                with st.spinner("Scanning market…"):
                    try:
                        result = fetcher.screen(
                            min_price=min_price,
                            max_price=max_price,
                            min_avg_vol_k=min_vol_k,
                            min_surge=min_surge,
                            min_chg_pct=min_chg,
                            max_float_m=max_float,
                            rsi_lo=rsi_lo,
                            rsi_hi=rsi_hi,
                            require_news=require_news,
                            top_n=10,
                            enrich=True,
                            on_progress=on_progress,
                        )
                        progress_bar.empty()
                        status_text.empty()

                        df_result = result["df"]
                        source    = result["source"]
                        warning   = result.get("warning")
                        dropped   = result.get("dropped", {})

                        st.session_state.screener_df      = df_result
                        st.session_state.screener_source  = source
                        st.session_state.screener_dropped = dropped
                        st.session_state.last_scan_time   = datetime.now()

                        # Show source banner
                        if source.startswith("historical"):
                            trade_date = source.split(":")[1]
                            st.warning(
                                f"⚠️ **Market closed or no live data** — showing last trading day "
                                f"({trade_date}) from historical bars. "
                                f"Use for analysis and practice only."
                            )
                        else:
                            st.success(f"✅ Live scan complete — {len(df_result)} results")

                        # Show warning if empty with filter diagnostics
                        if warning:
                            d = dropped
                            st.error(f"⚠️ {warning}")
                            st.markdown(
                                f"**Filter funnel:** "
                                f"Snapshot → **{d.get('after_snapshot',0)}** stocks in price range · "
                                f"Volume filter → **{d.get('after_volume',0)}** · "
                                f"Surge filter → **{d.get('after_surge',0)}** · "
                                f"% Change filter → **{d.get('after_change',0)}** · "
                                f"After enrichment → **{d.get('after_enrich',0)}**"
                            )

                        # Reset agent context
                        agent = get_agent()
                        if agent and not df_result.empty:
                            agent.reset()
                            st.session_state.chat_history = []

                    except Exception as e:
                        st.error(f"Scan error: {e}")
                        progress_bar.empty()
                        status_text.empty()

    df = st.session_state.screener_df
    source_label = getattr(st.session_state, "screener_source", "")

    if df.empty:
        st.info("Click **▶ Run Live Scan** to fetch today's top movers.")
        with st.expander("Demo mode — what the screener will show"):
            st.markdown("""
            Each result will show:
            - **Setup Grade** (A/B/C/D) based on Ross Cameron's 5 criteria
            - Price, % change, volume surge ratio
            - Float size, RSI, bid-ask spread
            - News catalyst flag
            - Dollar volume score (ranking metric)
            
            Then ask the AI agent anything about the results.
            """)
    else:
        # ── Summary metrics ──────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        a_setups = sum(1 for _, r in df.iterrows() if criteria_score(r.to_dict())[0] == "A")
        b_setups = sum(1 for _, r in df.iterrows() if criteria_score(r.to_dict())[0] == "B")
        avg_surge_val = df["surge_ratio"].mean() if "surge_ratio" in df.columns else 0
        avg_chg_val = df["change_pct"].mean() if "change_pct" in df.columns else 0

        m1.metric("Total Results", len(df))
        m2.metric("A-Grade Setups", a_setups, help="All 5 criteria met")
        m3.metric("B-Grade Setups", b_setups, help="4/5 criteria met")
        m4.metric("Avg Vol Surge", f"{avg_surge_val:.1f}×")
        m5.metric("Avg % Change", f"+{avg_chg_val:.1f}%")

        st.markdown("---")

        # ── Results table ────────────────────────────────
        st.markdown("### Top Picks · Ranked by Dollar Volume")
        st.caption("Click a ticker to load its chart and send it to the AI agent.")

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            grade, passes = criteria_score(row_dict)
            ticker = row_dict.get("ticker", "?")

            with st.container():
                cols = st.columns([0.6, 1.2, 1, 1, 1, 1, 1, 1, 1, 1.5])

                with cols[0]:
                    st.markdown(grade_badge(grade), unsafe_allow_html=True)

                with cols[1]:
                    if st.button(f"**{ticker}**", key=f"ticker_{ticker}",
                                  help="Load chart + analyze with AI"):
                        st.session_state.selected_ticker = ticker

                with cols[2]:
                    price = row_dict.get("price", 0)
                    st.markdown(f"**${price:.3f}**")

                with cols[3]:
                    chg = row_dict.get("change_pct", 0)
                    color = "green" if chg > 0 else "red"
                    st.markdown(f":{color}[+{chg:.1f}%]")

                with cols[4]:
                    surge = row_dict.get("surge_ratio", 0)
                    surge_color = "green" if surge >= 5 else "orange" if surge >= 3 else "red"
                    st.markdown(f":{surge_color}[{surge:.1f}×]")

                with cols[5]:
                    rsi = row_dict.get("rsi")
                    if rsi:
                        rsi_color = "green" if 40 <= rsi <= 70 else "orange"
                        st.markdown(f":{rsi_color}[RSI {rsi}]")
                    else:
                        st.caption("RSI —")

                with cols[6]:
                    float_m = row_dict.get("float_m")
                    if float_m:
                        fc = "green" if float_m <= 10 else "orange" if float_m <= 20 else "red"
                        st.markdown(f":{fc}[{float_m}M]")
                    else:
                        st.caption("Float —")

                with cols[7]:
                    has_news = row_dict.get("has_news", False)
                    st.markdown("✅ news" if has_news else "⚠️ no news")

                with cols[8]:
                    spread = row_dict.get("spread", row_dict.get("bid_ask_spread"))
                    if spread:
                        spread_color = "green" if spread < 2 else "orange" if spread < 4 else "red"
                        st.markdown(f":{spread_color}[{spread:.1f}% sprd]")
                    else:
                        st.caption("Spread —")

                with cols[9]:
                    criteria_count = f"{passes}/5 criteria"
                    st.caption(criteria_count)

                st.divider()

        # ── Volume surge chart ───────────────────────────
        if "surge_ratio" in df.columns and "ticker" in df.columns:
            st.markdown("### Volume Surge vs % Change")
            fig = px.scatter(
                df.reset_index(),
                x="change_pct",
                y="surge_ratio",
                size="volume_today",
                color=[criteria_score(r.to_dict())[0] for _, r in df.iterrows()],
                color_discrete_map={"A": "#4ade80", "B": "#a3e635",
                                     "C": "#fbbf24", "D": "#f87171"},
                text="ticker",
                labels={"change_pct": "% Change", "surge_ratio": "Volume Surge (×)"},
                title="Bubble size = today's volume",
                template="plotly_dark",
            )
            fig.update_traces(textposition="top center")
            fig.add_hline(y=5, line_dash="dash", line_color="#666",
                          annotation_text="5× Ross minimum")
            fig.add_vline(x=10, line_dash="dash", line_color="#666",
                          annotation_text="10% A-grade")
            fig.update_layout(height=380, showlegend=True,
                               legend_title="Setup Grade",
                               plot_bgcolor="#0f0f0f", paper_bgcolor="#0f0f0f")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2: CHART
# ═══════════════════════════════════════════════
with tab_chart:
    ticker_input = st.text_input(
        "Ticker", value=st.session_state.selected_ticker or "",
        placeholder="e.g. ABIO", key="chart_ticker"
    ).upper().strip()

    col_days, col_fetch = st.columns([2, 1])
    with col_days:
        chart_days = st.slider("Days of history", 3, 30, 10)
    with col_fetch:
        fetch_chart = st.button("Load Chart", type="primary")

    if fetch_chart and ticker_input:
        fetcher = get_fetcher()
        if not fetcher:
            st.error("Add API keys in sidebar.")
        else:
            with st.spinner(f"Loading {ticker_input}…"):
                try:
                    bars_df = fetcher.get_agg_bars(ticker_input, days=chart_days)
                    news = fetcher.get_news(ticker_input, limit=5)

                    if bars_df.empty:
                        st.warning("No price data found.")
                    else:
                        # ── OHLCV candlestick ──
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=bars_df["date"],
                            open=bars_df["open"],
                            high=bars_df["high"],
                            low=bars_df["low"],
                            close=bars_df["close"],
                            name=ticker_input,
                            increasing_line_color="#4ade80",
                            decreasing_line_color="#f87171",
                        ))
                        if "vwap" in bars_df.columns:
                            fig.add_trace(go.Scatter(
                                x=bars_df["date"], y=bars_df["vwap"],
                                name="VWAP", line=dict(color="#60a5fa", width=1.5, dash="dot")
                            ))

                        # High of day line
                        hod = bars_df["high"].max()
                        fig.add_hline(y=hod, line_color="#fbbf24", line_dash="dash",
                                      annotation_text=f"HOD ${hod:.3f}",
                                      annotation_position="top right")

                        fig.update_layout(
                            title=f"{ticker_input} — {chart_days}d | Bull flag: look for squeeze → pullback → new high",
                            template="plotly_dark",
                            plot_bgcolor="#0f0f0f",
                            paper_bgcolor="#0f0f0f",
                            height=440,
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # ── Volume bars ──
                        fig_vol = go.Figure()
                        colors = ["#4ade80" if c >= o else "#f87171"
                                  for c, o in zip(bars_df["close"], bars_df["open"])]
                        fig_vol.add_trace(go.Bar(
                            x=bars_df["date"], y=bars_df["volume"],
                            marker_color=colors, name="Volume"
                        ))
                        fig_vol.update_layout(
                            template="plotly_dark", height=160,
                            plot_bgcolor="#0f0f0f", paper_bgcolor="#0f0f0f",
                            showlegend=False, margin=dict(t=10)
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)

                        # ── Trade Level Calculator ──
                        st.markdown("### 📐 Trade Level Calculator (2:1 P/L Rule)")
                        last_price = float(bars_df["close"].iloc[-1])
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            entry = st.number_input("Entry Price ($)", value=last_price,
                                                     format="%.4f", step=0.01, key="calc_entry")
                        with c2:
                            stop = st.number_input("Stop Loss ($) — low of pullback",
                                                    value=round(last_price * 0.95, 4),
                                                    format="%.4f", step=0.01, key="calc_stop")
                        with c3:
                            target = st.number_input("Target ($) — high of day",
                                                      value=round(hod, 4),
                                                      format="%.4f", step=0.01, key="calc_target")

                        if entry > stop > 0 and target > entry:
                            risk = entry - stop
                            reward = target - entry
                            pl = reward / risk if risk > 0 else 0
                            meets = pl >= 2.0
                            be_acc = 1 / (1 + pl) * 100 if pl > 0 else 100

                            col_pl, col_be, col_verdict = st.columns(3)
                            col_pl.metric("P/L Ratio", f"{pl:.2f}:1",
                                           delta="✓ Meets 2:1" if meets else "✗ Below 2:1",
                                           delta_color="normal" if meets else "inverse")
                            col_be.metric("Breakeven Accuracy", f"{be_acc:.0f}%",
                                           help="You only need to be right this % of the time")
                            col_verdict.metric("Risk per share", f"${risk:.4f}",
                                                delta=f"Target: ${reward:.4f}")

                            if meets:
                                st.success(f"✅ **Valid setup.** {pl:.1f}:1 P/L — only need {be_acc:.0f}% accuracy to break even.")
                            else:
                                st.error(f"❌ **Skip this trade.** {pl:.1f}:1 is below 2:1 minimum. "
                                          f"Move target up or tighten stop.")

                            # Sizing table
                            st.markdown("**Position size scenarios (remember: start at ¼ size)**")
                            sizes = [50, 100, 250, 500, 1000, 2500, 5000]
                            size_data = [{
                                "Shares": s,
                                "¼ Size Shares": s // 4,
                                "Capital ($)": f"${s * entry:,.0f}",
                                "Max Loss ($)": f"${s * risk:,.2f}",
                                "Profit Target ($)": f"${s * reward:,.2f}",
                                "If Double Position ($)": f"${s * reward * 2:,.2f}",
                            } for s in sizes]
                            st.dataframe(pd.DataFrame(size_data), use_container_width=True,
                                          hide_index=True)

                        # ── News ──
                        if news:
                            st.markdown(f"### 📰 Catalyst Check ({len(news)} articles, last 48h)")
                            for article in news:
                                st.markdown(
                                    f"**{article['publisher']}** · {article['published'][:10]}  \n"
                                    f"[{article['title']}]({article['url']})"
                                )
                        else:
                            st.warning("⚠️ No news in last 48 hours — **Criteria #3 FAILS.** "
                                        "Volume without catalyst = high pump-and-dump risk.")

                except Exception as e:
                    st.error(f"Chart error: {e}")
    else:
        st.info("Enter a ticker above and click **Load Chart** — or click a ticker in the Screener tab.")


# ═══════════════════════════════════════════════
# TAB 3: AI AGENT
# ═══════════════════════════════════════════════
with tab_chat:
    st.markdown("### 🤖 Day Trade AI Agent")
    st.caption("Powered by Ross Cameron's methodology · Ask about setups, grading, sizing, or market temperature")

    if not st.session_state.anthropic_key:  # reusing key for Gemini
        st.error("Add your Google AI Studio API key in the sidebar to use the AI agent.")
    elif not st.session_state.polygon_key:
        st.warning("Add your Polygon.io API key for live data lookups during chat.")
    else:
        # ── Quick analysis buttons ──────────────────────────────────
        st.markdown("**Quick Analysis:**")
        qcols = st.columns(4)
        quick_prompts = [
            ("🏆 Best setup today", "Which stock on the screener is the best A-quality setup today? Grade all of them against the 5 criteria and rank them."),
            ("🌡️ Market temperature", "What's today's market temperature based on the screener? Should I size up or stay at quarter size?"),
            ("📐 Bull flag check", "Are there any bull flag patterns forming on the screener stocks? Walk me through the entry, stop, and 2:1 target for the best one."),
            ("⚠️ Risk check", "What are the biggest risks in today's screener? Flag any pump-and-dump signals, missing catalysts, or wide spreads."),
        ]
        for i, (label, prompt) in enumerate(quick_prompts):
            if qcols[i].button(label, key=f"quick_{i}"):
                st.session_state.chat_history.append(("user", prompt))
                agent = get_agent()
                if agent:
                    df = st.session_state.screener_df
                    context = agent.build_screener_context(df) if not df.empty else ""
                    with st.spinner("Analyzing…"):
                        try:
                            tool_status = st.empty()
                            def on_tool(name, inp):
                                ticker = inp.get("ticker") or (inp.get("tickers", [""])[0])
                                labels = {
                                    "get_stock_details": f"🔍 Fetching details for {ticker}…",
                                    "get_stock_news": f"📰 Checking news for {ticker}…",
                                    "compare_tickers": "⚖️ Comparing tickers…",
                                    "get_price_bars": f"📊 Loading price bars for {ticker}…",
                                    "assess_market_temperature": "🌡️ Assessing market temperature…",
                                    "calculate_trade_levels": f"📐 Calculating trade levels for {ticker}…",
                                    "recommend_exit": f"🚪 Running exit analysis for {inp.get('ticker','')}…",
                                }
                                tool_status.caption(labels.get(name, f"Using tool: {name}…"))
                            reply = agent.chat(prompt, screener_context=context, on_tool_call=on_tool)
                            tool_status.empty()
                            st.session_state.chat_history.append(("agent", reply))
                        except Exception as e:
                            st.error(f"Agent error: {e}")
                st.rerun()

        # ── Exit recommendation panel ─────────────────────────────
        st.markdown("---")
        st.markdown("**🚪 Open Position — Exit Recommendation:**")
        st.caption("Fill in your current position and get a CLOSE / SCALE OUT / HOLD / ADD verdict from the agent.")

        with st.expander("Enter open position details", expanded=False):
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                exit_ticker  = st.text_input("Ticker", key="exit_ticker", placeholder="e.g. ABIO").upper().strip()
                exit_dir     = st.selectbox("Direction", ["LONG", "SHORT"], key="exit_dir")
                exit_dom     = st.selectbox("Day of move", [1, 2, 3], key="exit_dom",
                                            format_func=lambda x: f"Day {x}")
            with ex2:
                exit_entry   = st.number_input("Entry price ($)", 0.01, value=5.00, format="%.4f", key="exit_entry")
                exit_stop    = st.number_input("Stop loss ($)",   0.01, value=4.85, format="%.4f", key="exit_stop")
                exit_target  = st.number_input("Target ($)",      0.01, value=5.30, format="%.4f", key="exit_target")
            with ex3:
                exit_current = st.number_input("Current price ($)", 0.01, value=5.15, format="%.4f", key="exit_current")
                exit_shares  = st.number_input("Shares held", 1, value=500, step=50, key="exit_shares")
                exit_slip    = st.number_input("Slippage/share (¢)", 0.0, 10.0, 2.0, 0.5, key="exit_slip")

            ex4, ex5, ex6 = st.columns(3)
            with ex4:
                exit_entry_time   = st.text_input("Entry time (HH:MM)", placeholder="09:45", key="exit_etime")
            with ex5:
                exit_current_time = st.text_input("Current time (HH:MM)", placeholder="10:30", key="exit_ctime")
            with ex6:
                exit_cushion = st.toggle("Cushion built today?", value=False, key="exit_cushion",
                                         help="Have you already banked 25%+ of your daily goal?")

            # Live P&L preview
            if exit_dir == "LONG":
                preview_gross = (exit_current - exit_entry) * exit_shares
            else:
                preview_gross = (exit_entry - exit_current) * exit_shares
            preview_slip = (exit_slip / 100) * exit_shares * 2
            preview_net  = preview_gross - preview_slip
            prev_col = "#4ade80" if preview_net >= 0 else "#f87171"
            risk_sh = abs(exit_entry - exit_stop)
            achieved = (abs(exit_current - exit_entry) / risk_sh) if risk_sh > 0 else 0

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;
                        padding:10px 16px;margin:6px 0;font-size:13px;display:flex;gap:24px;flex-wrap:wrap">
              <span>Gross: <b style="color:{'#4ade80' if preview_gross>=0 else '#f87171'}">${preview_gross:+,.2f}</b></span>
              <span>Slip: <b style="color:#f87171">-${preview_slip:.2f}</b></span>
              <span>Net: <b style="color:{prev_col};font-size:15px">${preview_net:+,.2f}</b></span>
              <span>P/L achieved: <b>{achieved:.2f}:1</b></span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚪 Get Exit Recommendation", type="primary", key="exit_btn"):
                if not exit_ticker:
                    st.error("Enter a ticker.")
                else:
                    prompt = (
                        f"I'm currently in an open {exit_dir} position on {exit_ticker}. "
                        f"Here are my details: entry ${exit_entry:.4f}, current price ${exit_current:.4f}, "
                        f"stop ${exit_stop:.4f}, target ${exit_target:.4f}, {exit_shares} shares, "
                        f"day {exit_dom} of the move, entered at {exit_entry_time or 'unknown time'}, "
                        f"current time {exit_current_time or 'unknown'}, "
                        f"slippage {exit_slip:.1f}¢/share, "
                        f"cushion {'built' if exit_cushion else 'NOT built'} today. "
                        f"Should I close, scale out, hold, or add? Give me a specific verdict."
                    )
                    st.session_state.chat_history.append(("user", prompt))
                    agent = get_agent()
                    if agent:
                        with st.spinner("Running exit analysis…"):
                            try:
                                tool_status = st.empty()
                                def on_tool_exit(name, inp):
                                    tool_status.caption(
                                        f"🚪 Fetching live RSI, VWAP, and price data for {inp.get('ticker',exit_ticker)}…"
                                    )
                                reply = agent.chat(prompt, on_tool_call=on_tool_exit)
                                tool_status.empty()
                                st.session_state.chat_history.append(("agent", reply))
                            except Exception as e:
                                st.error(f"Agent error: {e}")
                    st.rerun()

        st.markdown("---")

        # Chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="chat-user">👤 {message}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-agent">🤖 {message}</div>',
                            unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask the agent…",
                placeholder=(
                    "Examples:\n"
                    "• Grade ABIO against the 5 criteria\n"
                    "• Is there a bull flag on PHIO? What's the 2:1 entry?\n"
                    "• Should I size up today or stay at quarter size?\n"
                    "• Compare ABIO and BIOR — which is the better setup?\n"
                    "• What's my max loss if I trade 500 shares of VERB with a 5¢ stop?"
                ),
                height=100,
            )
            submit = st.form_submit_button("Send ↗", type="primary", use_container_width=True)

        if submit and user_input.strip():
            st.session_state.chat_history.append(("user", user_input))
            agent = get_agent()
            if agent:
                df = st.session_state.screener_df
                context = agent.build_screener_context(df) if not df.empty else ""
                with st.spinner("Agent thinking…"):
                    try:
                        tool_status = st.empty()
                        def on_tool(name, inp):
                            ticker = inp.get("ticker") or (inp.get("tickers", [""])[0] if inp.get("tickers") else "")
                            labels = {
                                "get_stock_details": f"🔍 Fetching details for {ticker}…",
                                "get_stock_news": f"📰 Checking catalyst for {ticker}…",
                                "compare_tickers": "⚖️ Comparing tickers…",
                                "get_price_bars": f"📊 Loading price bars for {ticker}…",
                                "assess_market_temperature": "🌡️ Reading market temperature…",
                                "calculate_trade_levels": f"📐 Calculating 2:1 levels for {ticker}…",
                            }
                            tool_status.caption(labels.get(name, f"Using tool: {name}…"))

                        reply = agent.chat(user_input, screener_context=context, on_tool_call=on_tool)
                        tool_status.empty()
                        st.session_state.chat_history.append(("agent", reply))
                    except Exception as e:
                        st.error(f"Agent error: {e}")
            st.rerun()


# ═══════════════════════════════════════════════
# TAB 4: FRAMEWORK REFERENCE
# ═══════════════════════════════════════════════
with tab_guide:
    st.markdown("## 📚 Ross Cameron's Day Trading Framework")
    st.caption("Warrior Trading methodology — the rules baked into this agent")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🎯 The 5 Criteria")
        st.markdown("""
        | # | Criteria | A-Grade | B-Grade |
        |---|----------|---------|---------|
        | 1 | % Move | 10%+ | 2–10% |
        | 2 | Relative Volume | 5×+ | 2–5× |
        | 3 | News Catalyst | Required | Required |
        | 4 | Price Range | $2–$20 | $1–$25 |
        | 5 | Float | <10M | 10–20M |
        
        **Only trade A and B setups.** C = high risk. D = skip.
        """)

        st.markdown("### 📐 The 2:1 P/L Rule")
        st.markdown("""
        Never take a trade where reward < 2× risk.
        
        | P/L Ratio | Breakeven Accuracy |
        |-----------|-------------------|
        | 1:1 | 50% |
        | **2:1** | **33%** |
        | 3:1 | 25% |
        
        **Entry:** First candle to make a new high  
        **Stop:** Low of the pullback  
        **Target:** High of day (retest)
        """)

        st.markdown("### 📈 The Bull Flag Pattern")
        st.markdown("""
        ```
        ↑ Squeeze up (3–7 green candles, rising volume)
            ↘ Pullback (lighter volume, holds 50%+ of move)
                → Entry: first candle making new high
        ```
        - **First pullbacks** = strongest entry
        - **Second pullbacks** = acceptable  
        - **Third pullbacks** = caution
        - Never chase — if you missed it, wait for the next one
        """)

    with c2:
        st.markdown("### 💰 The Profit Cushion System")
        st.markdown("""
        **Why:** Prevents emotional spiral on bad days (46% accuracy on red days vs 70% on green days)
        
        **PHASE 1 — Build Cushion (¼ size)**
        - Start every day at ¼ of normal share size
        - Stay there until P&L = 25% of daily goal
        - If you never reach it: ¼ size ALL day
        
        **PHASE 2 — Full Size**  
        - Cross the 25% cushion → size up to full
        - Give it back → drop back to ¼ size immediately
        
        **PHASE 3 — Add to Winners**
        - Trade working? Double the position
        - Move stop to breakeven
        - **Never add to losers**
        
        **STOP TRADING when:**
        - Hit daily max loss (= daily goal $)
        - No trade in 30 minutes
        - 3 losses in a row — reassess quality
        """)

        st.markdown("### 🌡️ Hot vs Cold Market")
        st.markdown("""
        **HOT 🔥** — Multiple A-setups, surge 5×+, first pullbacks resolving clean  
        → Full size after cushion. Add to winners. Trade longer.
        
        **WARM 🌤** — 1 A-setup, some B-setups, mixed action  
        → ¼ size until cushion. Selective full size.
        
        **COLD ❄️** — Few/no A-setups, stocks fading, spreads wide  
        → ¼ size ALL day. Walk away early. Protect capital.
        
        *"Like the taxi driver: on a slow day, go home early.  
        On a busy day, keep driving."* — Annie Duke / Ross Cameron
        """)

        st.markdown("### ⚠️ The Negative Feedback Loop")
        st.markdown("""
        ```
        Loss → Emotional pain
             → Revenge trading (lower quality)  
             → Bigger losses  
             → Deeper spiral
        ```
        **Break it:** Accept the loss. Reset to ¼ size.  
        Raise your quality filter back to A-only. Walk away if needed.
        
        **Two causes of failure:**
        1. Trading without a strategy (shooting from the hip)
        2. Having a strategy but lacking discipline to follow it
        """)

    st.markdown("---")
    st.markdown("### 📈 Scaling Path for New Traders")
    scale_data = {
        "Phase": ["Start", "Week 1–2", "Month 1", "1,000 trades", "2,000 trades", "Advanced"],
        "Share Size": ["10", "20–50", "100–160", "500–1,000", "1,500–5,000", "10,000–16,000+"],
        "Goal": ["Prove the concept", "Build habits", "Cover commissions", "$10K profit", "$100K profit", "$1M+ profit"],
        "Key Rule": ["Paper trade first", "2 trades/day max", "Track every metric",
                     "Survive till you thrive", "Scale winners", "Find sweet spot — diminishing returns exist"],
    }
    st.dataframe(pd.DataFrame(scale_data), use_container_width=True, hide_index=True)

    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. "
                "Day trading is extremely risky. Most retail traders lose money. "
                "This is not financial advice. Always practice in a simulator first. "
                "Never risk money you cannot afford to lose.")
