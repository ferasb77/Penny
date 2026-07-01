"""
technical_analysis.py
Generates a professional technical analysis PDF report for a selected stock.
Uses reportlab for PDF creation. Called from render_ticker_panel() in app.py.

Report sections:
  1. Header — ticker, date, setup grade
  2. 5-Criteria Scorecard
  3. Price Action Summary (from daily bars)
  4. Bull Flag Assessment
  5. Trade Level Calculator (entry / stop / target / 2:1 check)
  6. Risk Management (cushion, position sizing)
  7. AI Narrative (the full agent analysis, plain text)
  8. Disclaimer
"""

import io
from datetime import datetime

import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_RIGHT


# ── Colour palette ─────────────────────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0f172a")
ACCENT     = colors.HexColor("#22c55e")
RED        = colors.HexColor("#ef4444")
ORANGE     = colors.HexColor("#f97316")
YELLOW     = colors.HexColor("#eab308")
BLUE       = colors.HexColor("#3b82f6")
LIGHT_GREY = colors.HexColor("#94a3b8")
MID_GREY   = colors.HexColor("#334155")
WHITE      = colors.white
OFF_WHITE  = colors.HexColor("#f8fafc")
GRADE_COLORS = {
    "A": colors.HexColor("#16a34a"),
    "B": colors.HexColor("#65a30d"),
    "C": colors.HexColor("#d97706"),
    "D": colors.HexColor("#dc2626"),
}


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=22, textColor=DARK_BG, spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, textColor=LIGHT_GREY, spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Heading2"],
            fontSize=13, textColor=DARK_BG, spaceBefore=14, spaceAfter=4,
            fontName="Helvetica-Bold", borderPad=0,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9.5, textColor=colors.HexColor("#1e293b"),
            leading=14, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"],
            fontSize=8, textColor=LIGHT_GREY, leading=11,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"],
            fontSize=9, fontName="Courier",
            textColor=colors.HexColor("#1e293b"), leading=13,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", parent=base["Normal"],
            fontSize=7.5, textColor=LIGHT_GREY,
            leading=11, spaceBefore=8,
        ),
    }
    return custom


def _hr(story, color=MID_GREY):
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=color))
    story.append(Spacer(1, 4))


def _grade_badge_table(grade: str) -> Table:
    """Render a coloured grade badge as a small table cell."""
    bg = GRADE_COLORS.get(grade, RED)
    t = Table([[grade]], colWidths=[0.4 * inch], rowHeights=[0.28 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), bg),
        ("TEXTCOLOR",   (0, 0), (-1, -1), WHITE),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [bg]),
    ]))
    return t


def _criteria_table(criteria: dict, styles: dict) -> Table:
    """
    Build the 5-criteria scorecard table.
    criteria: dict of {name: {pass, partial, label, note}}
    """
    NAMES = {
        "pct_change":      "1. % Move",
        "relative_volume": "2. Rel. Volume",
        "news_catalyst":   "3. News Catalyst",
        "price_range":     "4. Price Range",
        "float":           "5. Float",
    }
    rows = [["Criterion", "Value", "Status", "Note"]]
    for key, label in NAMES.items():
        c = criteria.get(key, {})
        passed  = c.get("pass", False)
        partial = c.get("partial", False)
        if passed:
            status = "PASS"
        elif partial:
            status = "PARTIAL"
        else:
            status = "FAIL"

        rows.append([
            label,
            c.get("label", "—"),
            status,
            c.get("note", ""),
        ])

    col_widths = [1.5 * inch, 1.2 * inch, 0.7 * inch, 3.1 * inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OFF_WHITE, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MID_GREY),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    # Colour the status column per row
    status_colors = {"PASS": GRADE_COLORS["A"], "PARTIAL": ORANGE, "FAIL": RED}
    for i, row in enumerate(rows[1:], 1):
        sc = status_colors.get(row[2], LIGHT_GREY)
        style_cmds.append(("TEXTCOLOR",  (2, i), (2, i), sc))
        style_cmds.append(("FONTNAME",   (2, i), (2, i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style_cmds))
    return t


def _price_action_table(bars_df: pd.DataFrame) -> Table:
    """Last 10 days of OHLCV as a compact table."""
    if bars_df.empty:
        return None

    recent = bars_df.tail(10).copy()
    rows   = [["Date", "Open", "High", "Low", "Close", "Volume", "VWAP", "Chg%"]]

    prev_close = None
    for _, row in recent.iterrows():
        date_str = str(row["date"])[:10]
        close    = row.get("close", 0)
        pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
        rows.append([
            date_str,
            f"${row.get('open', 0):.2f}",
            f"${row.get('high', 0):.2f}",
            f"${row.get('low',  0):.2f}",
            f"${close:.2f}",
            f"{int(row.get('volume', 0)):,}",
            f"${row.get('vwap', close):.2f}",
            f"{pct:+.1f}%" if prev_close else "—",
        ])
        prev_close = close

    col_widths = [0.9*inch, 0.65*inch, 0.65*inch, 0.65*inch,
                  0.65*inch, 1.0*inch, 0.65*inch, 0.65*inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ALIGN",       (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OFF_WHITE, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Colour the Chg% column
    for i, row in enumerate(rows[1:], 1):
        val = row[7]
        if val != "—":
            c = GRADE_COLORS["A"] if val.startswith("+") else RED
            style_cmds.append(("TEXTCOLOR", (7, i), (7, i), c))
            style_cmds.append(("FONTNAME",  (7, i), (7, i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style_cmds))
    return t


def _trade_levels_table(
    entry: float,
    stop: float,
    target: float,
) -> tuple[Table, bool, float, float]:
    """Build the trade levels table and return it plus key metrics."""
    risk   = entry - stop
    reward = target - entry
    pl     = reward / risk if risk > 0 else 0
    be_acc = 1 / (1 + pl) * 100 if pl > 0 else 100
    meets  = pl >= 2.0

    sizes  = [100, 250, 500, 1000, 2500]
    rows   = [["Shares", "¼ Size", "Capital", "Max Loss", "Target Profit", "If Doubled"]]
    for s in sizes:
        rows.append([
            str(s),
            str(s // 4),
            f"${s * entry:,.0f}",
            f"${s * risk:,.2f}",
            f"${s * reward:,.2f}",
            f"${s * reward * 2:,.2f}",
        ])

    col_widths = [0.65*inch, 0.55*inch, 0.85*inch, 0.85*inch, 1.0*inch, 0.95*inch]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ALIGN",       (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OFF_WHITE, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t, meets, pl, be_acc


def _bull_flag_assessment(bars_df: pd.DataFrame, styles: dict) -> list:
    """Analyse the price bars for bull flag characteristics."""
    story = []
    if bars_df.empty or len(bars_df) < 4:
        story.append(Paragraph("Insufficient price history for bull flag analysis.", styles["body"]))
        return story

    highs  = bars_df["high"].values
    lows   = bars_df["low"].values
    closes = bars_df["close"].values
    vols   = bars_df["volume"].values

    hod        = float(max(highs))
    current    = float(closes[-1])
    recent_low = float(min(lows[-3:]))
    pullback   = hod - recent_low
    pullback_pct = pullback / hod * 100 if hod > 0 else 0
    retrace    = (current - recent_low) / pullback * 100 if pullback > 0 else 0

    avg_vol_early = float(sum(vols[:3]) / 3) if len(vols) >= 3 else 0
    avg_vol_late  = float(sum(vols[-3:]) / 3) if len(vols) >= 3 else 0
    vol_declining = avg_vol_late < avg_vol_early

    flag_valid  = pullback_pct <= 50
    holding_50  = retrace >= 50

    checks = [
        ("High of Day",      f"${hod:.3f}",                     True),
        ("Current Price",    f"${current:.3f}",                  True),
        ("Pullback from HOD",f"{pullback_pct:.1f}%",            flag_valid),
        ("Holding 50%+",     f"{retrace:.0f}% of move intact",  holding_50),
        ("Volume declining", "Yes" if vol_declining else "No",  vol_declining),
    ]

    rows = [["Check", "Value", "Signal"]]
    for label, val, ok in checks:
        rows.append([label, val, "✓ Bullish" if ok else "✗ Warning"])

    t = Table(rows, colWidths=[1.8*inch, 1.5*inch, 1.2*inch])
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [OFF_WHITE, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, (_, _, ok) in enumerate(checks, 1):
        c = GRADE_COLORS["A"] if ok else RED
        style_cmds.append(("TEXTCOLOR", (2, i), (2, i), c))
        style_cmds.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)

    # Narrative
    if flag_valid and holding_50:
        verdict = "Bull flag structure appears <b>valid</b>. Price has pulled back less than 50% of the initial move with declining volume — a healthy consolidation."
    elif flag_valid and not holding_50:
        verdict = "Flag structure is forming but price has retraced more than 50% of the move. Wait for stabilisation before entry."
    else:
        verdict = "Price has pulled back more than 50% — flag is <b>broken</b>. Skip this setup and wait for a fresh move."

    story.append(Spacer(1, 6))
    story.append(Paragraph(verdict, styles["body"]))
    return story


def generate_pdf(
    ticker: str,
    bars_df: pd.DataFrame,
    criteria: dict,
    grade: str,
    entry: float,
    stop: float,
    target: float,
    ai_analysis: str = "",
    news: list = None,
    company_name: str = "",
) -> bytes:
    """
    Generate a technical analysis PDF report.
    Returns the PDF as bytes (for st.download_button).
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.65*inch, rightMargin=0.65*inch,
        topMargin=0.65*inch, bottomMargin=0.65*inch,
    )
    styles = _styles()
    story  = []
    now    = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── 1. Header ─────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph(f"<b>{ticker}</b>", ParagraphStyle(
            "hdr", fontSize=26, textColor=DARK_BG, fontName="Helvetica-Bold")),
        Paragraph(
            f"Technical Analysis Report<br/>"
            f"<font size=9 color='#94a3b8'>{company_name or ticker} &nbsp;·&nbsp; {now}</font>",
            ParagraphStyle("hdrsub", fontSize=13, textColor=MID_GREY,
                           fontName="Helvetica-Bold", leading=18)),
        _grade_badge_table(grade),
    ]]
    header_tbl = Table(header_data, colWidths=[1.2*inch, 4.6*inch, 0.7*inch])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",   (2, 0), (2, 0), "RIGHT"),
    ]))
    story.append(header_tbl)
    _hr(story, ACCENT)

    # Setup grade summary line
    passes = sum(1 for c in criteria.values() if c.get("pass"))
    grade_note = {
        "A": "All 5 criteria met — A-quality setup.",
        "B": f"{passes}/5 criteria met — B-quality setup. Trade with standard size.",
        "C": f"{passes}/5 criteria met — C-quality setup. Reduce size or skip.",
        "D": f"{passes}/5 criteria met — Below minimum threshold. Do not trade.",
    }.get(grade, "")
    grade_color = GRADE_COLORS.get(grade, RED)
    story.append(Paragraph(
        f'<font color="#{hex(int(grade_color.hexval()[2:], 16))[2:].zfill(6)}">'
        f'<b>Setup Grade: {grade}</b></font> &nbsp; {grade_note}',
        styles["body"]
    ))
    story.append(Spacer(1, 8))

    # ── 2. 5-Criteria Scorecard ────────────────────────────────────────────────
    story.append(Paragraph("Ross Cameron's 5-Criteria Scorecard", styles["section"]))
    story.append(_criteria_table(criteria, styles))
    story.append(Spacer(1, 8))

    # ── 3. Price Action ────────────────────────────────────────────────────────
    story.append(Paragraph("Price Action — Last 10 Sessions", styles["section"]))
    price_tbl = _price_action_table(bars_df)
    if price_tbl:
        story.append(price_tbl)
    else:
        story.append(Paragraph("No price history available.", styles["body"]))
    story.append(Spacer(1, 8))

    # Key levels summary
    if not bars_df.empty:
        hod  = float(bars_df["high"].max())
        lod  = float(bars_df["low"].min())
        last = float(bars_df["close"].iloc[-1])
        vwap = float(bars_df["vwap"].iloc[-1]) if "vwap" in bars_df.columns else last
        avg_vol = int(bars_df["volume"].mean())
        kl_data = [
            ["High of Day", f"${hod:.3f}", "Low of Period", f"${lod:.3f}"],
            ["Last Close",  f"${last:.3f}", "VWAP",         f"${vwap:.3f}"],
            ["Avg Volume",  f"{avg_vol:,}", "", ""],
        ]
        kl_tbl = Table(kl_data, colWidths=[1.2*inch, 1.0*inch, 1.2*inch, 1.0*inch])
        kl_tbl.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (0, -1), DARK_BG),
            ("TEXTCOLOR", (2, 0), (2, -1), DARK_BG),
            ("TEXTCOLOR", (1, 0), (1, -1), BLUE),
            ("TEXTCOLOR", (3, 0), (3, -1), BLUE),
            ("GRID",      (0, 0), (-1, -1), 0.3, MID_GREY),
            ("BACKGROUND",(0, 0), (-1, -1), OFF_WHITE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(kl_tbl)
        story.append(Spacer(1, 8))

    # ── 4. Bull Flag Assessment ────────────────────────────────────────────────
    story.append(Paragraph("Bull Flag Assessment", styles["section"]))
    story.extend(_bull_flag_assessment(bars_df, styles))
    story.append(Spacer(1, 8))

    # ── 5. Trade Levels ────────────────────────────────────────────────────────
    story.append(Paragraph("Trade Level Calculator", styles["section"]))

    levels_meta = [
        ["Entry (first candle to new high)", f"${entry:.4f}"],
        ["Stop Loss (low of pullback)",       f"${stop:.4f}"],
        ["Profit Target (high of day)",       f"${target:.4f}"],
        ["Risk per share",                    f"${entry - stop:.4f}"],
        ["Reward per share",                  f"${target - entry:.4f}"],
    ]
    risk   = entry - stop
    reward = target - entry
    pl     = reward / risk if risk > 0 else 0
    meets  = pl >= 2.0
    be_acc = 1 / (1 + pl) * 100 if pl > 0 else 100

    levels_meta.append([
        "P/L Ratio",
        f"{pl:.2f}:1  {'✓ MEETS 2:1 MINIMUM' if meets else '✗ BELOW 2:1 — skip trade'}"
    ])
    levels_meta.append(["Breakeven Accuracy", f"{be_acc:.0f}%"])

    lm_tbl = Table(levels_meta, colWidths=[2.8*inch, 3.7*inch])
    lm_style = [
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [OFF_WHITE, WHITE]),
        ("GRID",      (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    pl_row = len(levels_meta) - 2
    lm_style.append(
        ("TEXTCOLOR", (1, pl_row), (1, pl_row),
         GRADE_COLORS["A"] if meets else RED)
    )
    lm_style.append(("FONTNAME", (1, pl_row), (1, pl_row), "Helvetica-Bold"))
    lm_tbl.setStyle(TableStyle(lm_style))
    story.append(lm_tbl)
    story.append(Spacer(1, 6))

    # Position sizing table
    story.append(Paragraph("Position Sizing (remember: start at ¼ size until cushion built)", styles["small"]))
    story.append(Spacer(1, 4))
    sizing_tbl, _, _, _ = _trade_levels_table(entry, stop, target)
    story.append(sizing_tbl)
    story.append(Spacer(1, 8))

    # ── 6. News / Catalyst ────────────────────────────────────────────────────
    if news:
        story.append(Paragraph("News Catalyst (last 48h)", styles["section"]))
        for a in (news or [])[:5]:
            title     = a.get("title", "")
            publisher = a.get("publisher", "")
            published = a.get("published", "")[:10]
            story.append(Paragraph(
                f'<b>{publisher}</b> · {published}<br/>'
                f'<font size=8 color="#1d4ed8"><u>{title}</u></font>',
                styles["body"]
            ))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 4))

    # ── 7. AI Narrative ───────────────────────────────────────────────────────
    if ai_analysis and ai_analysis.strip():
        story.append(Paragraph("AI Analysis", styles["section"]))
        _hr(story)
        # Split into paragraphs, strip markdown bold/italic
        import re
        clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ai_analysis)
        clean = re.sub(r"\*(.+?)\*",   r"<i>\1</i>", clean)
        clean = re.sub(r"#{1,6}\s*", "", clean)
        for para in clean.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, styles["body"]))
                story.append(Spacer(1, 4))

    # ── 8. Disclaimer ─────────────────────────────────────────────────────────
    _hr(story)
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-powered tool for "
        "educational and informational purposes only. It does not constitute "
        "financial advice. Day trading involves substantial risk of loss. "
        "Past performance is not indicative of future results. Always practice "
        "in a simulator before trading with real money. Never risk capital you "
        "cannot afford to lose. The author of this tool is not a licensed "
        "financial advisor.",
        styles["disclaimer"]
    ))
    story.append(Paragraph(
        f"Generated by Day Trade Agent · enablemygrowth.com · {now}",
        ParagraphStyle("footer", fontSize=7, textColor=LIGHT_GREY,
                       alignment=TA_RIGHT, spaceBefore=4)
    ))

    doc.build(story)
    return buf.getvalue()
