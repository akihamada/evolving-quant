# -*- coding: utf-8 -*-
"""
generate_dashboard.py — 静的HTMLダッシュボード生成スクリプト
============================================================
JSON データファイルを読み込み、Chart.js を使った
6タブ構成の静的 index.html を生成する。

GitHub Pages で公開可能な完全スタンドアロンHTML。

使い方:
    python generate_dashboard.py
"""
from __future__ import annotations

import html as html_mod
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "latest_evolution_results.json"
TRACK_RECORD_PATH = BASE_DIR / "ai_track_record.json"
HOLDINGS_PATH = BASE_DIR / "data" / "portfolio_holdings.json"
PERFORMANCE_PATH = BASE_DIR / "data" / "performance_history.json"
OUTPUT_PATH = BASE_DIR / "index.html"

# セクター分類（advanced_predictor と同期）
SECTOR_MAP = {
    "Semiconductor": ["NVDA", "ARM", "MRVL", "TSM", "COHR"],
    "Software":      ["MSFT", "PLTR", "DDOG"],
    "Energy":        ["NNE", "OKLO"],
    "Infrastructure": ["VRT", "ETN", "GLW"],
    "Pharma":        ["LLY"],
    "Dividend ETF":  ["VYM", "1489.T"],
    "Gold":          ["GLD", "1328.T"],
    "Retail":        ["8173.T"],
}


def get_sector(ticker: str) -> str:
    """ティッカーからセクター名を取得する。"""
    for sec, tickers in SECTOR_MAP.items():
        if ticker in tickers:
            return sec
    return "Other"


def load_json(path: Path) -> dict:
    """JSONファイルを読み込む。存在しない場合は空辞書を返す。"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("File not found: %s", path)
    return {}


def yahoo_chart_url(ticker: str) -> str:
    """ティッカーからYahoo FinanceチャートURLを生成する。"""
    if ticker.endswith(".T"):
        return f"https://finance.yahoo.co.jp/quote/{ticker}/chart"
    return f"https://finance.yahoo.com/quote/{ticker}/chart/"


def build_holdings_data(holdings: dict) -> list[dict]:
    """portfolio_holdings.json から表示用データを構築する。"""
    stocks = []
    for account_type in ["tokutei", "nisa"]:
        for s in holdings.get("us_stocks", {}).get(account_type, []):
            stocks.append({
                "ticker": s.get("ticker", ""),
                "shares": s.get("shares", 0),
                "cost": round(s.get("cost_basis_usd", 0), 1),
                "price": round(s.get("current_price_usd", 0), 1),
                "pnl": round(s.get("unrealized_pnl_usd", 0), 0),
                "account": account_type.upper(),
            })
    return stocks


def build_action_cards(record: dict, stocks: list[dict]) -> str:
    """最新のtrack recordからアクションカードHTMLを生成する。"""
    actions = record.get("actions", {})
    reasons = record.get("action_reasons", {})
    ai_alloc = record.get("ai_allocation", {})

    # stocks lookup
    stock_map: dict[str, dict] = {}
    for s in stocks:
        t = s["ticker"]
        if t not in stock_map:
            stock_map[t] = {"shares": 0, "cost": 0, "price": s["price"], "pnl": 0}
        stock_map[t]["shares"] += s["shares"]
        stock_map[t]["cost"] = round(
            (stock_map[t]["cost"] * (stock_map[t]["shares"] - s["shares"]) +
             s["cost"] * s["shares"]) / max(stock_map[t]["shares"], 1), 1
        ) if stock_map[t]["shares"] > 0 else s["cost"]
        stock_map[t]["pnl"] += s["pnl"]

    def card(ticker: str, action: str) -> str:
        """個別カードHTMLを生成する。"""
        css_class = action.lower()
        badge_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(action, "⚪")
        sm = stock_map.get(ticker, {})
        pnl = sm.get("pnl", 0)
        pnl_class = "pnl-positive" if pnl >= 0 else "pnl-negative"
        pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
        alloc_pct = ai_alloc.get(ticker, 0) * 100
        reason_text = html_mod.escape(reasons.get(ticker, ""))

        holdings_html = ""
        if sm:
            holdings_html = f'''<div class="holdings">
              <span>{sm.get("shares",0)}株</span>
              <span>取得${sm.get("cost",0):,.0f} → 現${sm.get("price",0):,.0f}</span>
              <span class="{pnl_class}">{pnl_str}</span>
            </div>'''

        return f'''<div class="action-card {css_class} fade-in">
          <div class="card-header">
            <span class="ticker">{ticker}</span>
            <span class="action-badge {css_class}">{badge_emoji} {action}</span>
          </div>
          {holdings_html}
          <div class="reason">{reason_text}</div>
          <div class="alloc-bar"><div class="alloc-bar-fill" style="width: {min(alloc_pct * 6.67, 100):.0f}%;"></div></div>
          <div class="alloc-label">推奨配分 {alloc_pct:.1f}%</div>
        </div>'''

    groups = {"BUY": [], "SELL": [], "HOLD": []}
    for t, a in actions.items():
        groups.setdefault(a, []).append(t)

    html = ""
    if groups.get("BUY"):
        html += f'<div class="section"><div class="section-title"><span class="icon">🟢</span> BUY — 買い増し推奨（{len(groups["BUY"])}銘柄）</div><div class="action-grid">'
        for t in groups["BUY"]:
            html += card(t, "BUY")
        html += '</div></div>'

    if groups.get("SELL"):
        html += f'<div class="section"><div class="section-title"><span class="icon">🔴</span> SELL — 売却推奨（{len(groups["SELL"])}銘柄）</div><div class="action-grid">'
        for t in groups["SELL"]:
            html += card(t, "SELL")
        html += '</div></div>'

    if groups.get("HOLD"):
        html += f'<div class="section"><div class="section-title"><span class="icon">🟡</span> HOLD — 現状維持（{len(groups["HOLD"])}銘柄）</div><div class="action-grid">'
        for t in groups["HOLD"]:
            html += card(t, "HOLD")
        html += '</div></div>'

    # New picks
    new_picks = record.get("new_picks", [])
    if new_picks:
        html += f'<div class="section"><div class="section-title"><span class="icon">🆕</span> 新規推奨銘柄（{len(new_picks)}銘柄）</div><div class="picks-grid">'
        for p in new_picks:
            html += f'''<div class="pick-card fade-in">
              <div class="pick-ticker">{p.get("ticker","")}</div>
              <div class="pick-sector">{p.get("sector","")}</div>
              <div class="pick-reason">{html_mod.escape(p.get("reason",""))}</div>
            </div>'''
        html += '</div></div>'

    # Risk scenarios
    risk = record.get("risk_scenarios", {})
    if risk:
        html += '<div class="section"><div class="section-title"><span class="icon">⚖️</span> リスクシナリオ</div><div class="risk-grid">'
        for key, css, emoji in [("bull", "bull", "🐂"), ("base", "base", "📊"), ("bear", "bear", "🐻")]:
            s = risk.get(key, {})
            html += f'''<div class="risk-card {css}">
              <div class="risk-prob">{s.get("probability",0)*100:.0f}%</div>
              <div class="risk-label">{emoji} {key.upper()}</div>
              <div class="risk-desc">{html_mod.escape(s.get("description",""))}</div>
            </div>'''
        html += '</div></div>'

    return html


def build_overview_data(results: dict) -> str:
    """Overview タブ用のChart.jsデータをJSON文字列で返す。"""
    tickers = results.get("tickers", [])
    allocs = results.get("allocations", {})
    ensemble = allocs.get("ensemble", {})
    bl = allocs.get("bl", {})

    equal_w = 1.0 / len(tickers) if tickers else 0

    return json.dumps({
        "tickers": tickers,
        "ensemble": [round(ensemble.get(t, 0) * 100, 1) for t in tickers],
        "bl": [round(bl.get(t, 0) * 100, 1) for t in tickers],
        "equal": [round(equal_w * 100, 1)] * len(tickers),
    })


def build_track_record_data(track: dict) -> str:
    """AI Report Card用のデータをJSON文字列で返す。"""
    records = track.get("records", [])
    data = []
    for r in records:
        data.append({
            "date": r.get("date", ""),
            "confidence": r.get("confidence", 0),
            "evaluated": r.get("evaluated", False),
            "reasoning": r.get("ai_reasoning", "")[:80],
            "actions": r.get("actions", {}),
        })

    evaluations = track.get("evaluations", [])
    eval_data = []
    for e in evaluations:
        pv = e.get("predicted_vs_actual", {})
        eval_data.append({
            "date": e.get("eval_date", ""),
            "rmse": round(pv.get("rmse", 0), 4),
            "direction_accuracy": round(pv.get("direction_accuracy", 0) * 100, 1),
            "calibration_score": round(pv.get("calibration_score", 0), 2),
        })

    return json.dumps({"records": data, "evaluations": eval_data})


def build_portfolio_html(holdings: dict, results: dict) -> str:
    """💼 Portfolio タブのHTML — 保有全体を一覧化。

    Args:
        holdings: portfolio_holdings.json の内容
        results: latest_evolution_results.json の内容

    Returns:
        HTMLフラグメント
    """
    rows = []
    for account_type in ["tokutei", "nisa"]:
        for s in holdings.get("us_stocks", {}).get(account_type, []):
            shares = s.get("shares", 0) or 0
            cost = float(s.get("cost_basis_usd", 0) or 0)
            price = float(s.get("current_price_usd", 0) or 0)
            mv = price * shares
            pnl = (price - cost) * shares if cost > 0 else (s.get("unrealized_pnl_usd", 0) or 0)
            pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
            ticker = s.get("ticker", "")
            rows.append({
                "ticker": ticker,
                "sector": get_sector(ticker),
                "account": "特定" if account_type == "tokutei" else "NISA",
                "currency": "USD",
                "shares": shares,
                "cost": cost,
                "price": price,
                "market_value": mv,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

    # 日本株
    for s in holdings.get("japan_stocks", {}).get("nisa_growth", []):
        shares = s.get("shares", 0) or 0
        cost = float(s.get("cost_basis_jpy", 0) or 0)
        price = float(s.get("current_price_jpy", 0) or 0)
        mv = price * shares
        pnl = (price - cost) * shares if cost > 0 else 0
        pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
        code = s.get("code", "")
        rows.append({
            "ticker": code,
            "sector": get_sector(f"{code}.T"),
            "account": "NISA成長",
            "currency": "JPY",
            "shares": shares,
            "cost": cost,
            "price": price,
            "market_value": mv,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    if not rows:
        return '<div class="section"><p style="color:var(--text-muted)">保有データがありません。</p></div>'

    # USD換算サマリー
    fx = float(holdings.get("metadata", {}).get("fx_rate", {}).get("USD_JPY", 150))
    total_usd = sum(r["market_value"] if r["currency"] == "USD" else r["market_value"] / fx for r in rows)
    total_pnl_usd = sum(r["pnl"] if r["currency"] == "USD" else r["pnl"] / fx for r in rows)

    # セクター集計
    sector_data: dict[str, float] = {}
    for r in rows:
        mv_usd = r["market_value"] if r["currency"] == "USD" else r["market_value"] / fx
        sector_data[r["sector"]] = sector_data.get(r["sector"], 0) + mv_usd
    sectors_sorted = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)

    # アカウント別集計
    acct_data: dict[str, float] = {}
    for r in rows:
        mv_usd = r["market_value"] if r["currency"] == "USD" else r["market_value"] / fx
        acct_data[r["account"]] = acct_data.get(r["account"], 0) + mv_usd

    # 通貨別集計
    usd_total = sum(r["market_value"] for r in rows if r["currency"] == "USD")
    jpy_total = sum(r["market_value"] for r in rows if r["currency"] == "JPY")

    html = '<div class="section"><div class="section-title"><span class="icon">💼</span> ポートフォリオ全体</div>'

    # 概要メトリクス
    html += '<div class="port-summary-grid">'
    html += f'<div class="port-metric"><div class="port-metric-label">総資産（USD換算）</div>'
    html += f'<div class="port-metric-value">${total_usd:,.0f}</div>'
    html += f'<div class="port-metric-sub">{len(rows)}銘柄 / FX ¥{fx:.1f}</div></div>'

    pnl_color = "var(--accent-green)" if total_pnl_usd >= 0 else "var(--accent-red)"
    pnl_sign = "+" if total_pnl_usd >= 0 else ""
    html += f'<div class="port-metric"><div class="port-metric-label">含み損益</div>'
    html += f'<div class="port-metric-value" style="color:{pnl_color}">{pnl_sign}${total_pnl_usd:,.0f}</div>'
    html += f'<div class="port-metric-sub">{pnl_sign}{(total_pnl_usd/(total_usd-total_pnl_usd)*100 if total_usd > total_pnl_usd else 0):.1f}%</div></div>'

    html += f'<div class="port-metric"><div class="port-metric-label">米国株</div>'
    html += f'<div class="port-metric-value" style="color:var(--accent-blue)">${usd_total:,.0f}</div>'
    n_us = sum(1 for r in rows if r["currency"] == "USD")
    html += f'<div class="port-metric-sub">{n_us}銘柄</div></div>'

    html += f'<div class="port-metric"><div class="port-metric-label">日本株</div>'
    html += f'<div class="port-metric-value" style="color:var(--accent-purple)">¥{jpy_total:,.0f}</div>'
    n_jp = sum(1 for r in rows if r["currency"] == "JPY")
    html += f'<div class="port-metric-sub">{n_jp}銘柄</div></div>'
    html += '</div>'

    # セクター配分バー
    html += '<div class="sector-section"><div class="sector-title">📊 セクター配分</div>'
    html += '<div class="sector-bars">'
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#06b6d4", "#ef4444", "#84cc16", "#ec4899"]
    for i, (sec, val) in enumerate(sectors_sorted):
        pct = val / total_usd * 100 if total_usd > 0 else 0
        color = colors[i % len(colors)]
        html += f'<div class="sector-bar-row">'
        html += f'<div class="sector-bar-label">{html_mod.escape(sec)}</div>'
        html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        html += f'<div class="sector-bar-value">${val:,.0f} <span class="sector-pct">({pct:.1f}%)</span></div>'
        html += '</div>'
    html += '</div></div>'

    # 全保有テーブル（ソート可能）
    html += '<div class="port-table-section"><div class="port-table-title">📋 全保有銘柄（クリックでソート）</div>'
    html += '<div class="port-table-wrap"><table class="port-table" id="portTable">'
    html += '<thead><tr>'
    html += '<th onclick="sortPortTable(0,\'str\')">銘柄 ↕</th>'
    html += '<th onclick="sortPortTable(1,\'str\')">セクター</th>'
    html += '<th onclick="sortPortTable(2,\'str\')">口座</th>'
    html += '<th onclick="sortPortTable(3,\'num\')">数量</th>'
    html += '<th onclick="sortPortTable(4,\'num\')">取得単価</th>'
    html += '<th onclick="sortPortTable(5,\'num\')">現在値</th>'
    html += '<th onclick="sortPortTable(6,\'num\')">時価(USD換算)</th>'
    html += '<th onclick="sortPortTable(7,\'num\')">含み損益</th>'
    html += '<th onclick="sortPortTable(8,\'num\')">損益率</th>'
    html += '<th>チャート</th>'
    html += '</tr></thead><tbody>'

    # 時価(USD換算)で降順ソート
    rows_sorted = sorted(rows, key=lambda r: r["market_value"] if r["currency"] == "USD" else r["market_value"] / fx, reverse=True)

    for r in rows_sorted:
        mv_usd = r["market_value"] if r["currency"] == "USD" else r["market_value"] / fx
        pnl_usd = r["pnl"] if r["currency"] == "USD" else r["pnl"] / fx
        pnl_class = "pnl-positive" if pnl_usd >= 0 else "pnl-negative"
        pnl_sign = "+" if pnl_usd >= 0 else ""
        cur_sym = "$" if r["currency"] == "USD" else "¥"
        url = yahoo_chart_url(r["ticker"] if r["currency"] == "USD" else f"{r['ticker']}.T")

        html += f'<tr>'
        html += f'<td class="port-ticker"><strong>{r["ticker"]}</strong></td>'
        html += f'<td>{r["sector"]}</td>'
        html += f'<td><span class="acct-badge">{r["account"]}</span></td>'
        html += f'<td data-num="{r["shares"]}">{r["shares"]:,}</td>'
        html += f'<td data-num="{r["cost"]}">{cur_sym}{r["cost"]:,.2f}</td>'
        html += f'<td data-num="{r["price"]}">{cur_sym}{r["price"]:,.2f}</td>'
        html += f'<td data-num="{mv_usd}">${mv_usd:,.0f}</td>'
        html += f'<td data-num="{pnl_usd}" class="{pnl_class}">{pnl_sign}${abs(pnl_usd):,.0f}</td>'
        html += f'<td data-num="{r["pnl_pct"]}" class="{pnl_class}">{pnl_sign}{r["pnl_pct"]:.1f}%</td>'
        html += f'<td><a href="{url}" target="_blank" class="chart-link">📈</a></td>'
        html += '</tr>'

    html += '</tbody></table></div></div>'
    html += '</div>'
    return html


def build_advanced_signals_html(results: dict) -> str:
    """🔮 高精度予測シグナルのHTMLを構築する。

    Args:
        results: latest_evolution_results.json の内容

    Returns:
        HTMLフラグメント、データなしなら空文字列
    """
    advanced = results.get("advanced_signals", [])
    if not advanced:
        return ''

    html = '<div class="section"><div class="section-title"><span class="icon">🔮</span> 高精度予測アンサンブル（5モデル統合）</div>'
    html += '<div class="adv-grid">'

    signal_color = {
        "STRONG_BUY":  ("var(--accent-green)", "🟢🟢"),
        "BUY":         ("var(--accent-green)", "🟢"),
        "HOLD":        ("var(--accent-yellow)", "🟡"),
        "SELL":        ("var(--accent-red)", "🔴"),
        "STRONG_SELL": ("var(--accent-red)", "🔴🔴"),
    }

    # 高確信度のみ表示（HOLDは除外）
    visible = [s for s in advanced if s.get("signal") != "HOLD" or s.get("confidence", 0) >= 0.6]
    visible = sorted(visible, key=lambda x: x.get("confidence", 0), reverse=True)[:12]

    for s in visible:
        sig = s.get("signal", "HOLD")
        color, emoji = signal_color.get(sig, ("var(--text-muted)", "⚪"))
        conf = s.get("confidence", 0)
        score = s.get("composite_score", 0)
        ticker = s.get("ticker", "?")
        sub = s.get("sub_scores", {})
        url = yahoo_chart_url(ticker)

        html += f'<div class="adv-card" style="border-left:4px solid {color}">'
        html += f'<div class="adv-header"><a href="{url}" target="_blank" class="adv-ticker">{ticker} 📈</a>'
        html += f'<span class="adv-signal" style="color:{color}">{emoji} {sig}</span></div>'
        html += f'<div class="adv-score">合成スコア: <strong style="color:{color}">{score:+.3f}</strong></div>'
        html += f'<div class="adv-conf">確信度: {conf:.0%}</div>'

        # サブスコアのミニバー
        html += '<div class="adv-subbars">'
        sub_labels = {
            "kalman_trend":    "Kalman",
            "hurst_regime":    "Hurst",
            "cross_sectional": "C-Sect",
            "vol_regime":      "Vol",
            "mean_reversion":  "M-Rev",
        }
        for key, label in sub_labels.items():
            v = sub.get(key, 0)
            v_pct = abs(v) * 50
            sign_color = "var(--accent-green)" if v > 0 else "var(--accent-red)" if v < 0 else "var(--text-muted)"
            html += f'<div class="adv-subbar">'
            html += f'<span class="adv-sublabel">{label}</span>'
            html += f'<span class="adv-subval" style="color:{sign_color}">{v:+.2f}</span>'
            if v >= 0:
                html += f'<div class="adv-bar-track"><div class="adv-bar-pos" style="width:{v_pct:.0f}%"></div></div>'
            else:
                html += f'<div class="adv-bar-track"><div class="adv-bar-neg" style="width:{v_pct:.0f}%"></div></div>'
            html += '</div>'
        html += '</div></div>'

    html += '</div></div>'
    return html


def build_performance_html() -> str:
    """📈 過去パフォーマンス履歴チャートのHTMLを構築する。

    Returns:
        HTMLフラグメント、データなしなら空文字列
    """
    if not PERFORMANCE_PATH.exists():
        return ''
    try:
        with open(PERFORMANCE_PATH, encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ''

    records = history.get("records", [])
    if len(records) < 2:
        return ''

    return '<div class="section"><div class="section-title"><span class="icon">📈</span> パフォーマンス履歴</div><div class="chart-container"><div style="position:relative;height:320px"><canvas id="chartHistory"></canvas></div></div></div>'


def build_performance_json() -> str:
    """パフォーマンス履歴をJSON文字列で返す（Chart.js用）。"""
    if not PERFORMANCE_PATH.exists():
        return '{"records":[]}'
    try:
        with open(PERFORMANCE_PATH, encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return '{"records":[]}'

    records = history.get("records", [])
    return json.dumps({
        "dates": [r.get("date", "") for r in records],
        "values_usd": [r.get("total_usd", 0) for r in records],
        "pnl_usd": [r.get("total_pnl_usd", 0) for r in records],
    })


def generate_meta_prompt_text(results: dict, track: dict) -> str:
    """メタプロンプトテキストを生成する。"""
    tickers = results.get("tickers", [])
    allocs = results.get("allocations", {})
    ensemble = allocs.get("ensemble", {})
    bl = allocs.get("bl", {})
    ew = results.get("ensemble_weights", [0.25, 0.25, 0.25, 0.25])
    regime = results.get("regime", "unknown")
    kl_val = results.get("kl_value", 0)
    evaluations = track.get("evaluations", [])

    regime_jp = {"low_vol": "低ボラティリティ",
                 "transition": "移行期（ボラ上昇中）",
                 "crisis": "危機（高ボラ）"}.get(regime, "不明")

    lines = [
        f"🧬 EVOLVING QUANT META-PROMPT — {datetime.now().strftime('%Y-%m-%d')}",
        f"Self-Reflection Generation: {len(evaluations)}",
        "",
        "═══ SECTION A: 数学的ベースライン ═══",
        f"マーケットレジーム: {regime_jp} (KL={kl_val:.4f})",
        "",
        "Ensemble最適配分:",
    ]
    for t in tickers:
        lines.append(f"  {t}: {ensemble.get(t, 0):.1%}")

    names = ["NCO", "RP", "MV", "MD"]
    lines.append(f"  Ensemble重み: {', '.join(f'{n}={w:.1%}' for n, w in zip(names, ew))}")
    lines.append("")
    lines.append("Black-Litterman配分:")
    for t in tickers:
        lines.append(f"  {t}: {bl.get(t, 0):.1%}")

    return "\\n".join(lines)


def generate_html(results: dict, track: dict, holdings: dict) -> str:
    """完全な静的HTMLダッシュボードを生成する。"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    # Data preparation
    stocks = build_holdings_data(holdings)
    records = track.get("records", [])
    latest_record = records[-1] if records else {}
    confidence = latest_record.get("confidence", 0)
    reasoning = html_mod.escape(latest_record.get("ai_reasoning", ""))

    # Summary stats
    summary = holdings.get("us_stocks", {}).get("summary", {})
    total_usd = summary.get("grand_total_usd", 0)
    total_pnl = summary.get("grand_total_pnl_usd", 0)
    pnl_pct = (total_pnl / (total_usd - total_pnl) * 100) if total_usd > total_pnl else 0

    regime = results.get("regime", "unknown")
    kl_val = results.get("kl_value", 0)
    regime_label = {"low_vol": "安定（低ボラ）",
                    "transition": "移行期（ボラ上昇中）",
                    "crisis": "危機（高ボラ）"}.get(regime, "不明")

    actions = latest_record.get("actions", {})
    n_buy = sum(1 for a in actions.values() if a == "BUY")
    n_hold = sum(1 for a in actions.values() if a == "HOLD")
    n_sell = sum(1 for a in actions.values() if a == "SELL")

    action_cards_html = build_action_cards(latest_record, stocks)
    overview_json = build_overview_data(results)
    track_json = build_track_record_data(track)
    prompt_text = generate_meta_prompt_text(results, track)
    portfolio_html = build_portfolio_html(holdings, results)
    advanced_html = build_advanced_signals_html(results)
    performance_chart_html = build_performance_html()
    performance_json = build_performance_json()

    # Tsumitate advice
    tsumi = latest_record.get("tsumitate_advice", {})
    tsumi_html = ""
    if tsumi:
        changes = tsumi.get("changes", [])
        tsumi_reason = tsumi.get("reasoning", "")
        tsumi_html = '<div class="section"><div class="section-title"><span class="icon">📊</span> 積立設定レビュー</div><div class="tsumitate-grid">'
        tsumi_html += '<div class="tsumitate-card"><h4>⚡ 変更提案</h4>'
        for c in changes:
            tsumi_html += f'<div class="change-item"><span class="arrow">→</span> {html_mod.escape(c)}</div>'
        tsumi_html += '</div>'
        tsumi_html += f'<div class="tsumitate-card"><h4>💡 理由</h4><p style="font-size:13px;color:var(--text-secondary);line-height:1.7">{html_mod.escape(tsumi_reason)}</p></div>'
        tsumi_html += '</div></div>'

    # Records table
    records_html = ""
    for r in reversed(records[-10:]):
        status = "✅" if r.get("evaluated") else "⏳"
        records_html += f'''<div class="record-row">
          <span class="record-date">{r.get("date","")}</span>
          <span class="record-status">{status}</span>
          <span class="record-conf">確信度 {r.get("confidence",0):.0%}</span>
          <span class="record-reason">{html_mod.escape(r.get("ai_reasoning","")[:60])}...</span>
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧬 Evolving Quant Dashboard</title>
<meta name="description" content="Self-learning AI portfolio dashboard">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-primary:#0a0e1a;--bg-card:#111827;--bg-card-hover:#1a2332;
  --border:#1e293b;--text-primary:#f1f5f9;--text-secondary:#94a3b8;--text-muted:#64748b;
  --accent-green:#10b981;--accent-green-bg:rgba(16,185,129,0.12);
  --accent-red:#ef4444;--accent-red-bg:rgba(239,68,68,0.12);
  --accent-yellow:#f59e0b;--accent-yellow-bg:rgba(245,158,11,0.12);
  --accent-blue:#3b82f6;--accent-blue-bg:rgba(59,130,246,0.12);
  --accent-purple:#a855f7;--accent-purple-bg:rgba(168,85,247,0.12);
  --accent-cyan:#06b6d4;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg-primary);color:var(--text-primary);min-height:100vh;line-height:1.5}}
.container{{max-width:1400px;margin:0 auto;padding:24px}}

/* Header */
.header{{display:flex;justify-content:space-between;align-items:center;padding:28px 32px;background:linear-gradient(135deg,#111827,#1a2332);border:1px solid var(--border);border-radius:20px;margin-bottom:24px;position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent-blue),var(--accent-purple),var(--accent-cyan))}}
.header h1{{font-size:26px;font-weight:800;letter-spacing:-0.5px}}
.header h1 span{{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.header-meta{{text-align:right}}
.header-meta .date{{color:var(--text-secondary);font-size:14px}}
.header-meta .confidence{{display:inline-flex;align-items:center;gap:6px;background:var(--accent-blue-bg);border:1px solid rgba(59,130,246,0.3);border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;color:var(--accent-blue);margin-top:6px}}
.header-meta .confidence .dot{{width:8px;height:8px;border-radius:50%;background:var(--accent-blue);animation:pulse 2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}

/* Stats */
.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.stat-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:20px 24px;transition:all 0.3s}}
.stat-card:hover{{transform:translateY(-2px);border-color:rgba(59,130,246,0.3)}}
.stat-label{{font-size:12px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;font-weight:600}}
.stat-value{{font-size:28px;font-weight:800;margin-top:4px;font-family:'JetBrains Mono',monospace}}
.stat-sub{{font-size:13px;color:var(--text-secondary);margin-top:2px}}

/* Tabs */
.tab-nav{{display:flex;gap:4px;margin-bottom:20px;overflow-x:auto;padding-bottom:4px}}
.tab-btn{{padding:10px 20px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px 10px 0 0;color:var(--text-secondary);cursor:pointer;font-size:14px;font-weight:600;white-space:nowrap;transition:all 0.3s;font-family:'Inter',sans-serif}}
.tab-btn:hover{{background:var(--bg-card-hover);color:var(--text-primary)}}
.tab-btn.active{{background:linear-gradient(135deg,#3b82f6,#6c5ce7);color:#fff;border-color:transparent}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}

/* Strategy Banner */
.strategy-banner{{background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(168,85,247,0.08));border:1px solid rgba(59,130,246,0.2);border-radius:16px;padding:20px 28px;margin-bottom:24px;font-size:15px;color:var(--text-secondary);line-height:1.7}}
.strategy-banner strong{{color:var(--text-primary)}}

/* Sections & Cards */
.section{{margin-bottom:28px}}
.section-title{{font-size:18px;font-weight:700;margin-bottom:14px;display:flex;align-items:center;gap:10px}}
.section-title .icon{{font-size:22px}}

.action-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
.action-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;display:flex;flex-direction:column;gap:10px;transition:all 0.3s;overflow:hidden}}
.action-card:hover{{transform:translateY(-2px)}}
.action-card.buy{{border-left:4px solid var(--accent-green)}}
.action-card.hold{{border-left:4px solid var(--accent-yellow)}}
.action-card.sell{{border-left:4px solid var(--accent-red)}}
.action-card .card-header{{display:flex;justify-content:space-between;align-items:center}}
.action-card .ticker{{font-size:20px;font-weight:800;font-family:'JetBrains Mono',monospace}}
.action-badge{{font-size:11px;font-weight:700;text-transform:uppercase;padding:4px 12px;border-radius:20px;letter-spacing:0.5px}}
.action-badge.buy{{background:var(--accent-green-bg);color:var(--accent-green);border:1px solid rgba(16,185,129,0.3)}}
.action-badge.hold{{background:var(--accent-yellow-bg);color:var(--accent-yellow);border:1px solid rgba(245,158,11,0.3)}}
.action-badge.sell{{background:var(--accent-red-bg);color:var(--accent-red);border:1px solid rgba(239,68,68,0.3)}}
.holdings{{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);font-family:'JetBrains Mono',monospace}}
.pnl-positive{{color:var(--accent-green)}}
.pnl-negative{{color:var(--accent-red)}}
.reason{{font-size:13px;color:var(--text-secondary);line-height:1.6;border-top:1px solid var(--border);padding-top:10px}}
.alloc-bar{{height:4px;border-radius:2px;background:rgba(255,255,255,0.06);margin-top:auto}}
.alloc-bar-fill{{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent-blue),var(--accent-purple))}}
.alloc-label{{font-size:11px;color:var(--text-muted);margin-top:2px;font-family:'JetBrains Mono',monospace}}

.picks-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}}
.pick-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;border-left:4px solid var(--accent-purple);transition:all 0.3s}}
.pick-card:hover{{transform:translateY(-2px)}}
.pick-ticker{{font-size:20px;font-weight:800;font-family:'JetBrains Mono',monospace}}
.pick-sector{{font-size:12px;color:var(--accent-purple);font-weight:600;margin-top:2px}}
.pick-reason{{font-size:13px;color:var(--text-secondary);margin-top:8px;line-height:1.6}}

.tsumitate-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.tsumitate-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:20px 24px}}
.tsumitate-card h4{{font-size:15px;font-weight:700;margin-bottom:12px}}
.change-item{{display:flex;align-items:flex-start;gap:8px;font-size:13px;color:var(--text-secondary);margin-bottom:8px;line-height:1.5}}
.change-item .arrow{{color:var(--accent-yellow);font-weight:700;flex-shrink:0}}

.risk-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.risk-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:20px;text-align:center}}
.risk-card.bull{{border-top:3px solid var(--accent-green)}}
.risk-card.base{{border-top:3px solid var(--accent-blue)}}
.risk-card.bear{{border-top:3px solid var(--accent-red)}}
.risk-prob{{font-size:36px;font-weight:900;font-family:'JetBrains Mono',monospace}}
.risk-card.bull .risk-prob{{color:var(--accent-green)}}
.risk-card.base .risk-prob{{color:var(--accent-blue)}}
.risk-card.bear .risk-prob{{color:var(--accent-red)}}
.risk-label{{font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:var(--text-muted);margin-top:4px}}
.risk-desc{{font-size:13px;color:var(--text-secondary);margin-top:10px;line-height:1.5}}

/* Charts */
.chart-container{{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.chart-title{{font-size:16px;font-weight:700;margin-bottom:16px;color:var(--text-primary)}}

/* Meta Prompt */
.prompt-block{{background:#0d1117;border:2px solid var(--accent-purple);border-radius:12px;padding:20px;font-size:12px;font-family:'JetBrains Mono',monospace;color:#c9d1d9;white-space:pre-wrap;word-break:break-all;max-height:500px;overflow-y:auto;line-height:1.6}}

/* Records */
.record-row{{display:flex;gap:16px;align-items:center;padding:12px 16px;background:var(--bg-card);border:1px solid var(--border);border-radius:10px;margin-bottom:8px;font-size:13px}}
.record-date{{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--accent-blue);min-width:90px}}
.record-status{{min-width:30px}}
.record-conf{{color:var(--accent-yellow);font-weight:600;min-width:80px}}
.record-reason{{color:var(--text-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}

/* Metrics Row */
.metrics-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
.metric-box{{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;text-align:center}}
.metric-box .m-label{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px}}
.metric-box .m-value{{font-size:24px;font-weight:800;font-family:'JetBrains Mono',monospace;margin-top:4px}}

.footer{{text-align:center;padding:32px 0 20px;font-size:12px;color:var(--text-muted)}}
.fade-in{{animation:fadeIn 0.6s ease forwards;opacity:0}}
@keyframes fadeIn{{to{{opacity:1}}}}

/* Portfolio Tab */
.port-summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
.port-metric{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;transition:all 0.3s}}
.port-metric:hover{{transform:translateY(-2px);border-color:rgba(59,130,246,0.4)}}
.port-metric-label{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;font-weight:600}}
.port-metric-value{{font-size:26px;font-weight:800;font-family:'JetBrains Mono',monospace;margin-top:6px;color:var(--text-primary)}}
.port-metric-sub{{font-size:12px;color:var(--text-secondary);margin-top:4px}}
.sector-section{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:20px 24px;margin-bottom:20px}}
.sector-title{{font-size:15px;font-weight:700;margin-bottom:14px}}
.sector-bars{{display:flex;flex-direction:column;gap:10px}}
.sector-bar-row{{display:grid;grid-template-columns:140px 1fr 160px;align-items:center;gap:12px;font-size:13px}}
.sector-bar-label{{color:var(--text-secondary);font-weight:600}}
.sector-bar-track{{height:14px;background:rgba(255,255,255,0.05);border-radius:7px;overflow:hidden}}
.sector-bar-fill{{height:100%;border-radius:7px;transition:width 0.6s ease}}
.sector-bar-value{{font-family:'JetBrains Mono',monospace;color:var(--text-primary);text-align:right}}
.sector-pct{{color:var(--text-muted);font-size:11px}}
.port-table-section{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:20px 0 8px;margin-bottom:20px}}
.port-table-title{{font-size:15px;font-weight:700;margin-bottom:12px;padding:0 24px}}
.port-table-wrap{{overflow-x:auto;max-width:100%}}
.port-table{{width:100%;border-collapse:collapse;font-size:13px;min-width:900px}}
.port-table th{{text-align:left;padding:10px 12px;color:var(--text-muted);border-bottom:2px solid var(--border);font-weight:700;text-transform:uppercase;font-size:10px;letter-spacing:0.6px;cursor:pointer;user-select:none;background:rgba(255,255,255,0.02);white-space:nowrap}}
.port-table th:hover{{color:var(--accent-blue)}}
.port-table td{{padding:11px 12px;border-bottom:1px solid var(--border);color:var(--text-secondary);white-space:nowrap}}
.port-table tr:hover{{background:var(--bg-card-hover)}}
.port-table .port-ticker strong{{color:var(--text-primary);font-family:'JetBrains Mono',monospace;font-size:14px}}
.acct-badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;background:rgba(59,130,246,0.15);color:var(--accent-blue);font-weight:600}}
.chart-link{{text-decoration:none;font-size:16px;opacity:0.7;transition:opacity 0.2s}}
.chart-link:hover{{opacity:1}}

/* Advanced Predictor Tab */
.adv-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.adv-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;transition:all 0.3s}}
.adv-card:hover{{transform:translateY(-2px)}}
.adv-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.adv-ticker{{font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--text-primary);text-decoration:none}}
.adv-ticker:hover{{color:var(--accent-blue)}}
.adv-signal{{font-size:12px;font-weight:700;letter-spacing:0.5px}}
.adv-score{{font-size:14px;color:var(--text-secondary);margin-bottom:4px}}
.adv-conf{{font-size:13px;color:var(--text-muted);margin-bottom:12px}}
.adv-subbars{{display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--border);padding-top:12px}}
.adv-subbar{{display:grid;grid-template-columns:60px 60px 1fr;align-items:center;gap:8px;font-size:11px}}
.adv-sublabel{{color:var(--text-muted);font-weight:600}}
.adv-subval{{font-family:'JetBrains Mono',monospace;font-weight:700;text-align:right}}
.adv-bar-track{{height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;position:relative}}
.adv-bar-pos{{height:100%;background:var(--accent-green);border-radius:3px}}
.adv-bar-neg{{height:100%;background:var(--accent-red);border-radius:3px}}

@media(max-width:768px){{
  .stats-row,.metrics-row{{grid-template-columns:repeat(2,1fr)}}
  .risk-grid,.tsumitate-grid{{grid-template-columns:1fr}}
  .action-grid{{grid-template-columns:1fr}}
  .chart-row{{grid-template-columns:1fr}}
  .tab-nav{{gap:2px}}
  .tab-btn{{padding:8px 12px;font-size:12px}}
}}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div>
      <h1>🧬 <span>Evolving Quant</span> Dashboard</h1>
      <p style="color:var(--text-muted);font-size:14px;margin-top:4px">Self-Learning AI Portfolio · 6 Tabs</p>
    </div>
    <div class="header-meta">
      <div class="date">{date_str} 更新</div>
      <div class="confidence"><span class="dot"></span> 確信度 {confidence:.0%}</div>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">ポートフォリオ総額</div>
      <div class="stat-value" style="color:var(--accent-blue)">${total_usd:,.0f}</div>
      <div class="stat-sub">{len(stocks)}銘柄保有</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">含み損益</div>
      <div class="stat-value" style="color:{'var(--accent-green)' if total_pnl >= 0 else 'var(--accent-red)'}">{"+" if total_pnl >= 0 else ""}${total_pnl:,.0f}</div>
      <div class="stat-sub">{pnl_pct:+.1f}%</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">レジーム</div>
      <div class="stat-value" style="font-size:20px;color:var(--accent-yellow)">{regime_label}</div>
      <div class="stat-sub">KL = {kl_val:.4f}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">アクション</div>
      <div class="stat-value" style="font-size:18px">
        <span style="color:var(--accent-green)">{n_buy} BUY</span>
        <span style="color:var(--text-muted)"> · </span>
        <span style="color:var(--accent-yellow)">{n_hold} HOLD</span>
        <span style="color:var(--text-muted)"> · </span>
        <span style="color:var(--accent-red)">{n_sell} SELL</span>
      </div>
      <div class="stat-sub">+{len(latest_record.get("new_picks",[]))} 新規推奨</div>
    </div>
  </div>

  <!-- Tab Navigation -->
  <div class="tab-nav">
    <button class="tab-btn active" data-tab="portfolio" onclick="switchTab(this,'portfolio')">💼 Portfolio</button>
    <button class="tab-btn" data-tab="action" onclick="switchTab(this,'action')">🎯 Action</button>
    <button class="tab-btn" data-tab="advanced" onclick="switchTab(this,'advanced')">🔮 Advanced AI</button>
    <button class="tab-btn" data-tab="overview" onclick="switchTab(this,'overview')">📊 Overview</button>
    <button class="tab-btn" data-tab="risk" onclick="switchTab(this,'risk')">🔥 Risk</button>
    <button class="tab-btn" data-tab="report" onclick="switchTab(this,'report')">🎯 AI Report Card</button>
    <button class="tab-btn" data-tab="prompt" onclick="switchTab(this,'prompt')">🧬 Meta-Prompt</button>
    <button class="tab-btn" data-tab="record" onclick="switchTab(this,'record')">📝 Record</button>
  </div>

  <!-- Tab: Portfolio (default) -->
  <div id="tab-portfolio" class="tab-content active">
    {portfolio_html}
    {performance_chart_html}
  </div>

  <!-- Tab: Advanced AI -->
  <div id="tab-advanced" class="tab-content">
    {advanced_html if advanced_html else '<div class="section"><p style="color:var(--text-muted);text-align:center;padding:40px">高精度予測データがまだありません。次回の <code>daily_evolution.py</code> 実行後に表示されます。</p></div>'}
  </div>

  <!-- Tab: Action -->
  <div id="tab-action" class="tab-content">
    <div class="strategy-banner">
      <strong>🎯 戦略:</strong> {reasoning}
    </div>
    {action_cards_html}
    {tsumi_html}
  </div>

  <!-- Tab: Overview -->
  <div id="tab-overview" class="tab-content">
    <div class="section">
      <div class="section-title"><span class="icon">📊</span> Portfolio Allocation Overview</div>
      <div class="chart-row">
        <div class="chart-container">
          <div class="chart-title">Ensemble配分（ドーナツ）</div>
          <div style="position:relative;height:300px"><canvas id="chartDonut"></canvas></div>
        </div>
        <div class="chart-container">
          <div class="chart-title">戦略比較（棒グラフ）</div>
          <div style="position:relative;height:300px"><canvas id="chartBar"></canvas></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab: Risk -->
  <div id="tab-risk" class="tab-content">
    <div class="section">
      <div class="section-title"><span class="icon">🔥</span> Risk Analysis</div>
      <div class="chart-container">
        <div class="chart-title">リスク寄与度</div>
        <div style="position:relative;height:400px"><canvas id="chartRisk"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Tab: AI Report Card -->
  <div id="tab-report" class="tab-content">
    <div class="section">
      <div class="section-title"><span class="icon">🎯</span> AI Report Card — 予測精度追跡</div>
      <div class="chart-container">
        <div class="chart-title">確信度の推移</div>
        <div style="position:relative;height:250px"><canvas id="chartConfidence"></canvas></div>
      </div>
      <div class="section" style="margin-top:20px">
        <div class="section-title"><span class="icon">📊</span> 評価データ</div>
        <div id="evalData" style="color:var(--text-secondary);font-size:14px;padding:16px;background:var(--bg-card);border-radius:12px;border:1px solid var(--border)"></div>
      </div>
    </div>
  </div>

  <!-- Tab: Meta-Prompt -->
  <div id="tab-prompt" class="tab-content">
    <div class="section">
      <div class="section-title"><span class="icon">🧬</span> Meta-Prompt（Claude Opus用）</div>
      <p style="color:var(--text-secondary);font-size:14px;margin-bottom:16px">以下のプロンプトをコピーしてAIに渡してください。</p>
      <div class="prompt-block" id="promptBlock">{prompt_text}</div>
      <button onclick="navigator.clipboard.writeText(document.getElementById('promptBlock').textContent).then(()=>alert('コピーしました！'))" style="margin-top:12px;padding:10px 24px;background:var(--accent-purple);color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:14px">📋 コピー</button>
    </div>
  </div>

  <!-- Tab: Record -->
  <div id="tab-record" class="tab-content">
    <div class="section">
      <div class="section-title"><span class="icon">📝</span> AI推奨記録（過去10件）</div>
      {records_html if records_html else '<p style="color:var(--text-muted)">まだ記録がありません。</p>'}
    </div>
  </div>

  <div class="footer">
    🧬 Evolving Quant · AI自動分析 · 更新: {date_str} · <a href="https://github.com/akihamada/evolving-quant" style="color:var(--accent-blue)">GitHub</a>
  </div>
</div>

<script>
const overviewData = {overview_json};
const trackData = {track_json};
const perfData = {performance_json};

function switchTab(btn, id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
  if (id === 'overview' && !window._overviewInit) {{ initOverview(); window._overviewInit = true; }}
  if (id === 'portfolio' && !window._perfInit) {{ initPerformance(); window._perfInit = true; }}
  if (id === 'risk' && !window._riskInit) {{ initRisk(); window._riskInit = true; }}
  if (id === 'report' && !window._reportInit) {{ initReport(); window._reportInit = true; }}
}}

function initOverview() {{
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#a855f7','#06b6d4','#ec4899','#84cc16','#f97316','#6366f1','#14b8a6','#e11d48','#8b5cf6','#22d3ee','#eab308','#fb923c','#4ade80','#f472b6'];

  new Chart(document.getElementById('chartDonut'), {{
    type: 'doughnut',
    data: {{
      labels: overviewData.tickers,
      datasets: [{{ data: overviewData.ensemble, backgroundColor: colors, borderWidth: 0 }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
      }}
    }}
  }});

  new Chart(document.getElementById('chartBar'), {{
    type: 'bar',
    data: {{
      labels: overviewData.tickers,
      datasets: [
        {{ label: 'Ensemble', data: overviewData.ensemble, backgroundColor: 'rgba(59,130,246,0.7)' }},
        {{ label: 'Black-Litterman', data: overviewData.bl, backgroundColor: 'rgba(168,85,247,0.7)' }},
        {{ label: 'Equal Weight', data: overviewData.equal, backgroundColor: 'rgba(100,116,139,0.4)' }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }} }},
        y: {{ ticks: {{ color: '#94a3b8', callback: v => v+'%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
    }}
  }});
}}

function initRisk() {{
  new Chart(document.getElementById('chartRisk'), {{
    type: 'bar',
    data: {{
      labels: overviewData.tickers,
      datasets: [{{
        label: 'Ensemble配分 (%)',
        data: overviewData.ensemble,
        backgroundColor: overviewData.ensemble.map(v => v > (100/overviewData.tickers.length) ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.7)')
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8', callback: v => v+'%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y: {{ ticks: {{ color: '#94a3b8', font: {{ family: 'JetBrains Mono', size: 11 }} }} }}
      }},
      plugins: {{ legend: {{ display: false }} }}
    }}
  }});
}}

function initReport() {{
  const recs = trackData.records;
  if (recs.length === 0) {{
    document.getElementById('evalData').innerHTML = '📭 まだ評価データがありません。';
    return;
  }}

  new Chart(document.getElementById('chartConfidence'), {{
    type: 'line',
    data: {{
      labels: recs.map(r => r.date),
      datasets: [{{
        label: '確信度',
        data: recs.map(r => r.confidence * 100),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: '#3b82f6'
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8' }} }},
        y: {{ min: 0, max: 100, ticks: {{ color: '#94a3b8', callback: v => v+'%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
    }}
  }});

  const evals = trackData.evaluations;
  if (evals.length > 0) {{
    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px">';
    html += '<tr style="border-bottom:1px solid #1e293b"><th style="text-align:left;padding:8px;color:#64748b">日付</th><th style="padding:8px;color:#64748b">RMSE</th><th style="padding:8px;color:#64748b">方向性</th><th style="padding:8px;color:#64748b">較正</th></tr>';
    evals.forEach(e => {{
      html += `<tr style="border-bottom:1px solid #1e293b"><td style="padding:8px;color:#94a3b8">${{e.date}}</td><td style="padding:8px;text-align:center;color:#ef4444">${{e.rmse}}</td><td style="padding:8px;text-align:center;color:#10b981">${{e.direction_accuracy}}%</td><td style="padding:8px;text-align:center;color:#a855f7">${{e.calibration_score}}</td></tr>`;
    }});
    html += '</table>';
    document.getElementById('evalData').innerHTML = html;
  }} else {{
    document.getElementById('evalData').innerHTML = '📭 まだ採点データがありません。AI推奨を記録し、7日以上経過すると自動採点されます。';
  }}
}}

// Portfolio table sort
let portSortState = {{col: -1, asc: false}};
function sortPortTable(colIdx, type) {{
  const table = document.getElementById('portTable');
  if (!table) return;
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  portSortState.asc = (portSortState.col === colIdx) ? !portSortState.asc : false;
  portSortState.col = colIdx;
  rows.sort((a, b) => {{
    let av, bv;
    if (type === 'num') {{
      av = parseFloat(a.cells[colIdx].getAttribute('data-num') || '0');
      bv = parseFloat(b.cells[colIdx].getAttribute('data-num') || '0');
    }} else {{
      av = a.cells[colIdx].innerText.trim();
      bv = b.cells[colIdx].innerText.trim();
    }}
    if (av < bv) return portSortState.asc ? -1 : 1;
    if (av > bv) return portSortState.asc ? 1 : -1;
    return 0;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// Performance history chart
function initPerformance() {{
  const canvas = document.getElementById('chartHistory');
  if (!canvas || !perfData.dates || perfData.dates.length < 2) return;
  new Chart(canvas, {{
    type: 'line',
    data: {{
      labels: perfData.dates,
      datasets: [
        {{
          label: 'ポートフォリオ総額 (USD)',
          data: perfData.values_usd,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.1)',
          fill: true,
          tension: 0.3,
          yAxisID: 'y',
          pointRadius: 2,
        }},
        {{
          label: '含み損益 (USD)',
          data: perfData.pnl_usd,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.1)',
          fill: false,
          tension: 0.3,
          yAxisID: 'y1',
          pointRadius: 2,
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
        y: {{ position: 'left', ticks: {{ color: '#94a3b8', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y1: {{ position: 'right', ticks: {{ color: '#10b981', callback: v => '$'+v.toLocaleString() }}, grid: {{ display: false }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
    }}
  }});
}}

// Auto-init Performance chart if Portfolio tab is default
document.addEventListener('DOMContentLoaded', () => {{
  if (document.getElementById('tab-portfolio').classList.contains('active') && !window._perfInit) {{
    initPerformance();
    window._perfInit = true;
  }}
}});
</script>
</body>
</html>'''


def main() -> None:
    """メインエントリーポイント。"""
    logger.info("📊 ダッシュボード生成開始...")

    results = load_json(RESULTS_PATH)
    track = load_json(TRACK_RECORD_PATH)
    holdings = load_json(HOLDINGS_PATH)

    if not results:
        logger.error("latest_evolution_results.json が見つかりません")
        return

    html = generate_html(results, track, holdings)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    logger.info("✅ %s を生成しました（%d bytes）", OUTPUT_PATH, len(html))


if __name__ == "__main__":
    main()
