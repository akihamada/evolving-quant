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
    colors = ["#7170ff", "#a78bfa", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899", "#84cc16"]
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


def build_learning_html(results: dict) -> str:
    """🧠 Learning タブ: 予測精度の推移 + 学習ジャーナル."""
    learning = results.get("learning_summary", {}) or {}

    # 学習ジャーナルを直接読み込む（過去エントリ用）
    journal_path = BASE_DIR / "data" / "learning_journal.json"
    journal_entries = []
    if journal_path.exists():
        try:
            with open(journal_path, encoding="utf-8") as f:
                journal_entries = json.load(f).get("entries", [])
        except (json.JSONDecodeError, OSError):
            journal_entries = []

    if not learning and not journal_entries:
        return ('<div class="section"><p style="color:var(--text-muted);text-align:center;padding:40px">'
                '学習データがまだありません。予測履歴が7日以上蓄積されると自動評価が始まります。</p></div>')

    html = '<div class="section"><div class="section-title"><span class="icon">🧠</span> 自己学習サイクル</div>'

    # サマリーメトリクス
    acc_recent = (learning.get("accuracy_recent", 0) or 0) * 100
    acc_30 = (learning.get("accuracy_30d", 0) or 0) * 100
    n_total = learning.get("n_total_evaluated", 0)
    n_today = learning.get("n_evaluated", 0)

    html += '<div class="port-summary-grid">'
    html += f'<div class="port-metric"><div class="port-metric-label">直近方向精度</div>'
    html += f'<div class="port-metric-value">{acc_recent:.0f}%</div>'
    html += f'<div class="port-metric-sub">直近30件</div></div>'

    html += f'<div class="port-metric"><div class="port-metric-label">30日精度</div>'
    html += f'<div class="port-metric-value">{acc_30:.0f}%</div>'
    html += f'<div class="port-metric-sub">直近100件</div></div>'

    html += f'<div class="port-metric"><div class="port-metric-label">累計評価</div>'
    html += f'<div class="port-metric-value">{n_total}</div>'
    html += f'<div class="port-metric-sub">本日 +{n_today} 新規</div></div>'

    html += f'<div class="port-metric"><div class="port-metric-label">ジャーナル</div>'
    html += f'<div class="port-metric-value">{len(journal_entries)}</div>'
    html += f'<div class="port-metric-sub">日次学習エントリ</div></div>'
    html += '</div>'

    # サブ予測器ウェイト
    sub_stats = {}
    if journal_entries:
        latest_entry = journal_entries[-1]
        sub_stats = latest_entry.get("sub_predictor_stats", {})

    if sub_stats:
        html += '<div class="sector-section"><div class="sector-title">🔬 サブ予測器の学習状態</div>'
        html += '<div class="sector-bars">'
        for name, s in sub_stats.items():
            acc = s.get("accuracy", 0) * 100
            weight = s.get("weight", 0)
            delta = s.get("weight_delta", 0)
            ev = s.get("evaluated", 0)
            delta_str = f'<span style="color:var(--accent-green)">+{delta:.2f}</span>' if delta > 0 else \
                        f'<span style="color:var(--accent-red)">{delta:.2f}</span>' if delta < 0 else \
                        f'<span style="color:var(--text-muted)">±0</span>'
            bar_pct = min(100, weight * 100)
            html += f'<div class="sector-bar-row">'
            html += f'<div class="sector-bar-label">{name}</div>'
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{bar_pct:.0f}%;background:var(--accent-bright)"></div></div>'
            html += f'<div class="sector-bar-value">精度 {acc:.0f}% / W {weight:.2f} {delta_str}<span class="sector-pct"> ({ev}件)</span></div>'
            html += '</div>'
        html += '</div></div>'

    # 最新の notable findings
    findings = learning.get("notable_findings", [])
    if findings:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">💡</span> 本日の学び</div>'
        html += '<div class="action-grid" style="grid-template-columns:1fr">'
        for f in findings:
            html += f'<div class="action-card" style="border-left:2px solid var(--accent-bright)">'
            html += f'<div class="reason" style="border-top:none;padding-top:0">{html_mod.escape(f)}</div>'
            html += '</div>'
        html += '</div>'

    # 過去ジャーナル（7日分）
    if len(journal_entries) > 1:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">📓</span> 学習ジャーナル（直近7日）</div>'
        html += '<div class="port-table-section"><div class="port-table-wrap"><table class="port-table">'
        html += '<thead><tr><th>日付</th><th>評価件数</th><th>直近精度</th><th>30日精度</th><th>主な学び</th></tr></thead><tbody>'
        for e in reversed(journal_entries[-7:]):
            acc_r = (e.get("accuracy_recent", 0) or 0) * 100
            acc_30 = (e.get("accuracy_30d", 0) or 0) * 100
            n_ev = e.get("n_evaluated_today", 0)
            ns = e.get("notable_findings", [])
            finding_text = ns[0][:60] + "…" if ns and len(ns[0]) > 60 else (ns[0] if ns else "—")
            html += f'<tr><td>{e.get("date", "")}</td><td>+{n_ev}</td>'
            html += f'<td>{acc_r:.0f}%</td><td>{acc_30:.0f}%</td>'
            html += f'<td style="white-space:normal;max-width:420px">{html_mod.escape(finding_text)}</td></tr>'
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
    # 新旧スキーマ両対応: total_market_value (current) / grand_total_usd (legacy)
    total_usd = summary.get("total_market_value", summary.get("grand_total_usd", 0))
    total_pnl = summary.get("total_unrealized_pnl", summary.get("grand_total_pnl_usd", 0))
    # summary が 0/欠損の場合は実保有から計算（ロバスト fallback）
    if not total_usd or total_usd == 0:
        total_usd = 0.0
        total_pnl = 0.0
        for acct in ["tokutei", "nisa"]:
            for s in holdings.get("us_stocks", {}).get(acct, []):
                shares = s.get("shares", 0) or 0
                price = float(s.get("current_price_usd", 0) or 0)
                cost = float(s.get("cost_basis_usd", 0) or 0)
                mv = s.get("market_value_usd", price * shares)
                pnl = s.get("unrealized_pnl_usd", (price - cost) * shares)
                total_usd += mv
                total_pnl += pnl
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
    learning_html = build_learning_html(results)
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
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- Inter Variable (Linear signature font) + Noto Sans JP (JP fallback) + JetBrains Mono (Berkeley Mono代替) -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;510;590;600&family=Noto+Sans+JP:wght@300;400;500;510;590;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{
  /* === Linear Dark-mode-native Backgrounds === */
  --bg-primary:    #08090a;    /* Marketing black / deepest canvas */
  --bg-panel:      #0f1011;    /* Panel / sidebar */
  --bg-elevated:   #191a1b;    /* Elevated surfaces / cards / dropdowns */
  --bg-secondary:  #28282c;    /* Hover / slightly elevated */
  --bg-card:       rgba(255,255,255,0.02);   /* 透過カード */
  --bg-card-hover: rgba(255,255,255,0.04);   /* ホバー */
  --bg-button:     rgba(255,255,255,0.02);
  --bg-button-hover: rgba(255,255,255,0.05);

  /* === Linear Text Hierarchy === */
  --text-primary:   #f7f8f8;    /* Near-white (not pure) */
  --text-secondary: #d0d6e0;    /* Cool silver-gray */
  --text-muted:     #8a8f98;    /* Tertiary gray */
  --text-subtle:    #62666d;    /* Quaternary / disabled */

  /* === Semi-transparent white borders (Linear signature) === */
  --border:         rgba(255,255,255,0.08);   /* Standard */
  --border-subtle:  rgba(255,255,255,0.05);   /* Ultra subtle */
  --border-solid:   #23252a;                   /* Solid when needed */
  --line-tint:      #141516;

  /* === Brand: Indigo-violet (唯一の有彩色) === */
  --accent:         #5e6ad2;    /* Brand indigo (bg) */
  --accent-bright:  #7170ff;    /* Accent violet (interactive) */
  --accent-hover:   #828fff;    /* Lighter hover */
  --accent-soft:    rgba(94, 106, 210, 0.12);
  --accent-soft-hover: rgba(94, 106, 210, 0.2);

  /* === Status Colors (sparingly used) === */
  --color-success:  #10b981;    /* Emerald */
  --color-success-alt: #27a644; /* Green */
  --color-error:    #ef4444;    /* Red (loss / sell) */
  --color-warn:     #f59e0b;

  /* === P&L / chart semantic === */
  --accent-green:   #10b981;
  --accent-green-bg:rgba(16, 185, 129, 0.12);
  --accent-red:     #ef4444;
  --accent-red-bg:  rgba(239, 68, 68, 0.12);
  --accent-yellow:  #f59e0b;
  --accent-yellow-bg:rgba(245, 158, 11, 0.12);
  --accent-blue:    #7170ff;
  --accent-blue-bg: rgba(113, 112, 255, 0.12);
  --accent-purple:  #a78bfa;
  --accent-purple-bg:rgba(167, 139, 250, 0.12);

  /* === Radius === */
  --radius-sm:     2px;
  --radius-md:     4px;
  --radius-lg:     6px;
  --radius-xl:     8px;
  --radius-panel:  12px;
  --radius-pill:   9999px;
  --radius-circle: 50%;

  /* === Spacing (8px base) === */
  --space-half: 4px;
  --space-1: 8px;  --space-2: 12px; --space-3: 16px;
  --space-4: 24px; --space-5: 32px; --space-6: 48px;
  --space-7: 64px; --space-8: 96px;

  /* === Linear Shadow: Luminance stacking + ring borders === */
  --shadow-ring:    rgba(0,0,0,0.2) 0px 0px 0px 1px;
  --shadow-inset:   rgba(0,0,0,0.2) 0px 0px 12px 0px inset;
  --shadow-elevated: rgba(0,0,0,0.4) 0px 2px 4px;
  --shadow-dialog:  rgba(0,0,0,0) 0px 8px 2px, rgba(0,0,0,0.01) 0px 5px 2px, rgba(0,0,0,0.04) 0px 3px 2px, rgba(0,0,0,0.07) 0px 1px 1px, rgba(0,0,0,0.08) 0px 0px 1px;
  --shadow-focus:   rgba(0,0,0,0.1) 0px 4px 12px, rgba(94, 106, 210, 0.5) 0px 0px 0px 2px;

  /* === Transitions === */
  --transition-color:   150ms ease;
  --transition-shadow:  200ms ease;
  --transition-bg:      150ms ease;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
html{{text-size-adjust:100%;font-kerning:auto}}

/* === jp-ui-contracts: Japanese line-breaking === */
html:lang(ja),
html{{
  line-break:strict;
  word-break:normal;
  overflow-wrap:anywhere;
}}
/* 見出しは自然な文節で改行 */
h1:lang(ja), h2:lang(ja), h3:lang(ja),
.section-title, .header h1{{
  word-break:auto-phrase;
}}

body{{
  /* Inter Variable (Linear) + 日本語 fallback 必須明示 */
  font-family:'Inter','Noto Sans JP','Hiragino Sans','Yu Gothic UI','SF Pro Display',-apple-system,system-ui,sans-serif;
  /* Linear の cv01 (single-story a) + ss03 OpenType features */
  font-feature-settings:'cv01','ss03';
  background:var(--bg-primary);
  color:var(--text-primary);
  min-height:100vh;
  line-height:1.6;                  /* Linear Body: 1.5-1.6 / JP ≥1.5 遵守 */
  font-weight:400;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}}
/* Linear の 510 signature weight をUIデフォルトに */
.ui-emphasize,
.section-title,
.tab-btn,
.stat-label,
.port-metric-label,
.port-table th,
.acct-badge,
.action-badge,
.adv-signal{{
  font-weight:510;
}}
.container{{max-width:1440px;margin:0 auto;padding:var(--space-5)}}

/* Focus-visible: Linear indigo ring */
*:focus{{outline:none}}
*:focus-visible{{
  outline:2px solid var(--accent-bright);
  outline-offset:2px;
  border-radius:var(--radius-md);
}}
button:focus-visible,
a:focus-visible{{
  outline:2px solid var(--accent-bright);
  outline-offset:2px;
}}
/* Selection color: Linear accent */
::selection{{background:var(--accent-soft);color:var(--text-primary)}}

/* Scrollbar (webkit) — thin Linear style */
::-webkit-scrollbar{{width:10px;height:10px}}
::-webkit-scrollbar-track{{background:var(--bg-panel)}}
::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.08);border-radius:var(--radius-pill);border:2px solid var(--bg-panel)}}
::-webkit-scrollbar-thumb:hover{{background:rgba(255,255,255,0.15)}}

/* Header — Linear dark panel */
.header{{
  display:flex;justify-content:space-between;align-items:center;
  padding:var(--space-4) var(--space-5);
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:var(--radius-panel);
  margin-bottom:var(--space-4);
}}
.header h1{{
  font-family:inherit;              /* Inter Variable */
  font-size:32px;
  font-weight:510;                  /* Linear signature */
  letter-spacing:-0.022em;           /* Progressive: ~-0.704px at 32px */
  color:var(--text-primary);
  line-height:1.13;
}}
.header h1 span{{
  color:var(--accent-bright);
  transition:color var(--transition-color);
}}
.header h1 span{{color:var(--accent);font-style:italic}}
.header-meta{{text-align:right}}
.header-meta .date{{
  color:var(--text-muted);
  font-size:13px;
  font-family:'JetBrains Mono',monospace;
}}
.header-meta .confidence{{
  display:inline-flex;align-items:center;gap:var(--space-half);
  background:var(--accent-soft);
  border:1px solid rgba(94, 106, 210, 0.3);
  border-radius:var(--radius-pill);
  padding:3px 10px;
  font-size:12px;font-weight:510;
  color:var(--accent-bright);
  margin-top:var(--space-1);
  font-family:'JetBrains Mono',monospace;
}}
.header-meta .confidence .dot{{width:6px;height:6px;border-radius:50%;background:var(--accent-bright);animation:pulse 2.4s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}

/* Stats — Linear surface cards */
.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-2);margin-bottom:var(--space-4)}}
.stat-card{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3);
  transition:background var(--transition-bg), border-color var(--transition-color);
}}
.stat-card:hover{{
  background:var(--bg-card-hover);
  border-color:rgba(255,255,255,0.12);
}}
.stat-label{{
  font-size:11px;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:0.08em;font-weight:510;
  font-family:inherit;
}}
.stat-value{{
  font-size:28px;font-weight:510;margin-top:var(--space-1);
  font-family:inherit;
  color:var(--text-primary);
  letter-spacing:-0.022em;     /* -0.616px at 28px */
  font-variant-numeric:tabular-nums;
}}
.stat-sub{{font-size:13px;color:var(--text-muted);margin-top:var(--space-half);line-height:1.5}}

/* Tabs — Linear subtle segmented */
.tab-nav{{
  display:flex;gap:2px;
  margin-bottom:var(--space-4);
  overflow-x:auto;
  padding:var(--space-half);
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:var(--radius-xl);
}}
.tab-btn{{
  padding:6px 14px;background:transparent;border:none;
  border-radius:var(--radius-lg);
  color:var(--text-muted);cursor:pointer;
  font-size:13px;font-weight:510;white-space:nowrap;
  font-family:inherit;
  letter-spacing:-0.01em;
  transition:color var(--transition-color), background var(--transition-bg);
}}
.tab-btn:hover{{background:var(--bg-button-hover);color:var(--text-primary)}}
.tab-btn.active{{background:var(--accent);color:#fff;font-weight:590}}
.tab-btn.active:hover{{color:#fff;background:var(--accent-hover)}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}

/* Strategy Banner — Linear accent-left */
.strategy-banner{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-left:2px solid var(--accent-bright);
  border-radius:var(--radius-xl);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-4);
  font-size:14px;
  color:var(--text-secondary);
  line-height:1.6;
}}
.strategy-banner strong{{color:var(--text-primary);font-weight:510}}

/* Sections & Cards */
.section{{margin-bottom:var(--space-5)}}
.section-title{{
  font-family:inherit;              /* Inter Variable (not serif) */
  font-size:20px;font-weight:510;
  margin-bottom:var(--space-3);
  display:flex;align-items:center;gap:var(--space-1);
  color:var(--text-primary);
  letter-spacing:-0.012em;           /* -0.24px at 20px */
  line-height:1.33;
}}
.section-title .icon{{font-size:20px}}

.action-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:var(--space-2)}}
.action-card{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3) var(--space-4);
  display:flex;flex-direction:column;gap:var(--space-1);
  transition:background var(--transition-bg), border-color var(--transition-color);
}}
.action-card:hover{{
  background:var(--bg-card-hover);
  border-color:rgba(255,255,255,0.12);
}}
.action-card.buy{{border-left:2px solid var(--accent-green)}}
.action-card.hold{{border-left:2px solid var(--accent-yellow)}}
.action-card.sell{{border-left:2px solid var(--accent-red)}}
.action-card .card-header{{display:flex;justify-content:space-between;align-items:center}}
.action-card .ticker{{
  font-size:18px;font-weight:590;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-primary);
  letter-spacing:-0.013em;
}}
.action-badge{{
  font-size:10px;font-weight:510;
  text-transform:uppercase;letter-spacing:0.06em;
  padding:2px 10px;
  border-radius:var(--radius-pill);
  font-family:inherit;
}}
.action-badge.buy{{background:var(--accent-green-bg);color:var(--accent-green);border:1px solid rgba(16, 185, 129, 0.25)}}
.action-badge.hold{{background:var(--accent-yellow-bg);color:var(--accent-yellow);border:1px solid rgba(245, 158, 11, 0.25)}}
.action-badge.sell{{background:var(--accent-red-bg);color:var(--accent-red);border:1px solid rgba(239, 68, 68, 0.25)}}
.holdings{{
  display:flex;gap:var(--space-3);
  font-size:12px;color:var(--text-muted);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:-0.01em;
}}
.pnl-positive{{color:var(--accent-green)}}
.pnl-negative{{color:var(--accent-red)}}
.reason{{
  font-size:13px;color:var(--text-secondary);
  line-height:1.6;
  border-top:1px solid var(--border);
  padding-top:var(--space-1);
  word-break:normal;overflow-wrap:anywhere;
}}
.alloc-bar{{height:2px;border-radius:var(--radius-pill);background:rgba(255,255,255,0.05);margin-top:auto;overflow:hidden}}
.alloc-bar-fill{{height:100%;border-radius:var(--radius-pill);background:var(--accent-bright)}}
.alloc-label{{
  font-size:11px;color:var(--text-subtle);
  margin-top:var(--space-half);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:0.02em;
}}

.picks-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:var(--space-2)}}
.pick-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3) var(--space-4);
  border-left:2px solid var(--accent-bright);
  transition:background var(--transition-bg), border-color var(--transition-color);
}}
.pick-card:hover{{
  background:var(--bg-card-hover);
  border-color:rgba(255,255,255,0.12);
  border-left-color:var(--accent-hover);
}}
.pick-ticker{{font-size:18px;font-weight:590;font-family:'JetBrains Mono',monospace;color:var(--text-primary);letter-spacing:-0.013em}}
.pick-sector{{
  display:inline-block;margin-top:var(--space-half);
  font-size:10px;color:var(--accent-bright);
  font-family:inherit;
  font-weight:510;letter-spacing:0.06em;
  text-transform:uppercase;
  padding:2px 8px;
  background:var(--accent-soft);
  border-radius:var(--radius-pill);
}}
.pick-reason{{font-size:13px;color:var(--text-secondary);margin-top:var(--space-1);line-height:1.6}}

.tsumitate-grid{{display:grid;grid-template-columns:1fr 1fr;gap:var(--space-3)}}
.tsumitate-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-4) var(--space-5);
}}
.tsumitate-card h4{{
  font-family:'Noto Serif JP',serif;
  font-size:15px;font-weight:600;
  margin-bottom:var(--space-3);color:var(--text-primary);
}}
.change-item{{
  display:flex;align-items:flex-start;gap:var(--space-2);
  font-size:13px;color:var(--text-secondary);
  margin-bottom:var(--space-2);line-height:1.7;
}}
.change-item .arrow{{color:var(--accent);font-weight:600;flex-shrink:0}}

.risk-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--space-2)}}
.risk-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-4) var(--space-3);
  text-align:center;
  transition:background var(--transition-bg);
}}
.risk-card:hover{{background:var(--bg-card-hover)}}
.risk-card.bull{{border-top:2px solid var(--accent-green)}}
.risk-card.base{{border-top:2px solid var(--accent-bright)}}
.risk-card.bear{{border-top:2px solid var(--accent-red)}}
.risk-prob{{
  font-size:36px;font-weight:510;
  font-family:inherit;
  letter-spacing:-0.022em;
  font-variant-numeric:tabular-nums;
}}
.risk-card.bull .risk-prob{{color:var(--accent-green)}}
.risk-card.base .risk-prob{{color:var(--accent-bright)}}
.risk-card.bear .risk-prob{{color:var(--accent-red)}}
.risk-label{{
  font-size:11px;text-transform:uppercase;
  letter-spacing:0.08em;font-weight:500;
  color:var(--text-muted);margin-top:var(--space-1);
  font-family:'JetBrains Mono',monospace;
}}
.risk-desc{{font-size:13px;color:var(--text-secondary);margin-top:var(--space-2);line-height:1.7}}

/* Charts */
.chart-container{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-4);margin-bottom:var(--space-3);
}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:var(--space-3)}}
.chart-title{{
  font-family:inherit;
  font-size:15px;font-weight:590;
  margin-bottom:var(--space-3);color:var(--text-primary);
  letter-spacing:-0.011em;
}}

/* Meta Prompt */
.prompt-block{{
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3);
  font-size:12px;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-secondary);
  white-space:pre-wrap;
  word-break:normal;overflow-wrap:anywhere;
  max-height:500px;overflow-y:auto;
  line-height:1.6;
  letter-spacing:-0.01em;
}}

/* Records */
.record-row{{
  display:flex;gap:var(--space-3);align-items:center;
  padding:var(--space-2) var(--space-3);
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  margin-bottom:var(--space-1);font-size:13px;
  transition:background var(--transition-bg);
}}
.record-row:hover{{background:var(--bg-card-hover)}}
.record-date{{
  font-family:'JetBrains Mono',monospace;
  font-weight:510;color:var(--accent-bright);
  min-width:100px;letter-spacing:-0.01em;
}}
.record-status{{min-width:30px}}
.record-conf{{color:var(--text-secondary);font-weight:510;min-width:90px;font-family:'JetBrains Mono',monospace}}
.record-reason{{color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;line-height:1.6}}

/* Metrics Row */
.metrics-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
.metric-box{{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;text-align:center}}
.metric-box .m-label{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;font-family:'JetBrains Mono',monospace}}
.metric-box .m-value{{font-size:24px;font-weight:800;font-family:'JetBrains Mono',monospace;margin-top:4px}}

.footer{{
  text-align:center;padding:var(--space-6) 0 var(--space-5);
  font-size:12px;color:var(--text-muted);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:0.02em;
  line-height:1.8;
}}
.footer a{{color:var(--accent-bright);text-decoration:none;transition:color var(--transition-color)}}
.footer a:hover{{color:var(--accent-hover);text-decoration:underline}}
.fade-in{{animation:fadeIn 0.4s ease forwards;opacity:0}}
@keyframes fadeIn{{to{{opacity:1}}}}

/* Portfolio Tab — Linear dense dashboard */
.port-summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-2);margin-bottom:var(--space-4)}}
.port-metric{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3);
  transition:background var(--transition-bg), border-color var(--transition-color);
}}
.port-metric:hover{{
  background:var(--bg-card-hover);
  border-color:rgba(255,255,255,0.12);
}}
.port-metric-label{{
  font-size:11px;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:0.08em;font-weight:510;
  font-family:inherit;
}}
.port-metric-value{{
  font-size:26px;font-weight:510;
  font-family:inherit;
  margin-top:var(--space-1);
  color:var(--text-primary);
  letter-spacing:-0.022em;
  font-variant-numeric:tabular-nums;
}}
.port-metric-sub{{font-size:12px;color:var(--text-muted);margin-top:var(--space-half);line-height:1.5;font-variant-numeric:tabular-nums}}

.sector-section{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-3);
}}
.sector-title{{
  font-family:inherit;
  font-size:15px;font-weight:590;
  margin-bottom:var(--space-3);color:var(--text-primary);
  letter-spacing:-0.011em;
}}
.sector-bars{{display:flex;flex-direction:column;gap:var(--space-1)}}
.sector-bar-row{{
  display:grid;grid-template-columns:140px 1fr 160px;
  align-items:center;gap:var(--space-2);font-size:13px;
}}
.sector-bar-label{{color:var(--text-secondary);font-weight:510}}
.sector-bar-track{{
  height:6px;background:rgba(255,255,255,0.04);
  border-radius:var(--radius-pill);overflow:hidden;
}}
.sector-bar-fill{{height:100%;border-radius:var(--radius-pill);transition:width 0.6s ease}}
.sector-bar-value{{
  font-family:'JetBrains Mono',monospace;
  color:var(--text-primary);text-align:right;
  letter-spacing:-0.01em;
  font-variant-numeric:tabular-nums;
}}
.sector-pct{{color:var(--text-muted);font-size:11px;font-family:'JetBrains Mono',monospace}}

.port-table-section{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3) 0 var(--space-1);
  margin-bottom:var(--space-3);
}}
.port-table-title{{
  font-family:inherit;
  font-size:15px;font-weight:590;
  margin-bottom:var(--space-2);
  padding:0 var(--space-4);
  color:var(--text-primary);
  letter-spacing:-0.011em;
}}
.port-table-wrap{{overflow-x:auto;max-width:100%}}
.port-table{{width:100%;border-collapse:collapse;font-size:13px;min-width:900px}}
.port-table th{{
  text-align:left;padding:var(--space-1) var(--space-2);
  color:var(--text-muted);
  border-bottom:1px solid var(--border);
  font-weight:510;text-transform:uppercase;
  font-size:10px;letter-spacing:0.06em;
  cursor:pointer;user-select:none;
  background:var(--bg-panel);
  font-family:inherit;
  white-space:nowrap;
  transition:color var(--transition-color);
}}
.port-table th:hover{{color:var(--text-primary)}}
.port-table td{{
  padding:var(--space-1) var(--space-2);
  border-bottom:1px solid var(--border-subtle);
  color:var(--text-secondary);
  white-space:nowrap;
  font-family:'JetBrains Mono',monospace;
  letter-spacing:-0.005em;
  font-variant-numeric:tabular-nums;
}}
.port-table tr:hover{{background:var(--bg-card-hover)}}
.port-table .port-ticker strong{{color:var(--text-primary);font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:590}}
.acct-badge{{
  display:inline-block;padding:1px 8px;
  border-radius:var(--radius-pill);
  font-size:10px;
  background:var(--accent-soft);
  color:var(--accent-bright);
  font-weight:510;
  font-family:inherit;
  letter-spacing:0.04em;
  text-transform:uppercase;
}}
.chart-link{{text-decoration:none;font-size:14px;opacity:0.55;transition:opacity 0.2s}}
.chart-link:hover{{opacity:1}}

/* Advanced Predictor Tab — Linear data cards */
.adv-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:var(--space-2)}}
.adv-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-xl);
  padding:var(--space-3) var(--space-4);
  transition:background var(--transition-bg), border-color var(--transition-color);
}}
.adv-card:hover{{
  background:var(--bg-card-hover);
  border-color:rgba(255,255,255,0.12);
}}
.adv-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--space-1)}}
.adv-ticker{{
  font-size:17px;font-weight:590;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-primary);
  text-decoration:none;
  letter-spacing:-0.013em;
  transition:color var(--transition-color);
}}
.adv-ticker:hover{{color:var(--accent-bright)}}
.adv-signal{{
  font-size:10px;font-weight:510;
  letter-spacing:0.06em;
  font-family:inherit;
  padding:2px 10px;
  border-radius:var(--radius-pill);
  text-transform:uppercase;
}}
.adv-score{{font-size:13px;color:var(--text-secondary);margin-bottom:var(--space-half);font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums}}
.adv-conf{{font-size:12px;color:var(--text-muted);margin-bottom:var(--space-2);font-family:'JetBrains Mono',monospace}}
.adv-subbars{{
  display:flex;flex-direction:column;gap:var(--space-half);
  border-top:1px solid var(--border);
  padding-top:var(--space-2);
}}
.adv-subbar{{
  display:grid;grid-template-columns:60px 60px 1fr;
  align-items:center;gap:var(--space-1);font-size:11px;
}}
.adv-sublabel{{color:var(--text-muted);font-weight:510;font-family:inherit;letter-spacing:0.02em;text-transform:uppercase}}
.adv-subval{{font-family:'JetBrains Mono',monospace;font-weight:510;text-align:right;letter-spacing:-0.01em;font-variant-numeric:tabular-nums}}
.adv-bar-track{{
  height:4px;background:rgba(255,255,255,0.04);
  border-radius:var(--radius-pill);overflow:hidden;
}}
.adv-bar-pos{{height:100%;background:var(--accent-green);border-radius:var(--radius-pill)}}
.adv-bar-neg{{height:100%;background:var(--accent-red);border-radius:var(--radius-pill)}}

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
    <button class="tab-btn" data-tab="learning" onclick="switchTab(this,'learning')">🧠 Learning</button>
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

  <!-- Tab: Learning -->
  <div id="tab-learning" class="tab-content">
    {learning_html}
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
  const colors = ['#7170ff','#10b981','#f59e0b','#ef4444','#a78bfa','#06b6d4','#ec4899','#84cc16','#f97316','#6366f1','#14b8a6','#e11d48','#8b5cf6','#22d3ee','#eab308','#fb923c','#4ade80','#f472b6'];

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
        legend: {{ position: 'right', labels: {{ color: '#8a8f98', font: {{ size: 11 }} }} }}
      }}
    }}
  }});

  new Chart(document.getElementById('chartBar'), {{
    type: 'bar',
    data: {{
      labels: overviewData.tickers,
      datasets: [
        {{ label: 'Ensemble', data: overviewData.ensemble, backgroundColor: 'rgba(113,112,255,0.75)' }},
        {{ label: 'Black-Litterman', data: overviewData.bl, backgroundColor: 'rgba(167,139,250,0.7)' }},
        {{ label: 'Equal Weight', data: overviewData.equal, backgroundColor: 'rgba(138,143,152,0.3)' }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#8a8f98', font: {{ size: 10 }} }} }},
        y: {{ ticks: {{ color: '#8a8f98', callback: v => v+'%' }}, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#8a8f98' }} }} }}
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
        backgroundColor: overviewData.ensemble.map(v => v > (100/overviewData.tickers.length) ? 'rgba(239,68,68,0.75)' : 'rgba(16,185,129,0.75)')
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {{
        x: {{ ticks: {{ color: '#8a8f98', callback: v => v+'%' }}, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }},
        y: {{ ticks: {{ color: '#8a8f98', font: {{ family: 'JetBrains Mono', size: 11 }} }} }}
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
        borderColor: '#7170ff',
        backgroundColor: 'rgba(113,112,255,0.12)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: '#7170ff'
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#8a8f98' }} }},
        y: {{ min: 0, max: 100, ticks: {{ color: '#8a8f98', callback: v => v+'%' }}, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#8a8f98' }} }} }}
    }}
  }});

  const evals = trackData.evaluations;
  if (evals.length > 0) {{
    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px">';
    html += '<tr style="border-bottom:1px solid var(--border)"><th style="text-align:left;padding:8px;color:var(--text-muted)">日付</th><th style="padding:8px;color:var(--text-muted)">RMSE</th><th style="padding:8px;color:var(--text-muted)">方向性</th><th style="padding:8px;color:var(--text-muted)">較正</th></tr>';
    evals.forEach(e => {{
      html += `<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px;color:var(--text-secondary);font-family:'JetBrains Mono',monospace">${{e.date}}</td><td style="padding:8px;text-align:center;color:#ef4444;font-family:'JetBrains Mono',monospace">${{e.rmse}}</td><td style="padding:8px;text-align:center;color:#10b981;font-family:'JetBrains Mono',monospace">${{e.direction_accuracy}}%</td><td style="padding:8px;text-align:center;color:#7170ff;font-family:'JetBrains Mono',monospace">${{e.calibration_score}}</td></tr>`;
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
          borderColor: '#7170ff',
          backgroundColor: 'rgba(113,112,255,0.12)',
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
        x: {{ ticks: {{ color: '#8a8f98', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
        y: {{ position: 'left', ticks: {{ color: '#8a8f98', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }},
        y1: {{ position: 'right', ticks: {{ color: '#10b981', callback: v => '$'+v.toLocaleString() }}, grid: {{ display: false }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#8a8f98' }} }} }}
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
