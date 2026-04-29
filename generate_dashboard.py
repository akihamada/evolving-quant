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
    """portfolio_holdings.json から表示用データを構築する。

    米国株 (us_stocks: tokutei / nisa) と日本株 (japan_stocks: 全サブセクション) を統合。
    日本株は ticker を "<code>.T" 形式に揃える (config の actions マップと一致させる)。
    """
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

    # 日本株: 全サブセクションを読む (nisa_growth / nisa_tsumitate / tokutei 等が将来追加されても拾う)
    fx = float(holdings.get("metadata", {}).get("fx_rate", {}).get("USD_JPY", 150)) or 150
    jp_section = holdings.get("japan_stocks", {}) or {}
    for sub_key, positions in jp_section.items():
        if not isinstance(positions, list):
            continue
        for s in positions:
            code = s.get("code", "") or s.get("ticker", "")
            if not code:
                continue
            ticker = code if code.endswith(".T") else f"{code}.T"
            shares = s.get("shares", 0) or 0
            cost_jpy = float(s.get("cost_basis_jpy", 0) or 0)
            price_jpy = float(s.get("current_price_jpy", 0) or 0)
            pnl_jpy = (price_jpy - cost_jpy) * shares if cost_jpy > 0 else float(s.get("unrealized_pnl_jpy", 0) or 0)
            stocks.append({
                "ticker": ticker,
                "shares": shares,
                "cost": round(cost_jpy / fx, 1),
                "price": round(price_jpy / fx, 1),
                "pnl": round(pnl_jpy / fx, 0),
                "account": sub_key.upper().replace("_", " "),
            })
    return stocks


def _normalize_ticker(ticker: str) -> str:
    """ticker を比較可能な正規形に揃える。
    1489 / 1489.T / 1489.t を全て "1489.T" に統一。
    """
    if not ticker:
        return ""
    t = str(ticker).strip().upper()
    if t.isdigit():
        return f"{t}.T"
    return t


def _filter_new_picks_excluding_held(new_picks: list, held_tickers: set) -> list:
    """new_picks から既保有 ticker を除外する。
    1489 vs 1489.T のような表記揺れに対応するため正規化して比較。
    """
    held_norm = {_normalize_ticker(t) for t in held_tickers}
    return [p for p in new_picks if _normalize_ticker(p.get("ticker", "")) not in held_norm]


def build_action_cards(record: dict, stocks: list[dict]) -> str:
    """最新のtrack recordからアクションカードHTMLを生成する。"""
    # actions / reasons / ai_allocation のキーを全て正規化
    actions = {_normalize_ticker(k): v for k, v in record.get("actions", {}).items()}
    reasons = {_normalize_ticker(k): v for k, v in record.get("action_reasons", {}).items()}
    ai_alloc = {_normalize_ticker(k): v for k, v in record.get("ai_allocation", {}).items()}

    # stocks lookup (キーは正規化形 "1489.T" に統一)
    stock_map: dict[str, dict] = {}
    for s in stocks:
        t = _normalize_ticker(s["ticker"])
        if t not in stock_map:
            stock_map[t] = {"shares": 0, "cost": 0, "price": s["price"], "pnl": 0}
        stock_map[t]["shares"] += s["shares"]
        stock_map[t]["cost"] = round(
            (stock_map[t]["cost"] * (stock_map[t]["shares"] - s["shares"]) +
             s["cost"] * s["shares"]) / max(stock_map[t]["shares"], 1), 1
        ) if stock_map[t]["shares"] > 0 else s["cost"]
        stock_map[t]["pnl"] += s["pnl"]

    def card(ticker: str, action: str) -> str:
        """個別カードHTML — トレーディングアプリ風表示。"""
        css_class = action.lower()
        badge_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(action, "⚪")
        sm = stock_map.get(_normalize_ticker(ticker), {})
        pnl = sm.get("pnl", 0)
        cost = sm.get("cost", 0)
        price = sm.get("price", 0)
        shares = sm.get("shares", 0)
        # P&L %
        pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
        is_profit = pnl >= 0
        pnl_class = "pnl-positive" if is_profit else "pnl-negative"
        arrow = "▲" if is_profit else "▼"
        pnl_sign = "+" if is_profit else ""
        pnl_str = f"{pnl_sign}${pnl:,.0f}"
        pnl_pct_str = f"{pnl_sign}{pnl_pct:.2f}%"
        alloc_pct = ai_alloc.get(ticker, 0) * 100
        reason_text = html_mod.escape(reasons.get(ticker, ""))

        holdings_html = ""
        if sm and shares > 0:
            holdings_html = f'''<div class="holdings-pro {pnl_class}">
              <div class="holdings-row">
                <span class="hold-label">保有</span>
                <span class="hold-value">{shares}株</span>
              </div>
              <div class="holdings-row">
                <span class="hold-label">取得 → 現在</span>
                <span class="hold-value">${cost:,.1f} → ${price:,.1f}</span>
              </div>
              <div class="holdings-pnl">
                <span class="pnl-arrow">{arrow}</span>
                <span class="pnl-amount">{pnl_str}</span>
                <span class="pnl-pct">({pnl_pct_str})</span>
              </div>
            </div>'''

        return f'''<div class="action-card {css_class} fade-in">
          <div class="card-header">
            <span class="ticker">{ticker}</span>
            <span class="action-badge {css_class}">{badge_emoji} {action}</span>
          </div>
          {holdings_html}
          <div class="reason">{reason_text}</div>
          <div class="alloc-bar"><div class="alloc-bar-fill" style="width: {min(alloc_pct * 6.67, 100):.0f}%;"></div></div>
          <div class="alloc-label">推奨配分 <strong>{alloc_pct:.1f}%</strong></div>
        </div>'''

    # actions のキーも正規化 (1489 → 1489.T)
    groups = {"BUY": [], "SELL": [], "HOLD": []}
    for t, a in actions.items():
        groups.setdefault(a, []).append(_normalize_ticker(t))

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

    # New picks (既保有を除外)
    new_picks = _filter_new_picks_excluding_held(
        record.get("new_picks", []),
        set(stock_map.keys())
    )
    if new_picks:
        html += f'<div class="section"><div class="section-title"><span class="icon">🆕</span> 新規推奨銘柄（{len(new_picks)}銘柄・既保有除外済）</div><div class="picks-grid">'
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


# ==============================================================================
# 用語集 — 専門用語は初出時に解説する
# ==============================================================================

GLOSSARY: dict[str, str] = {
    "VIX": "VIX (恐怖指数) = S&P500オプション市場が織り込む向こう30日のボラティリティ。20以下で平穏、25超で警戒、30超で危機局面",
    "ROE": "ROE (株主資本利益率) = 純利益 ÷ 株主資本。15%以上で優良企業、20%超でバフェットが好む経済的堀の証拠",
    "PER": "PER (株価収益率) = 株価 ÷ 1株あたり利益。成長株は高くなる傾向。15倍は割安・25倍は普通・40倍超は割高の目安",
    "FCF": "FCF (フリーキャッシュフロー) = 営業活動で稼いだ現金から設備投資を引いた残り。「企業が自由に使えるお金」",
    "FCF Yield": "FCF Yield = FCF ÷ 時価総額。5%以上で割安、債券利回りより高ければ魅力的",
    "DCF": "DCF (割引キャッシュフロー) = 将来のFCFを現在価値に割り引いて企業の本来価値を算出する手法。バフェットの「内在価値」評価軸",
    "Margin of Safety": "安全域 (Margin of Safety) = 内在価値と市場価格の差。「30%安く買えるなら買う」というバフェット哲学の核",
    "KL Divergence": "KLダイバージェンス = 2つの確率分布の「ずれ」を測る統計量。マーケットが平常時から逸脱している度合いを表現",
    "Sharpe": "シャープレシオ (Sharpe) = (リターン − 無リスク金利) ÷ ボラティリティ。1.0以上で優秀、2.0超で卓越",
    "Drawdown": "ドローダウン (DD) = 直近高値から最安値までの下落率。-20%超なら警戒、-30%超は重大な弱気局面",
    "Kelly基準": "Kelly基準 = 期待リターンとボラティリティから「最適な賭け金」を導く数学公式。投資額の上限を決める指針",
    "Kalman Filter": "カルマンフィルタ = ノイズ除去アルゴリズム。価格の真のトレンドを統計的に抽出",
    "Hurst指数": "Hurst指数 = 価格時系列の「持続性」を測る指標。0.55超でトレンド継続、0.45未満で平均回帰傾向",
    "Cross-Sectional": "Cross-Sectional Momentum = 同時点で複数銘柄を比較し、相対的に強い銘柄を買う戦略",
    "Mean Reversion": "Mean Reversion (平均回帰) = 「価格は長期平均に戻る」という統計的傾向",
    "Earnings Yield": "Earnings Yield = 1 ÷ PER。「株式の利回り換算」。国債利回りと比較してリスクプレミアムを評価",
    "Insider": "インサイダー取引 = 経営陣による自社株売買。買いが増えると将来の業績に自信のサイン",
    "Put/Call ratio": "Put/Call比率 = プットオプション ÷ コールオプション。1.0超で投資家が下落ヘッジに偏っている逆張りシグナル",
    "Walk-Forward": "Walk-Forward検証 = 過去の予測を実取引と仮定して累積リターンを検証する方法",
    "Black-Litterman": "ブラック・リッターマン (Black-Litterman) = 市場均衡 + 投資家の見解を統合する最適配分モデル",
}


def _explain(term: str, short: bool = False) -> str:
    """用語に解説リンク (tooltip) を埋め込む。"""
    expl = GLOSSARY.get(term, "")
    if not expl:
        return html_mod.escape(term)
    if short:
        return f'<span class="glossary-term" title="{html_mod.escape(expl)}">{html_mod.escape(term)}</span>'
    return (f'<span class="glossary-term" title="{html_mod.escape(expl)}">{html_mod.escape(term)}'
            f'<sup class="glossary-hint">?</sup></span>')


def _glossary_section_html(terms: list[str]) -> str:
    """記事の末尾に用語集を出力。"""
    items = []
    for t in terms:
        if t in GLOSSARY:
            items.append((t, GLOSSARY[t]))
    if not items:
        return ""
    html = '<details class="db-glossary"><summary class="db-glossary-toggle">📚 この記事で使った用語の解説 (クリックで展開)</summary>'
    html += '<dl class="db-glossary-list">'
    for term, expl in items:
        html += f'<dt>{html_mod.escape(term)}</dt><dd>{html_mod.escape(expl)}</dd>'
    html += '</dl></details>'
    return html


def build_daily_brief_html(results: dict, track: dict, holdings: dict) -> str:
    """🗞️ 今日の分析レポート — AI による記事形式の市場ブリーフ.

    Daily Brief を生成。`claude_analyst` / `gemini_analyst` で得られた
    最新の reasoning + action_reasons + risk_scenarios を物語として
    構造化する。
    """
    records = track.get("records", [])
    latest = records[-1] if records else {}
    today = datetime.now().strftime("%Y-%m-%d (%a)")

    regime = results.get("regime", "unknown")
    kl_val = results.get("kl_value", 0)
    regime_jp = {
        "low_vol":    "🟢 低ボラティリティ（安定上昇局面）",
        "transition": "🟡 移行期（ボラティリティ上昇中）",
        "crisis":     "🔴 危機（高ボラティリティ・急落局面）",
    }.get(regime, "⚪ 不明")

    confidence = latest.get("confidence", 0)
    ai_reasoning = latest.get("ai_reasoning") or latest.get("reasoning", "")
    actions = latest.get("actions", {})
    action_reasons = latest.get("action_reasons", {})
    risk_scenarios = latest.get("risk_scenarios", {})
    held_tickers_set = {s["ticker"] for s in build_holdings_data(holdings)}
    new_picks = _filter_new_picks_excluding_held(latest.get("new_picks", []), held_tickers_set)
    tsumitate = latest.get("tsumitate_advice", {})
    rebalance_opinion = latest.get("rebalance_opinion", {})

    # マスター Wisdom + Strategy Lab
    master_signals = results.get("master_signals", []) or []
    enh = results.get("enhancements", {}) or {}
    wf = enh.get("walk_forward", {}) or {}
    anomaly = enh.get("anomaly", {}) or {}
    sector_rot = enh.get("sector_rotation", {}) or {}

    # 保有サマリ
    summary = holdings.get("us_stocks", {}).get("summary", {})
    total_pnl = summary.get("total_unrealized_pnl", 0)
    total_value = summary.get("total_market_value", 0)
    total_pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if total_value > total_pnl else 0

    # アクション分類
    buys = sorted([(t, action_reasons.get(t, "")) for t, a in actions.items() if a == "BUY"],
                  key=lambda x: -len(x[1]))
    sells = [(t, action_reasons.get(t, "")) for t, a in actions.items() if a == "SELL"]
    holds_count = sum(1 for a in actions.values() if a == "HOLD")

    html = '<article class="daily-brief"><header class="db-header">'
    html += f'<div class="db-eyebrow">DAILY MARKET BRIEF — Powered by Master Wisdom AI</div>'
    html += f'<h1 class="db-title">{today} のポートフォリオ分析</h1>'
    html += f'<div class="db-meta">'
    html += f'<span class="db-meta-item">📊 レジーム: <strong>{regime_jp}</strong></span>'
    html += f'<span class="db-meta-item">🎯 AI 確信度: <strong>{confidence:.0%}</strong></span>'
    html += f'<span class="db-meta-item">📈 含み損益: <strong style="color:{"var(--accent-green)" if total_pnl >= 0 else "var(--accent-red)"}">{"+" if total_pnl >= 0 else ""}${total_pnl:,.0f} ({total_pnl_pct:+.1f}%)</strong></span>'
    html += '</div></header>'

    # === 長期保有方針バナー (常時表示) ===
    html += '<div class="db-longterm-banner">'
    html += '<span class="icon">🏛️</span>'
    html += '<div><strong>本ポートフォリオの基本方針: 長期保有 (Buy &amp; Hold)</strong>'
    html += '<p>短期の値動きでトレードはしません。<strong>「優れた企業を妥当な価格で買い、長く持つ」</strong>'
    html += 'というバフェット流の哲学に従います。日々のシグナルは「方針の微修正」であり、'
    html += '安易な売却 (BUY → SELL の往復) は税負担と機会損失を生むため避けます。</p></div></div>'

    # === Lead paragraph (なぜそう判断したか + 出典) ===
    if ai_reasoning:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">📰 リード — 本日の市場概況</h2>'
        html += f'<p class="db-lead">{html_mod.escape(ai_reasoning)}</p>'
        html += '<aside class="db-source">'
        html += '<strong>📌 なぜそう判断したか</strong>'
        kl_judgment = (
            "ほぼ平常時の分布に近く、予測の信頼度を高めに保てる" if kl_val < 0.05
            else "通常から少しずれており、予測には慎重さが必要" if kl_val < 0.15
            else "大きく逸脱しており、過去パターンが効きにくい異常局面"
        )
        html += (f'<p>マーケット状態は ' + _explain("KL Divergence") +
                f' という統計指標で評価しています。今日の値は {kl_val:.4f} で、{kl_judgment}と判定しました。'
                f'この指標は「現在の値動きの分布が、長期平均からどれだけ離れているか」を測ります。'
                '0に近いほど通常時、大きいほど異常時です。</p>')
        html += '<p>出典: マーケット価格データ (yfinance) ・ KL Divergence は内製エンジン daily_evolution.py で算出</p>'
        html += '</aside></section>'

    # === マクロ・異常検知 ===
    if anomaly or sector_rot:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">🌐 マクロ環境</h2>'
        if anomaly.get("is_anomaly"):
            sev = anomaly.get("severity", 0) * 100
            triggers = "、".join(anomaly.get("triggers", []))
            html += f'<div class="db-callout db-callout-warn">'
            html += f'<strong>⚠️ 異常検知 (深刻度 {sev:.0f}%)</strong>: {html_mod.escape(triggers)}。'
            html += f'防御的ポジショニング推奨。</div>'
        else:
            html += '<p class="db-body">VIX、30日リターン、直近5日トレンドを総合的に評価し、'
            html += '<strong>マーケットは正常範囲内</strong>と判断しています。'
            html += '黒い白鳥イベントの兆候は検出されていません。</p>'

        if sector_rot.get("hot_sectors"):
            html += '<p class="db-body">'
            html += f'🔥 直近30日で<strong>勢いのあるセクター</strong>: {", ".join(sector_rot["hot_sectors"])}。'
            if sector_rot.get("cold_sectors"):
                html += f' 一方<strong>勢いを失っているセクター</strong>: {", ".join(sector_rot["cold_sectors"])}。'
            html += '</p>'
        html += '</section>'

    # === キーアクション (BUY) — 各銘柄の「なぜ」を厚めに解説 ===
    if buys:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🟢 本日のBUY推奨 ({len(buys)}銘柄)</h2>'
        html += '<p class="db-meta-line">各銘柄について、AIが買いと判断した理由・根拠データ・参考にした指標を解説します。</p>'
        for ticker, reason in buys[:6]:
            url = yahoo_chart_url(ticker)
            ms_match = next((s for s in master_signals if s.get("ticker") == ticker), None)
            master_note = ""
            factor_table = ""
            if ms_match:
                comp = ms_match.get("composite_score", 0)
                conf = ms_match.get("confidence", 0)
                master_note = f' <span class="db-master-tag">9ファクター合成 {comp:+.2f} / 確信度 {conf:.0%}</span>'
                fs = ms_match.get("factor_scores", {})
                if fs:
                    factor_table = '<details class="db-factor-detail"><summary>9ファクターの内訳を見る</summary><table class="db-factor-table"><tr><th>ファクター</th><th>スコア</th><th>判定</th></tr>'
                    factor_jp = {
                        "quality_roe": "経営効率 (ROE)",
                        "quality_margin": "利益率の質",
                        "value_earnings_yield": "Earnings Yield",
                        "value_fcf_yield": "FCF Yield",
                        "value_margin_of_safety": "安全域 (DCF)",
                        "momentum_composite": "テクニカル統合",
                        "contrarian_fear_greed": "Fear-Greed逆張り",
                        "contrarian_insider_pulse": "インサイダー脈動",
                        "risk_kelly": "Kelly基準サイズ",
                    }
                    for fname, fdata in fs.items():
                        s = fdata.get("score", 0) if isinstance(fdata, dict) else fdata
                        v = fdata.get("verdict", "") if isinstance(fdata, dict) else ""
                        score_color = "var(--accent-green)" if s > 0 else "var(--accent-red)" if s < 0 else "var(--text-muted)"
                        factor_table += f'<tr><td>{factor_jp.get(fname, fname)}</td><td style="color:{score_color}">{s:+.2f}</td><td>{html_mod.escape(v)}</td></tr>'
                    factor_table += '</table><p class="db-meta-line">9つのファクター(指標)それぞれが [-1,+1] の範囲でスコアを出し、加重平均で合成スコアを算出しています。</p></details>'
            html += f'<article class="db-action-block db-buy">'
            html += f'<h3 class="db-action-title"><a href="{url}" target="_blank" class="ticker-link">{ticker} 📈</a>{master_note}</h3>'
            html += f'<p class="db-body">{html_mod.escape(reason)}</p>'
            html += factor_table
            html += '</article>'
        html += '<aside class="db-source">'
        html += '<strong>📚 BUYの判断ロジック</strong>'
        html += ('<p>Master Wisdom は9つの観点で銘柄を評価しています:</p>'
                 '<ol><li><strong>経営効率</strong>: ' + _explain("ROE") + ' — 「同じ資本でどれだけ稼げるか」</li>'
                 '<li><strong>利益率</strong>: 営業利益率の水準と安定性</li>'
                 '<li><strong>株価妥当性</strong>: ' + _explain("PER") + 'を国債利回りと比較</li>'
                 '<li><strong>キャッシュ生成力</strong>: ' + _explain("FCF Yield") + '</li>'
                 '<li><strong>安全域</strong>: ' + _explain("DCF") + 'で算出した本来価値と現在価格の差 (' + _explain("Margin of Safety") + ')</li>'
                 '<li><strong>テクニカル</strong>: ' + _explain("Kalman Filter") + 'などで価格トレンドを検出</li>'
                 '<li><strong>逆張り</strong>: ' + _explain("VIX") + 'が高い時に「皆が恐れている時こそ買う」というバフェット哲学</li>'
                 '<li><strong>スマートマネー</strong>: ' + _explain("Insider") + 'と' + _explain("Put/Call ratio") + '</li>'
                 '<li><strong>リスク量</strong>: ' + _explain("Kelly基準") + 'での最適投資額</li></ol>')
        html += '<p>これらを<strong>市場局面ごとに重み付け</strong>します。安定相場では成長性、危機局面では安全域を重視するように動的に調整しています。</p>'
        html += ('<p><strong>BUY の意味 (長期保有の文脈で):</strong> '
                 '本ポートフォリオでは「BUY = 翌日値上がりを狙う短期トレード」ではなく、'
                 '<strong>「3〜10年保有する価値がある銘柄を買い増す」</strong>を意味します。'
                 'シグナル変動による頻繁な売買は行いません。</p>')
        html += '<p>出典: 9ファクター = master_predictor.py / 学習ジャーナル = data/master_learning_journal.json</p>'
        html += '</aside></section>'

    # === キーアクション (SELL) ===
    if sells:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🔴 本日のSELL推奨 ({len(sells)}銘柄)</h2>'
        for ticker, reason in sells:
            url = yahoo_chart_url(ticker)
            ms_match = next((s for s in master_signals if s.get("ticker") == ticker), None)
            master_note = ""
            if ms_match:
                comp = ms_match.get("composite_score", 0)
                master_note = f' <span class="db-master-tag">Master Wisdom: 合成{comp:+.2f}</span>'
            html += f'<article class="db-action-block db-sell">'
            html += f'<h3 class="db-action-title"><a href="{url}" target="_blank" class="ticker-link">{ticker} 📈</a>{master_note}</h3>'
            html += f'<p class="db-body">{html_mod.escape(reason)}</p>'
            html += '</article>'
        html += '</section>'

    # === HOLD要約 ===
    if holds_count:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🟡 HOLD ({holds_count}銘柄) — 長期保有を継続</h2>'
        html += f'<p class="db-body">残り <strong>{holds_count}銘柄</strong> は<strong>長期保有を継続</strong>します。'
        html += '本ポートフォリオの基本方針は Buy &amp; Hold (バフェット流) であり、'
        html += '日々のシグナルゆらぎでの売買は<strong>原則行いません</strong>。</p>'
        html += '<p class="db-body">「素晴らしい企業を妥当な価格で買い、長く保有する」ことで、'
        html += '<strong>複利成長</strong>と<strong>税繰延</strong>の恩恵を最大化します。'
        html += '頻繁な売買は手数料・税金・タイミングミスのトリプルパンチで長期リターンを毀損します。</p>'
        html += '</section>'

    # === Walk-Forward 検証 ===
    if wf.get("n_trades", 0) > 0:
        sharpe = wf.get("sharpe", 0)
        ann_ret = wf.get("annual_return", 0) * 100
        win = wf.get("win_rate", 0) * 100
        rating = "卓越" if sharpe >= 1.5 else "良好" if sharpe >= 1.0 else "普通" if sharpe >= 0.5 else "要改善"
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">📊 戦略実証 (Walk-Forward)</h2>'
        html += f'<p class="db-body">'
        html += f'過去の予測を仮想実戦で検証した結果、<strong>シャープレシオ {sharpe:+.2f}</strong> ({rating})、'
        html += f'年率リターン <strong>{ann_ret:+.1f}%</strong>、勝率 <strong>{win:.0f}%</strong> '
        html += f'({wf.get("n_trades", 0)} 取引で算出)。'
        html += 'これは「本当に儲かる戦略か」を統計的に裏付ける数値です。</p>'
        html += '</section>'

    # === リスクシナリオ ===
    if risk_scenarios:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">⚖️ リスクシナリオ — 今後の3つの可能性</h2>'
        for key, emoji, jp in [("bull", "🐂", "Bull (強気)"), ("base", "📊", "Base (基本)"), ("bear", "🐻", "Bear (弱気)")]:
            s = risk_scenarios.get(key, {})
            prob = s.get("probability", 0) * 100
            desc = s.get("description", "")
            if not desc:
                continue
            html += f'<article class="db-scenario db-{key}">'
            html += f'<div class="db-scenario-header"><span class="db-scenario-emoji">{emoji}</span>'
            html += f'<span class="db-scenario-name">{jp}</span>'
            html += f'<span class="db-scenario-prob">{prob:.0f}%</span></div>'
            html += f'<p class="db-body">{html_mod.escape(desc)}</p>'
            html += '</article>'
        html += '</section>'

    # === 新規推奨銘柄 (非保有・長期候補) ===
    if new_picks:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🆕 新規購入候補 — 非保有の長期投資先 ({len(new_picks)}銘柄)</h2>'
        html += '<p class="db-meta-line">現在保有していないが、Master Wisdom が「10年以上保有する価値あり」と判定した銘柄です。'
        html += '短期の値上がりではなく、<strong>長期 (3〜10年) の複利成長</strong>を期待する候補として提示しています。</p>'
        for p in new_picks[:5]:
            ticker = p.get("ticker", "")
            url = yahoo_chart_url(ticker)
            horizon = p.get("horizon", "長期 (3〜10年)")
            quality_note = p.get("quality_note", "")
            html += f'<article class="db-pick-block">'
            html += f'<h3 class="db-action-title"><a href="{url}" target="_blank" class="ticker-link">{ticker} 📈</a>'
            html += f' <span class="db-sector-tag">{html_mod.escape(p.get("sector", ""))}</span>'
            html += f' <span class="db-master-tag">想定保有期間: {html_mod.escape(horizon)}</span></h3>'
            html += f'<p class="db-body">{html_mod.escape(p.get("reason", ""))}</p>'
            if quality_note:
                html += f'<p class="db-meta-line">📌 評価ポイント: {html_mod.escape(quality_note)}</p>'
            html += '</article>'
        html += '<aside class="db-source">'
        html += '<strong>📚 新規候補のスクリーニング基準 (長期保有前提)</strong>'
        html += ('<ol><li><strong>'+ _explain("ROE") +' ≥ 15%</strong> が直近5年安定 — 経済的堀の証拠</li>'
                 '<li><strong>'+ _explain("FCF Yield") +' ≥ 4%</strong> — 自由なキャッシュ生成力</li>'
                 '<li><strong>'+ _explain("Margin of Safety") +' ≥ 20%</strong> — 内在価値より20%以上安く買える</li>'
                 '<li><strong>負債比率が業界平均以下</strong> — 不況耐性</li>'
                 '<li><strong>10年後も需要が見える事業モデル</strong> — 長期保有の前提</li></ol>')
        html += '<p>これらは「3〜10年の保有を前提にした候補」であり、'
        html += '短期トレードではありません。一度買ったら基本は売らず、複利でポートフォリオを成長させます。</p>'
        html += '<p>出典: master_predictor.py の screening_universe (S&amp;P500 + 日本主要株) / 評価データ = data/master_screening_results.json</p>'
        html += '</aside></section>'

    # === 積立アドバイス ===
    if tsumitate and (tsumitate.get("changes") or tsumitate.get("reasoning")):
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">💰 積立設定アドバイス</h2>'
        if tsumitate.get("reasoning"):
            html += f'<p class="db-body">{html_mod.escape(tsumitate["reasoning"])}</p>'
        if tsumitate.get("changes"):
            html += '<ul class="db-list">'
            for c in tsumitate["changes"][:5]:
                html += f'<li>{html_mod.escape(c)}</li>'
            html += '</ul>'
        html += '</section>'

    # === AI 自身の意見 (リバランスオピニオン) ===
    if rebalance_opinion and rebalance_opinion.get("override_reason"):
        agree = rebalance_opinion.get("agree_with_proposals", True)
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🧠 AI のリバランス見解</h2>'
        html += '<p class="db-body">'
        html += f'<strong>{"アルゴリズム提案に賛同" if agree else "独自見解を提示"}</strong>: '
        html += html_mod.escape(rebalance_opinion["override_reason"])
        html += '</p>'
        swaps = rebalance_opinion.get("additional_swaps", [])
        if swaps:
            html += '<ul class="db-list">'
            for s in swaps[:3]:
                html += f'<li>{html_mod.escape(s.get("sell", "?"))} → {html_mod.escape(s.get("buy", "?"))}: {html_mod.escape(s.get("reason", ""))}</li>'
            html += '</ul>'
        html += '</section>'

    # === 用語集 (記事末尾) ===
    used_terms = ["KL Divergence", "ROE", "PER", "FCF Yield", "DCF", "Margin of Safety",
                  "VIX", "Sharpe", "Drawdown", "Kelly基準", "Kalman Filter",
                  "Insider", "Put/Call ratio"]
    html += _glossary_section_html(used_terms)

    html += '<footer class="db-footer">'
    html += '<p class="db-meta-line"><strong>📖 この記事の作り方</strong></p>'
    html += '<ul class="db-source-list">'
    html += '<li>マーケット価格データ: <em>yfinance</em> (Yahoo Finance APIラッパー)</li>'
    html += '<li>9ファクター予測: <em>master_predictor.py</em> (バフェット哲学+現代統計学)</li>'
    html += '<li>30年歴史パターン: <em>historical_pattern_extractor.py</em> (1987 Black Monday / 2000 Dot-com / 2008 GFC / 2020 COVID 等を含む)</li>'
    html += '<li>AI 日本語コメント: <em>Claude Sonnet</em> (Anthropic) 経由で <em>auto_prompt_cycle.py</em> が分析</li>'
    html += '<li>過去予測の答え合わせ: <em>data/master_prediction_history.json</em> に蓄積</li>'
    html += '<li>自己学習: 7日後・30日後の実リターンと予測の方向一致率からウェイトを Bayesian 更新</li>'
    html += '</ul>'
    html += '<p class="db-meta-line">毎営業日 7時に launchd が自動更新。本記事は AI 生成のため、投資判断は最終的にご自身でお願いします。'
    html += 'これは投資助言ではなく、自動化された分析レポートです。</p>'
    html += '</footer></article>'
    return html


def build_weekly_brief_html(results: dict, track: dict, holdings: dict) -> str:
    """📅 週次ブリーフ — 過去5営業日の振り返り + 来週展望.

    AIが何を予測し、それが当たったか/外れたか、何を学んだかを記事化する。
    """
    today = datetime.now().strftime("%Y-%m-%d (%a)")
    records = track.get("records", [])
    last_7 = records[-7:] if len(records) > 0 else []

    # 直近の評価
    evaluations = track.get("evaluations", [])
    recent_evals = evaluations[-5:] if evaluations else []

    # ファクター精度 (master_learning から)
    master_learning = results.get("master_learning", {}) or {}
    factor_accs = master_learning.get("factor_accuracies", {}) or {}

    enh = results.get("enhancements", {}) or {}
    wf = enh.get("walk_forward", {}) or {}

    html = '<article class="daily-brief"><header class="db-header">'
    html += '<div class="db-eyebrow">WEEKLY REPORT — 直近1週間の振り返り</div>'
    html += f'<h1 class="db-title">{today} 週次ポートフォリオレポート</h1>'
    html += '<div class="db-meta">'
    html += f'<span class="db-meta-item">📊 予測記録 <strong>{len(records)}件</strong></span>'
    html += f'<span class="db-meta-item">✅ 採点済み <strong>{len(evaluations)}件</strong></span>'
    html += '</div></header>'

    # === 長期保有方針バナー ===
    html += '<div class="db-longterm-banner">'
    html += '<span class="icon">🏛️</span>'
    html += '<div><strong>週次レビューの位置づけ</strong>'
    html += '<p>このレポートは「売買の指示書」ではなく、<strong>長期保有戦略の進捗確認</strong>です。'
    html += '1週間という単位は短期 (ノイズが多い) ですが、AI の学習が正しい方向に進んでいるか、'
    html += '保有銘柄の長期ストーリーが崩れていないかをチェックする目的で作成しています。</p></div></div>'

    # === 今週のハイライト ===
    html += '<section class="db-section">'
    html += '<h2 class="db-section-title">📌 今週のハイライト</h2>'
    if recent_evals:
        avg_dir = sum(e.get("predicted_vs_actual", {}).get("direction_accuracy", 0)
                      for e in recent_evals) / len(recent_evals)
        rating = "卓越" if avg_dir >= 0.7 else "良好" if avg_dir >= 0.6 else "普通" if avg_dir >= 0.5 else "要改善"
        html += f'<p class="db-lead">直近{len(recent_evals)}回の予測の方向一致率は <strong>{avg_dir*100:.0f}%</strong> ({rating})。'
        html += 'これは「BUY と判断した銘柄が実際に上昇したか / SELL と判断した銘柄が実際に下落したか」の的中率です。'
        html += '50%なら偶然の的中率なので、それを超える数字は AI が市場の方向性を捉えていることを示します。</p>'
    else:
        html += '<p class="db-lead">まだ採点済みの予測がありません。予測から7営業日経過すると自動採点が始まります。</p>'

    if wf.get("n_trades", 0) > 0:
        sharpe = wf.get("sharpe", 0)
        ann = wf.get("annual_return", 0) * 100
        win = wf.get("win_rate", 0) * 100
        html += '<aside class="db-source">'
        html += '<strong>📊 Walk-Forward 検証 (仮想実戦)</strong>'
        html += (f'<p>過去の確信度 ≥ 0.6 の予測を「実取引」と仮定して累積リターンを計算した結果、'
                 f'シャープレシオ ' + _explain("Sharpe") +
                 f' は <strong>{sharpe:+.2f}</strong>、年率リターン <strong>{ann:+.1f}%</strong>、'
                 f'勝率 <strong>{win:.0f}%</strong> でした ({wf.get("n_trades", 0)}取引)。</p>')
        html += '<p>シャープレシオは「リスク1単位あたりのリターン」で、値が高いほど効率的に運用できていることを示します。'
        html += 'プロのファンドマネージャーで 1.0、卓越したヘッジファンドで 2.0+ が目安です。</p>'
        html += '<p>出典: prediction_enhancements.walk_forward_sharpe / data/walk_forward_results.json</p>'
        html += '</aside>'
    html += '</section>'

    # === 今週の予測精度トレンド ===
    if factor_accs:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">⚙️ 9ファクターの精度ランキング</h2>'
        html += '<p class="db-body">Master Wisdom が使う9つの予測ファクターを精度順に並べました。'
        html += '精度が高いほど次回の予測でウェイトが上がり、低いものは縮小される自己進化の仕組みです。</p>'
        sorted_factors = sorted(factor_accs.items(), key=lambda x: -x[1])
        factor_jp = {
            "quality_roe": "経営効率 (ROE)",
            "quality_margin": "利益率の質",
            "value_earnings_yield": "Earnings Yield",
            "value_fcf_yield": "FCF Yield",
            "value_margin_of_safety": "安全域 (DCF)",
            "momentum_composite": "テクニカル統合",
            "contrarian_fear_greed": "Fear-Greed逆張り",
            "contrarian_insider_pulse": "インサイダー脈動",
            "risk_kelly": "Kelly基準",
        }
        html += '<table class="db-factor-table"><tr><th>順位</th><th>ファクター</th><th>精度</th><th>評価</th></tr>'
        for i, (fname, acc) in enumerate(sorted_factors, 1):
            rating = "🌟 卓越" if acc >= 0.7 else "✅ 良好" if acc >= 0.6 else "⚠️ 普通" if acc >= 0.5 else "🔴 改善要"
            html += f'<tr><td>{i}</td><td>{factor_jp.get(fname, fname)}</td><td>{acc*100:.1f}%</td><td>{rating}</td></tr>'
        html += '</table>'
        html += '<aside class="db-source">'
        html += '<strong>📌 なぜ精度が変動するのか</strong>'
        html += ('<p>市場の局面により有効なファクターは変わります。例えば成長相場ではテクニカル、危機局面では安全域 ('
                 + _explain("DCF") + ') が効きます。'
                 'システムは7日後/30日後の答え合わせを通じて、各ファクターの「真の予測力」を継続的に学習しています。</p>')
        html += '<p>出典: data/master_factor_weights.json (Bayesian更新後の重み)</p>'
        html += '</aside></section>'

    # === 今週の「学び」 ===
    findings = master_learning.get("notable_findings", []) or []
    if findings:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">💡 今週 AI が学んだこと</h2>'
        html += '<p class="db-body">過去予測との突き合わせから、次の発見がありました:</p>'
        html += '<ul class="db-list">'
        for f in findings[:10]:
            html += f'<li>{html_mod.escape(f)}</li>'
        html += '</ul>'
        html += '<aside class="db-source">'
        html += '<strong>🧠 自己進化のメカニズム</strong>'
        html += '<p>毎日、過去の予測の方向と実際の値動きを比較します。当たっていればそのファクターの重みを上げ、外れていれば下げる。'
        html += 'これを繰り返すことで、AI は「今のマーケットで何が効くか」を自分で学んでいきます。'
        html += '人間が手動でパラメータを調整する必要はありません。</p>'
        html += '<p>出典: data/master_learning_journal.json (日次の学習エントリ)</p>'
        html += '</aside></section>'

    # === 来週の展望 ===
    latest = records[-1] if records else {}
    risk_scenarios = latest.get("risk_scenarios", {})
    if risk_scenarios:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">🔮 来週の展望 — 3シナリオ</h2>'
        html += '<p class="db-body">今後1〜4週の市場展開について、Master Wisdom が3つの可能性を確率付きで予測しています。'
        html += 'これは確定的な予言ではなく、「どれくらい起こりやすいか」の主観的確率です。</p>'
        for key, emoji, jp in [("bull", "🐂", "Bull (強気)"), ("base", "📊", "Base (基本)"), ("bear", "🐻", "Bear (弱気)")]:
            s = risk_scenarios.get(key, {})
            prob = s.get("probability", 0) * 100
            desc = s.get("description", "")
            if not desc:
                continue
            html += f'<article class="db-scenario db-{key}">'
            html += f'<div class="db-scenario-header"><span class="db-scenario-emoji">{emoji}</span>'
            html += f'<span class="db-scenario-name">{jp}</span>'
            html += f'<span class="db-scenario-prob">{prob:.0f}%</span></div>'
            html += f'<p class="db-body">{html_mod.escape(desc)}</p></article>'
        html += '</section>'

    # === 非保有の長期候補ウォッチリスト ===
    latest = records[-1] if records else {}
    held_w = {s["ticker"] for s in build_holdings_data(holdings)}
    new_picks_w = _filter_new_picks_excluding_held(latest.get("new_picks", []), held_w)
    if new_picks_w:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">👀 今週の長期候補ウォッチ ({len(new_picks_w)}銘柄)</h2>'
        html += '<p class="db-body">非保有のうち、Master Wisdom が今週「長期 (3〜10年) で監視したい」と判定した銘柄です。'
        html += '今週すぐに買う必要はありません。<strong>「価格が割安まで下がるのを待つ」</strong>のがバフェット流の規律です。</p>'
        html += '<ul class="db-list">'
        for p in new_picks_w[:6]:
            t = p.get("ticker", "")
            sector = p.get("sector", "")
            reason = p.get("reason", "")
            html += f'<li><strong>{html_mod.escape(t)}</strong> <span class="db-sector-tag">{html_mod.escape(sector)}</span> — {html_mod.escape(reason)}</li>'
        html += '</ul>'
        html += '<aside class="db-source"><strong>📌 ウォッチリストの使い方</strong>'
        html += '<p>気になる銘柄を「すぐ買う」のではなく「下がったら買う」候補として記録しておきます。'
        html += '市場全体が ' + _explain("VIX") + ' > 25 になったとき、または個別株が内在価値の30%以下に下落したとき、'
        html += '実際の買い (BUY タブに昇格) を検討します。</p></aside></section>'

    # === 売却を急がないことの重要性 ===
    sell_count_recent = sum(1 for r in last_7 for a in r.get("actions", {}).values() if a == "SELL")
    if sell_count_recent > 0:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">🤔 売却シグナルとどう付き合うか</h2>'
        html += f'<p class="db-body">直近1週間で SELL シグナルは <strong>{sell_count_recent}件</strong> 出ました。'
        html += 'ただし長期保有方針では、<strong>シグナルが即「売り」を意味しない</strong>点に注意します。'
        html += '本当に売却すべきは以下のケースに絞っています:</p>'
        html += '<ol class="db-list">'
        html += '<li>企業の <strong>長期ストーリーが崩壊</strong> (経済的堀の喪失、不正会計、業界構造の終焉)</li>'
        html += '<li>株価が <strong>内在価値の2倍以上に高騰</strong> (極端な過大評価)</li>'
        html += '<li>ポートフォリオの <strong>1銘柄集中度が30%超</strong> (リバランス目的)</li>'
        html += '</ol>'
        html += '<p class="db-body">単なる短期下落・ボラティリティ・テクニカル悪化は売却理由になりません。'
        html += '「売らない」ことで税繰延と複利効果を最大化します。</p>'
        html += '</section>'

    # 用語集
    used_terms = ["Sharpe", "DCF", "ROE", "FCF Yield", "Margin of Safety", "VIX"]
    html += _glossary_section_html(used_terms)

    html += '<footer class="db-footer">'
    html += '<p class="db-meta-line">出典: ai_track_record.json (予測履歴) / master_prediction_history.json (Master Wisdom 記録) / walk_forward_results.json (実戦検証)</p>'
    html += '<p class="db-meta-line">この週次レポートは毎週土曜日に自動更新され、Slack DM でも配信されます。'
    html += '長期保有を前提とした自己進化型レポートです。</p>'
    html += '</footer></article>'
    return html


def build_monthly_brief_html(results: dict, track: dict, holdings: dict) -> str:
    """📊 月次ブリーフ — 過去30日の総括 + 来月の戦略.

    月次パフォーマンス、ファクター進化、リスク管理の総合評価。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    month = datetime.now().strftime("%Y年%m月")

    records = track.get("records", [])
    evaluations = track.get("evaluations", [])
    last_30_evals = evaluations[-30:] if evaluations else []

    # パフォーマンス履歴
    perf_path = BASE_DIR / "data" / "performance_history.json"
    perf_history = []
    if perf_path.exists():
        try:
            with open(perf_path, encoding="utf-8") as f:
                perf_history = json.load(f).get("records", [])
        except (json.JSONDecodeError, OSError):
            pass

    master_learning = results.get("master_learning", {}) or {}
    factor_accs = master_learning.get("factor_accuracies", {}) or {}

    # 月次パフォーマンス
    monthly_change_pct = None
    monthly_pnl_change = None
    if len(perf_history) >= 22:  # 約1ヶ月
        first = perf_history[-22]
        last = perf_history[-1]
        if first.get("total_usd", 0) > 0:
            monthly_change_pct = (last["total_usd"] - first["total_usd"]) / first["total_usd"] * 100
            monthly_pnl_change = last.get("total_pnl_usd", 0) - first.get("total_pnl_usd", 0)

    html = '<article class="daily-brief"><header class="db-header">'
    html += '<div class="db-eyebrow">MONTHLY REPORT — 月次総括</div>'
    html += f'<h1 class="db-title">{month}のポートフォリオ総括</h1>'
    html += '<div class="db-meta">'
    html += f'<span class="db-meta-item">📅 報告期間: 過去30日</span>'
    html += f'<span class="db-meta-item">🧠 採点済み予測 <strong>{len(last_30_evals)}件</strong></span>'
    html += '</div></header>'

    # === 長期保有方針バナー ===
    html += '<div class="db-longterm-banner">'
    html += '<span class="icon">🏛️</span>'
    html += '<div><strong>月次レビューの位置づけ — 長期保有の進捗確認</strong>'
    html += '<p>1ヶ月は長期保有 (3〜10年) のうちの<strong>ほんの一部</strong>です。'
    html += '月次パフォーマンスが良くても悪くても、「長期ストーリーが崩れていないか」を最重要視します。'
    html += '短期の上げ下げに振り回されず、<strong>10年後の自分が今日のポートフォリオを見て後悔しないか</strong>を毎月確認します。</p></div></div>'

    # === 月次パフォーマンスサマリー ===
    html += '<section class="db-section">'
    html += '<h2 class="db-section-title">📊 月次パフォーマンス</h2>'
    if monthly_change_pct is not None:
        color = "var(--accent-green)" if monthly_change_pct >= 0 else "var(--accent-red)"
        sign = "+" if monthly_change_pct >= 0 else ""
        html += f'<p class="db-lead">過去30日でポートフォリオ総額は <strong style="color:{color}">{sign}{monthly_change_pct:.1f}%</strong> 変動しました。'
        if monthly_pnl_change is not None:
            html += f'含み損益は <strong style="color:{color}">{sign}${monthly_pnl_change:,.0f}</strong> 推移。'
        html += 'これを ' + _explain("Drawdown") + ' (過去最高値からの落ち込み) と比べて健全性を評価します。</p>'
    else:
        html += '<p class="db-lead">パフォーマンス履歴がまだ十分に蓄積されていません。1ヶ月の継続運用後に自動表示されます。</p>'

    if last_30_evals:
        avg_dir = sum(e.get("predicted_vs_actual", {}).get("direction_accuracy", 0)
                      for e in last_30_evals) / len(last_30_evals)
        avg_rmse = sum(e.get("predicted_vs_actual", {}).get("rmse", 0)
                       for e in last_30_evals) / len(last_30_evals)
        html += '<aside class="db-source">'
        html += '<strong>🎯 月次予測精度</strong>'
        html += (f'<p>過去30日の予測の方向一致率: <strong>{avg_dir*100:.1f}%</strong> '
                 f'(50%が偶然、それを超える分が「AIの実力」)。'
                 f'平均誤差 (RMSE): <strong>{avg_rmse:.4f}</strong></p>')
        html += '<p>方向だけでなく「どれだけ動くか」も評価しているため、RMSE が小さいほど予測の解像度が高いことを示します。</p>'
        html += '<p>出典: ai_track_record.json (evaluations セクション)</p>'
        html += '</aside>'
    html += '</section>'

    # === ファクター進化の月次トレンド ===
    if factor_accs:
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">⚙️ ファクター進化の1ヶ月</h2>'
        html += '<p class="db-body">9ファクターそれぞれが過去1ヶ月でどう「育った」かを表で示します。'
        html += 'ウェイトの変化は「市場が何を要求するようになったか」のシグナルです。</p>'
        factor_jp = {
            "quality_roe": "経営効率 (ROE)",
            "quality_margin": "利益率の質",
            "value_earnings_yield": "Earnings Yield",
            "value_fcf_yield": "FCF Yield",
            "value_margin_of_safety": "安全域 (DCF)",
            "momentum_composite": "テクニカル統合",
            "contrarian_fear_greed": "Fear-Greed逆張り",
            "contrarian_insider_pulse": "インサイダー脈動",
            "risk_kelly": "Kelly基準",
        }
        html += '<table class="db-factor-table"><tr><th>ファクター</th><th>精度</th><th>意味</th></tr>'
        for fname, acc in sorted(factor_accs.items(), key=lambda x: -x[1]):
            meaning = ("市場の中核ドライバー" if acc >= 0.7
                       else "概ね機能" if acc >= 0.6
                       else "中立的シグナル" if acc >= 0.5
                       else "現在の局面では効きにくい")
            html += f'<tr><td>{factor_jp.get(fname, fname)}</td><td>{acc*100:.1f}%</td><td>{meaning}</td></tr>'
        html += '</table>'
        html += '</section>'

    # === 大暴落歴史との比較 ===
    patterns_path = BASE_DIR / "data" / "historical_patterns.json"
    patterns = {}
    if patterns_path.exists():
        try:
            with open(patterns_path, encoding="utf-8") as f:
                patterns = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    vix_stats = patterns.get("vix_regimes", {})
    if vix_stats:
        cur_pct = vix_stats.get("current_vix_percentile", 50)
        html += '<section class="db-section">'
        html += '<h2 class="db-section-title">📚 30年歴史的視点</h2>'
        html += '<p class="db-body">過去30年の市場データから、現在の位置を統計的に評価します。'
        html += 'バフェット哲学:「歴史は完全には繰り返されないが、韻を踏む」</p>'
        html += (f'<p class="db-body">現在の ' + _explain("VIX") +
                 f' は過去30年で <strong>{cur_pct} パーセンタイル</strong> です。'
                 f'これは「過去30年の{cur_pct}%の日々がこれより低かった」という意味です。'
                 '高すぎる時は恐怖局面で「皆が恐れる時に貪欲になれ」(バフェット)、'
                 '低すぎる時は楽観局面で「皆が貪欲な時に恐れろ」と判断します。</p>')

        crashes = patterns.get("major_crashes", {})
        if crashes.get("n_events", 0) > 0:
            html += (f'<p class="db-body">過去30年では <strong>{crashes["n_events"]}回</strong> の大暴落 (60日以内に -15% 以上) を観測。'
                     f'平均深さは <strong>{crashes.get("avg_depth", 0)*100:.0f}%</strong>、'
                     f'平均回復日数は <strong>{crashes.get("avg_recovery_days", 0):.0f}日</strong>。'
                     '主要イベント: 1987 Black Monday / 2000 Dot-com / 2008 GFC / 2020 COVID 等。</p>')

        buffett = patterns.get("buffett_contrarian_validation", {})
        h_252 = buffett.get("by_horizon", {}).get("252d")
        if h_252:
            mean = h_252.get("mean_return", 0) * 100
            wr = h_252.get("win_rate", 0) * 100
            html += '<aside class="db-source">'
            html += '<strong>🎯 バフェット逆張りの統計検証</strong>'
            html += (f'<p>過去30年で「VIX > 30 (恐怖局面) で買った場合、1年後に何%儲かったか」を検証した結果、'
                     f'平均 <strong>{mean:+.1f}%</strong>、勝率 <strong>{wr:.0f}%</strong>。'
                     '「他人が恐れる時に買う」という哲学はデータでも裏付けられています。</p>')
            html += '<p>出典: data/historical_patterns.json (1990年代以降のSP500+VIX)</p>'
            html += '</aside>'
        html += '</section>'

    # === 来月の戦略 ===
    html += '<section class="db-section">'
    html += '<h2 class="db-section-title">🎯 来月の戦略指針</h2>'
    regime = results.get("regime", "unknown")
    if regime == "low_vol":
        strategy = ("低ボラ局面では <strong>Quality + Momentum</strong> を重視します。"
                    "経営効率の高い企業 (ROE > 20%) を中心にトレンドフォロー。"
                    "Cross-Sectional で相対的に強い銘柄に集中。")
    elif regime == "transition":
        strategy = ("移行期は <strong>Value + Risk管理</strong> を重視します。"
                    "DCFで割安と判断される銘柄 (安全域あり) を選別し、"
                    "Kelly基準で過剰なポジションを避けます。")
    elif regime == "crisis":
        strategy = ("危機局面では <strong>Contrarian + Margin of Safety</strong> を最大化します。"
                    "バフェット流「皆が恐れる時に買う」を発動。"
                    "ただし内在価値が確認できる優良銘柄に限定。")
    else:
        strategy = "市場局面の判定が確定次第、最適な戦略を提示します。"
    html += f'<p class="db-body">{strategy}</p>'
    html += '<p class="db-body">この戦略は固定ではなく、毎日の予測精度フィードバックで自動調整されます。'
    html += '<strong>1ヶ月後の振り返り</strong>でこの判断が正しかったかが採点され、次回の戦略に反映されます。</p>'
    html += '</section>'

    # === 非保有の長期候補 (月次ウォッチリスト) ===
    latest_m = records[-1] if records else {}
    held_m = {s["ticker"] for s in build_holdings_data(holdings)}
    new_picks_m = _filter_new_picks_excluding_held(latest_m.get("new_picks", []), held_m)
    if new_picks_m:
        html += '<section class="db-section">'
        html += f'<h2 class="db-section-title">🆕 月次ウォッチ — 非保有の長期投資候補 ({len(new_picks_m)}銘柄)</h2>'
        html += '<p class="db-body">Master Wisdom が今月「10年保有を前提に検討すべき」と評価した非保有銘柄です。'
        html += '今すぐ買う必要はなく、<strong>「割安まで下がるのを待つ」</strong>ことが規律です。</p>'
        html += '<ul class="db-list">'
        for p in new_picks_m[:8]:
            t = p.get("ticker", "")
            sector = p.get("sector", "")
            reason = p.get("reason", "")
            horizon = p.get("horizon", "3〜10年")
            html += (f'<li><strong>{html_mod.escape(t)}</strong> '
                     f'<span class="db-sector-tag">{html_mod.escape(sector)}</span> '
                     f'<span class="db-master-tag">想定保有: {html_mod.escape(horizon)}</span><br>'
                     f'<span style="color:var(--text-secondary);font-size:14px">{html_mod.escape(reason)}</span></li>')
        html += '</ul>'
        html += '<aside class="db-source"><strong>📌 「買う」と「待つ」の判断基準</strong>'
        html += ('<p>非保有候補は<strong>すぐ買わない</strong>のが原則です。以下の条件を満たしたときに実際の購入を検討します:</p>'
                 '<ol><li>株価が <strong>内在価値の70%以下</strong> (' + _explain("Margin of Safety") + ' ≥ 30%)</li>'
                 '<li><strong>' + _explain("VIX") + ' > 25</strong> でマーケット全体が恐怖局面</li>'
                 '<li>個別株の <strong>30日リターン &lt; -15%</strong> で短期過剰売り</li>'
                 '<li>長期ストーリー (経済的堀・需要・財務) に変化がない</li></ol>')
        html += '<p>これらが揃わなければ、ウォッチリストに置いたまま「待つ」のが長期投資の規律です。</p>'
        html += '<p>出典: master_predictor.py (スクリーニング) / data/master_screening_results.json</p>'
        html += '</aside></section>'

    # === 10年ストーリーチェックリスト ===
    html += '<section class="db-section">'
    html += '<h2 class="db-section-title">📋 月次「10年ストーリー」チェックリスト</h2>'
    html += '<p class="db-body">保有銘柄について、毎月以下の項目を確認します。'
    html += '<strong>1つでも崩れたら売却検討</strong>、すべて健在なら長期保有を継続します:</p>'
    html += '<ol class="db-list">'
    html += '<li><strong>経済的堀 (Economic Moat) は無傷か</strong> — ブランド・ネットワーク効果・スイッチコスト・規模の経済が劣化していないか</li>'
    html += '<li><strong>10年後も需要が見える事業か</strong> — 業界構造の根本変化 (EV/AI/規制) で陳腐化していないか</li>'
    html += '<li><strong>経営陣の質と資本配分</strong> — 自社株買い・配当・再投資が株主視点で行われているか</li>'
    html += '<li><strong>財務健全性</strong> — 過剰負債・非合理な買収・会計の不審な変化がないか</li>'
    html += '<li><strong>株価が内在価値の2倍を超えていないか</strong> — 極端な過大評価なら一部利確検討</li>'
    html += '</ol>'
    html += '<p class="db-body">これらは「定量的シグナル」を超えた<strong>定性的判断</strong>です。'
    html += 'AI が完全に自動判定できる部分ではないため、毎月人間が目を通すべき項目として明示しています。</p>'
    html += '</section>'

    # 用語集
    used_terms = ["VIX", "ROE", "PER", "FCF Yield", "DCF", "Margin of Safety",
                  "Sharpe", "Drawdown", "Kelly基準", "Cross-Sectional", "Mean Reversion"]
    html += _glossary_section_html(used_terms)

    html += '<footer class="db-footer">'
    html += '<p class="db-meta-line"><strong>📖 月次レポートの作り方</strong></p>'
    html += '<ul class="db-source-list">'
    html += '<li>パフォーマンス履歴: data/performance_history.json (yfinanceから日次更新)</li>'
    html += '<li>予測評価: ai_track_record.json (7日経過後に自動採点)</li>'
    html += '<li>30年歴史: data/historical_patterns.json (S&P500 + VIX 1990-現在)</li>'
    html += '<li>ファクター学習: data/master_factor_weights.json (Bayesian更新)</li>'
    html += '</ul>'
    html += '<p class="db-meta-line">この月次レポートは毎月1日に自動更新されます。</p>'
    html += '</footer></article>'
    return html


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

    # 日本株 — 全サブセクション (nisa_growth / nisa_tsumitate / tokutei 等) を読む
    jp_account_label = {
        "nisa_growth": "NISA成長",
        "nisa_tsumitate": "NISA積立",
        "tokutei": "特定",
        "general": "一般",
    }
    jp_section = holdings.get("japan_stocks", {}) or {}
    for sub_key, positions in jp_section.items():
        if not isinstance(positions, list):
            continue
        for s in positions:
            shares = s.get("shares", 0) or 0
            cost = float(s.get("cost_basis_jpy", 0) or 0)
            price = float(s.get("current_price_jpy", 0) or 0)
            mv = price * shares
            pnl = (price - cost) * shares if cost > 0 else 0
            pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
            code = s.get("code", "") or s.get("ticker", "").replace(".T", "")
            rows.append({
                "ticker": code,
                "sector": get_sector(f"{code}.T"),
                "account": jp_account_label.get(sub_key, sub_key),
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
    colors = ["#0075de", "#213183", "#1aae39", "#2a9d99", "#dd5b00", "#ff64c8", "#c08532", "#391c57"]
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
        is_profit = pnl_usd >= 0
        pnl_class = "pnl-positive" if is_profit else "pnl-negative"
        pnl_sign = "+" if is_profit else "-"
        arrow = "▲" if is_profit else "▼"
        cur_sym = "$" if r["currency"] == "USD" else "¥"
        url = yahoo_chart_url(r["ticker"] if r["currency"] == "USD" else f"{r['ticker']}.T")
        row_class = "row-profit" if is_profit else "row-loss"

        html += f'<tr class="{row_class}">'
        html += f'<td class="port-ticker"><strong>{r["ticker"]}</strong></td>'
        html += f'<td>{r["sector"]}</td>'
        html += f'<td><span class="acct-badge">{r["account"]}</span></td>'
        html += f'<td data-num="{r["shares"]}">{r["shares"]:,}</td>'
        html += f'<td data-num="{r["cost"]}">{cur_sym}{r["cost"]:,.2f}</td>'
        html += f'<td data-num="{r["price"]}">{cur_sym}{r["price"]:,.2f}</td>'
        html += f'<td data-num="{mv_usd}">${mv_usd:,.0f}</td>'
        html += f'<td data-num="{pnl_usd}" class="{pnl_class} pnl-cell"><span class="pnl-arrow-sm">{arrow}</span>{pnl_sign}${abs(pnl_usd):,.0f}</td>'
        html += f'<td data-num="{r["pnl_pct"]}" class="{pnl_class} pnl-cell">{pnl_sign}{abs(r["pnl_pct"]):.2f}%</td>'
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

    # === Learning 解説 ===
    html += '<div class="explainer-box">'
    html += '<div class="explainer-title">🧠 自己学習とはどういう仕組みか</div>'
    html += '<p>このシステムは <strong>「予測 → 答え合わせ → 修正」</strong>のループを毎日自動で回しています。'
    html += '人間が手動でパラメータを調整しなくても、過去の的中率を見て自動でアルゴリズムが改善されていきます。</p>'
    html += '<div class="explainer-title" style="margin-top:14px;font-size:14px">学習サイクルの流れ</div>'
    html += '<ol class="explainer-list">'
    html += '<li><strong>毎朝7時に予測を出す</strong>: 各銘柄について「BUY/HOLD/SELL」と確信度 (0〜1)、'
    html += 'リスクシナリオ (Bull/Base/Bear) を <code>ai_track_record.json</code> に保存。</li>'
    html += '<li><strong>7日後・30日後に自動採点</strong>: 当時の予測と実際の値動きを照合。'
    html += '「BUYと言った銘柄は本当に上がったか?」「Bullシナリオの確率は妥当だったか?」を計測。</li>'
    html += '<li><strong>方向一致率 / RMSE / Calibration を計算</strong>: 採点結果を <code>evaluations</code> として記録。</li>'
    html += '<li><strong>各ファクターの精度に応じてウェイトを Bayesian 更新</strong>: '
    html += '当たれば信頼度↑、外れれば信頼度↓。</li>'
    html += '<li><strong>その日の「学び」を学習ジャーナルに記録</strong>: 「どのファクターが効いたか」「どの局面で外したか」を文字で残す。</li>'
    html += '</ol>'
    html += '<div class="explainer-title" style="margin-top:14px;font-size:14px">主要指標の意味</div>'
    html += '<dl class="explainer-dl">'
    html += '<dt>方向一致率 (Direction Accuracy)</dt>'
    html += '<dd>BUYと予測した銘柄が実際に上昇したか / SELLが下落したかの的中率。<strong>50%は偶然、'
    html += '60%超で実力あり、70%超は卓越</strong>。これが最も基本的な「AI が市場を読めているか」の指標。</dd>'
    html += '<dt>RMSE (二乗平均誤差)</dt>'
    html += '<dd>予測リターンと実際のリターンの「平均的なズレ」。'
    html += '値が小さいほど精密に予測できている。0.05 (5%) なら「平均5%の誤差」。'
    html += '方向だけでなく「どれだけ動くか」の解像度を測る。</dd>'
    html += '<dt>Calibration Score</dt>'
    html += '<dd>確信度の正しさ。「確信度70%と言った時、実際に70%の確率で当たっているか」を測る。'
    html += '<strong>1.0 が完璧、0.7 でも実用的、0.5未満はキャリブレーション要</strong>。'
    html += 'AI が <strong>謙虚に予測しているか</strong>を表す。</dd>'
    html += '</dl>'
    html += '<p class="explainer-takeaway">💡 <strong>なぜ7日後/30日後の2段階か?</strong> '
    html += '7日は「短期のノイズ」、30日は「中期のトレンド」の答え合わせ。'
    html += '異なる時間軸でファクターの効き方が違うため、両方測ることで「どのファクターが短期向き / 長期向き」が分かります。'
    html += '長期保有方針なので、特に <strong>30日採点</strong>を重視しています。</p>'
    html += '<p class="explainer-source">📂 予測記録: <code>ai_track_record.json</code> / '
    html += '学習エントリ: <code>data/learning_journal.json</code> / '
    html += 'ファクター精度: <code>data/master_factor_weights.json</code></p>'
    html += '</div>'

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
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{bar_pct:.0f}%;background:var(--accent)"></div></div>'
            html += f'<div class="sector-bar-value">精度 {acc:.0f}% / W {weight:.2f} {delta_str}<span class="sector-pct"> ({ev}件)</span></div>'
            html += '</div>'
        html += '</div></div>'

    # 最新の notable findings
    findings = learning.get("notable_findings", [])
    if findings:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">💡</span> 本日の学び</div>'
        html += '<div class="action-grid" style="grid-template-columns:1fr">'
        for f in findings:
            html += f'<div class="action-card" style="border-left:2px solid var(--accent)">'
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


def build_history_html(results: dict) -> str:
    """📚 History タブ — 30年歴史パターンと現在位置."""
    patterns_path = BASE_DIR / "data" / "historical_patterns.json"
    if not patterns_path.exists():
        return ('<div class="section"><p style="color:var(--text-muted);text-align:center;padding:40px">'
                '歴史パターンデータがまだありません。<br>'
                '<code>python historical_pattern_extractor.py</code> を一度実行してください。</p></div>')
    try:
        with open(patterns_path, encoding="utf-8") as f:
            patterns = json.load(f)
    except (json.JSONDecodeError, OSError):
        return '<div class="section"><p>歴史データ読み込みエラー</p></div>'

    html = '<div class="section"><div class="section-title"><span class="icon">📚</span> 30年の歴史から学ぶ — Buffett 流統計</div>'

    # === タブ全体の趣旨説明 ===
    html += '<div class="explainer-box">'
    html += '<div class="explainer-title">📖 このページの目的</div>'
    html += '<p>このタブは <strong>「過去30年の市場で何が起きたか」</strong>を統計的に振り返り、'
    html += '今のマーケットがその歴史のどこに位置するかを把握するためのものです。</p>'
    html += '<blockquote class="explainer-quote">'
    html += '「歴史は完全には繰り返されないが、韻を踏む」 ― マーク・トウェイン (バフェットがしばしば引用)'
    html += '</blockquote>'
    html += '<p>ウォーレン・バフェットは「市場のタイミングを当てる天才」ではありません。'
    html += '<strong>彼が実際にやってきたのは「歴史的な平均と比べて今が割高か割安かを判断すること」</strong>です。'
    html += 'このページは同じことを統計データで再現します:</p>'
    html += '<ul class="explainer-list">'
    html += '<li><strong>大暴落の頻度と深さ</strong> ― 「-30% の下落は10年に1回」のような感覚を数字で持つ</li>'
    html += '<li><strong>恐怖指数 (VIX) の分布</strong> ― 今の VIX が「ほぼ毎日の値」なのか「30年に数回しかない異常値」なのか</li>'
    html += '<li><strong>「皆が恐れている時に買えば儲かったか」の検証</strong> ― バフェットの哲学を実データで答え合わせ</li>'
    html += '<li><strong>個別銘柄の長期成績</strong> ― 30年保有していた場合の年率リターン・最大下落幅</li>'
    html += '</ul>'
    html += '<p class="explainer-source">📂 データソース: Yahoo Finance (yfinance ライブラリ) で取得した S&amp;P500 (^GSPC) と VIX (^VIX) の1990年代以降の日次データ。'
    html += '計算は <code>historical_pattern_extractor.py</code> が行い、結果を <code>data/historical_patterns.json</code> に保存しています。</p>'
    html += '</div>'

    # 大暴落イベント
    crashes_data = patterns.get("major_crashes", {})
    crashes = crashes_data.get("events", [])
    if crashes:
        avg_depth = crashes_data.get("avg_depth", 0) * 100
        avg_recovery = crashes_data.get("avg_recovery_days", 0)
        html += '<div class="explainer-box explainer-sub">'
        html += '<div class="explainer-title">💥 「大暴落」の定義と読み方</div>'
        html += '<p>ここでの<strong>「大暴落」</strong>とは、S&amp;P500 が <strong>60日 (約3ヶ月) 以内に高値から -15% 以上下落した</strong>イベントを指します。'
        html += '日々のニュースで「下落」と騒がれるレベルではなく、<strong>歴史的に見て本当に痛手だった出来事</strong>のみカウントしています。</p>'
        html += '<p>下の3つのカードの読み方:</p>'
        html += '<ul class="explainer-list">'
        html += '<li><strong>大暴落イベント数</strong> = 過去30年に何回起きたか。だいたい「6〜10回」が普通の数字です。'
        html += 'つまり <strong>3〜5年に1回</strong>はこういう局面が来ます。</li>'
        html += '<li><strong>平均深さ</strong> = 高値からどれだけ下げたか。-25% 〜 -35% 程度になることが多いです。'
        html += 'これは「最悪の場合これくらい覚悟しておく」という心理的準備の指標。</li>'
        html += '<li><strong>平均回復日数</strong> = 底値から元の高値まで戻るのにかかった営業日数。'
        html += '長いほど「精神的に我慢が必要」を意味します (200営業日 = 約9ヶ月)。</li>'
        html += '</ul>'
        html += '<p class="explainer-takeaway">💡 <strong>長期投資家にとっての示唆</strong>: '
        html += 'これだけ大暴落があっても、長期保有していれば指数は新高値を更新し続けてきました。'
        html += '「いつ来るか」ではなく「来たときに売らない覚悟」が長期リターンの源泉です。</p>'
        html += '</div>'
        html += '<div class="port-summary-grid">'
        html += f'<div class="port-metric"><div class="port-metric-label">大暴落イベント</div>'
        html += f'<div class="port-metric-value">{len(crashes)}</div>'
        html += f'<div class="port-metric-sub">過去30年 (-15%超)</div></div>'
        html += f'<div class="port-metric"><div class="port-metric-label">平均深さ</div>'
        html += f'<div class="port-metric-value" style="color:var(--accent-red)">{avg_depth:.1f}%</div>'
        html += f'<div class="port-metric-sub">底値時点</div></div>'
        html += f'<div class="port-metric"><div class="port-metric-label">平均回復日数</div>'
        html += f'<div class="port-metric-value">{avg_recovery:.0f}日</div>'
        html += f'<div class="port-metric-sub">底値→前高値</div></div>'

        cycles = patterns.get("bull_bear_cycles", {})
        if cycles:
            html += f'<div class="port-metric"><div class="port-metric-label">ブル/ベア</div>'
            html += f'<div class="port-metric-value">{cycles.get("n_bull", 0)} / {cycles.get("n_bear", 0)}</div>'
            html += f'<div class="port-metric-sub">平均{cycles.get("avg_bull_duration_days", 0):.0f}日 / {cycles.get("avg_bear_duration_days", 0):.0f}日</div></div>'
        html += '</div>'

    # 主要暴落リスト
    if crashes:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">💥</span> 過去の主要暴落 (深い順)</div>'
        html += '<div class="explainer-text">'
        html += '<p>過去30年で実際に起きた暴落を深さ順に並べました。'
        html += '代表的なイベント: <strong>1987年ブラックマンデー</strong> (-22% の1日)、<strong>2000-2002年ドットコム崩壊</strong> (-49% / 約2年半)、'
        html += '<strong>2008年リーマンショック</strong> (-57% / 約1年半)、<strong>2020年コロナショック</strong> (-34% / 約1ヶ月)。</p>'
        html += '<p><strong>列の意味</strong>:</p>'
        html += '<ul class="explainer-list">'
        html += '<li><strong>概算年</strong>: 暴落が始まった年。複数の年にまたがる場合は開始年。</li>'
        html += '<li><strong>深さ</strong>: 高値からの最大下落率 (マイナス値)。-30% なら 100万円が70万円になった意味。</li>'
        html += '<li><strong>下落期間</strong>: 高値→底値までの営業日数。短いほど急落 (パニック型)、長いほどジワ下げ (構造的弱気相場)。</li>'
        html += '<li><strong>回復日数</strong>: 底値→次の新高値までの日数。「未回復」なら今もまだ過去高値に戻っていない (古いデータでは稀)。</li>'
        html += '</ul>'
        html += '</div>'
        sorted_crashes = sorted(crashes, key=lambda c: c.get("depth", 0))[:10]
        html += '<div class="port-table-section"><div class="port-table-wrap"><table class="port-table">'
        html += '<thead><tr><th>概算年</th><th>深さ</th><th>下落期間</th><th>回復日数</th></tr></thead><tbody>'
        for c in sorted_crashes:
            depth = c.get("depth", 0) * 100
            yr = c.get("approx_year", "?")
            dur = c.get("duration_days", 0)
            rec = c.get("recovery_days")
            rec_str = f"{rec}日" if rec else "未回復"
            html += f'<tr><td>{yr}</td><td style="color:var(--accent-red)">{depth:.1f}%</td>'
            html += f'<td>{dur}日</td><td>{rec_str}</td></tr>'
        html += '</tbody></table></div></div>'

    # VIX 分布 + 現在位置
    vix_stats = patterns.get("vix_regimes", {})
    if vix_stats:
        pct = vix_stats.get("vix_percentiles", {})
        cur_pct = vix_stats.get("current_vix_percentile", 50)
        html += '<div class="explainer-box explainer-sub" style="margin-top:20px">'
        html += '<div class="explainer-title">📊 VIX (恐怖指数) とは何か</div>'
        html += '<p><strong>VIX</strong> は <strong>「向こう30日で S&amp;P500 がどれだけ大きく動くと市場が予想しているか」</strong>を、'
        html += 'オプション市場の値段から逆算した数字です。VIX = 20 なら「年率20%の標準偏差で動くと予想」を意味します。</p>'
        html += '<p>感覚的な目安:</p>'
        html += '<ul class="explainer-list">'
        html += '<li><strong>VIX 12 〜 16</strong>: 非常に平穏。投資家が安心しきっている。バフェット流では <strong>「皆が貪欲な時は恐れろ」</strong>のサイン</li>'
        html += '<li><strong>VIX 17 〜 22</strong>: 通常レベル。これが「平常時の日常」</li>'
        html += '<li><strong>VIX 23 〜 30</strong>: 警戒レベル。何か不穏なニュースが出ている</li>'
        html += '<li><strong>VIX 30 〜 40</strong>: パニック開始。下落相場の中盤</li>'
        html += '<li><strong>VIX 40+</strong>: 極度の恐怖。歴史的には買い場 (バフェットが動く局面)</li>'
        html += '</ul>'
        html += '<p>下のグラフは過去30年で VIX が <strong>各水準を超えた頻度</strong>です。'
        html += '「中央値」は <strong>「だいたい毎日このあたりの値」</strong>、'
        html += '「99%」は <strong>「30年に数日しかなかった超パニック値」</strong>を意味します。</p>'
        html += '</div>'
        html += '<div class="sector-section" style="margin-top:20px">'
        html += '<div class="sector-title">📊 VIX 30年分布 + 現在位置</div>'
        html += '<div class="sector-bars">'
        for label, key in [("中央値 (50%)", "p50"), ("75%", "p75"), ("90%", "p90"), ("95%", "p95"), ("99%", "p99"), ("最大観測", "max_observed")]:
            v = pct.get(key, 0)
            html += f'<div class="sector-bar-row">'
            html += f'<div class="sector-bar-label">{label}</div>'
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{min(100, v*2):.0f}%;background:var(--accent)"></div></div>'
            html += f'<div class="sector-bar-value">VIX {v:.1f}</div>'
            html += '</div>'
        html += '</div>'
        html += f'<p style="margin-top:12px;color:var(--text-muted);font-size:13px">現在の VIX は過去30年で <strong style="color:var(--text-primary)">{cur_pct} パーセンタイル</strong></p>'

        freqs = vix_stats.get("frequencies", {})
        if freqs:
            html += '<div class="sector-bars" style="margin-top:12px">'
            for r, jp in [("low_vol", "低ボラ (安定)"), ("transition", "移行期"), ("crisis", "危機")]:
                f = freqs.get(r, 0) * 100
                html += f'<div class="sector-bar-row">'
                html += f'<div class="sector-bar-label">{jp}</div>'
                html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{f:.0f}%;background:var(--accent-bright)"></div></div>'
                html += f'<div class="sector-bar-value">{f:.0f}% の日</div>'
                html += '</div>'
            html += '</div>'
        html += '</div>'

    # Buffett 逆張り検証
    buffett = patterns.get("buffett_contrarian_validation", {})
    horizons = buffett.get("by_horizon", {})
    if horizons:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">🎯</span> Buffett 逆張り検証 — VIX&gt;30 で買えば...</div>'
        html += '<div class="explainer-box">'
        html += '<div class="explainer-title">🎯 この検証の意味 — バフェット哲学を実データで答え合わせ</div>'
        html += '<blockquote class="explainer-quote">'
        html += '「他の人が貪欲な時は恐れ、他の人が恐れている時に貪欲になれ」 ― ウォーレン・バフェット'
        html += '</blockquote>'
        html += '<p><strong>バフェットがよく言うこの言葉、実は本当に儲かるのか?</strong> — '
        html += '過去30年のデータで以下の手順で検証しました:</p>'
        html += '<ol class="explainer-list">'
        html += '<li><strong>「皆が恐れている瞬間」を VIX > 30 と定義</strong> '
        html += '(VIX 30 超えは1990年以降で全営業日の数%しか発生していない<strong>稀な恐怖局面</strong>)</li>'
        html += '<li><strong>そういう日に S&amp;P500 を買ったと仮定</strong> '
        html += '(2008/10/10、2020/3/16 などが該当)</li>'
        html += '<li><strong>その後 1ヶ月 / 3ヶ月 / 1年 / 2年 / 5年 持ち続けた場合のリターンを計算</strong></li>'
        html += '<li><strong>すべての該当日について平均値を取り、勝率 (利益が出た日の割合) を集計</strong></li>'
        html += '</ol>'
        html += '<p><strong>下の表の各列の読み方</strong>:</p>'
        html += '<dl class="explainer-dl">'
        html += '<dt>ホライズン</dt>'
        html += '<dd>「VIX > 30 の日に買って、何日 (何ヶ月) 持ったか」を表します。'
        html += '<strong>1ヶ月</strong> ≈ 21営業日、<strong>1年</strong> ≈ 252営業日 が目安です。</dd>'
        html += '<dt>平均リターン</dt>'
        html += '<dd>「該当する全ての日について、買って X 期間後に売ったとしたら平均で何%儲かったか」。'
        html += '<strong style="color:var(--accent-green)">緑のプラス値</strong>なら「歴史的には儲かった」、'
        html += '<strong style="color:var(--accent-red)">赤のマイナス</strong>なら「歴史的には損が出た」を意味します。</dd>'
        html += '<dt>勝率</dt>'
        html += '<dd>「該当する全ての日について、X 期間後にプラスで終わっていた日の割合」。'
        html += '<strong>50%</strong>なら偶然と同じ (コイン投げと変わらない)、'
        html += '<strong>60%超</strong>で「明らかに有利」、<strong>80%超</strong>で「ほぼ確実に勝てる」歴史的バイアスありを示します。</dd>'
        html += '<dt>最高 / 最低</dt>'
        html += '<dd>「全サンプルの中で一番うまくいった場合」と「一番ひどかった場合」。'
        html += '最高値が大きいほど「うまくハマるとリターンが伸びる」、'
        html += '最低値が浅い (マイナス幅が小さい) ほど「失敗してもダメージが限定的」を意味します。</dd>'
        html += '<dt>サンプル</dt>'
        html += '<dd>「VIX > 30 だった日が30年で何日あったか」。'
        html += 'サンプルが多いほど統計の信頼性が上がります。'
        html += '長期ホライズン (5年) では「5年経過後のデータ」が必要なので、新しい暴落 (2020年コロナなど) は<strong>サンプルから除外</strong>される点に注意 (5年ホライズンのサンプルは少なくなりがち)。</dd>'
        html += '</dl>'
        html += '<p class="explainer-takeaway">💡 <strong>典型的な結果のパターン</strong>: '
        html += '1ヶ月だと勝率 50〜60% (短期はノイズが多い)、'
        html += '<strong>1年保有すると勝率 80%超・平均 +15〜20%</strong> になることが多いです。'
        html += '「皆が恐れる時に買って1年待つ」がバフェット哲学のコアであり、これがデータで裏付けられます。</p>'
        html += '<p class="explainer-source">📂 計算: <code>historical_pattern_extractor.py</code> の <code>validate_buffett_contrarian()</code> 関数。'
        html += 'データ範囲: <code>data/historical_patterns.json</code> の生成日時に依存 (毎日更新)。</p>'
        html += '</div>'
        html += '<div class="port-table-section"><div class="port-table-wrap"><table class="port-table">'
        html += '<thead><tr><th>ホライズン</th><th>平均リターン</th><th>勝率</th><th>最高</th><th>最低</th><th>サンプル</th></tr></thead><tbody>'
        h_jp = {"30d": "1ヶ月", "90d": "3ヶ月", "252d": "1年", "504d": "2年", "1260d": "5年"}
        for hkey, h_data in horizons.items():
            mean_ret = h_data.get("mean_return", 0) * 100
            wr = h_data.get("win_rate", 0) * 100
            mx = h_data.get("max", 0) * 100
            mn = h_data.get("min", 0) * 100
            n = h_data.get("n_observations", 0)
            mean_color = "var(--accent-green)" if mean_ret > 0 else "var(--accent-red)"
            wr_color = "var(--accent-green)" if wr > 60 else "var(--accent-yellow)" if wr > 50 else "var(--accent-red)"
            html += f'<tr><td>{h_jp.get(hkey, hkey)}</td>'
            html += f'<td style="color:{mean_color}">{mean_ret:+.1f}%</td>'
            html += f'<td style="color:{wr_color}">{wr:.0f}%</td>'
            html += f'<td style="color:var(--accent-green)">{mx:+.1f}%</td>'
            html += f'<td style="color:var(--accent-red)">{mn:+.1f}%</td>'
            html += f'<td>{n}</td></tr>'
        html += '</tbody></table></div></div>'
        validation = buffett.get("buffett_validation", "")
        if validation:
            html += f'<p style="margin-top:8px;color:var(--text-secondary);font-size:13px;font-style:italic">{html_mod.escape(validation)}</p>'

    # 個別銘柄長期統計
    ticker_stats = patterns.get("ticker_long_term_stats", {})
    if ticker_stats:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">📈</span> 個別銘柄 長期統計</div>'
        html += '<div class="explainer-box explainer-sub">'
        html += '<div class="explainer-title">📈 各銘柄を「長期で持ち続けたら」どうなったか</div>'
        html += '<p>保有銘柄それぞれについて、<strong>過去のデータ全期間で買い持ち (Buy &amp; Hold) した場合の成績</strong>を計算しました。'
        html += 'これはバフェット流の「長期保有が本当に報われたか」を銘柄ごとに見るためのテストです。</p>'
        html += '<dl class="explainer-dl">'
        html += '<dt>履歴年数</dt>'
        html += '<dd>その銘柄の上場 (もしくは取得可能データの開始) からの年数。新しい銘柄ほど短い。'
        html += '10年未満は「長期」と呼ぶには短いので、参考程度に。</dd>'
        html += '<dt>年率リターン</dt>'
        html += '<dd>1年あたり平均で何%値上がりしたか (配当除く)。<strong>S&amp;P500 の歴史的な年率リターンは約 +10%</strong>。'
        html += 'これより高ければ市場平均超え、低ければアンダーパフォーム。</dd>'
        html += '<dt>年率ボラ</dt>'
        html += '<dd>年率の値動きの標準偏差。<strong>15% なら穏やか、25% で平均的、40%超でハイリスク</strong>。'
        html += '株価の変動の激しさを表します。</dd>'
        html += '<dt>Sharpe (シャープレシオ)</dt>'
        html += '<dd><strong>「リスクの割にリターンが出ているか」</strong>を測る指標 = (年率リターン − 無リスク金利) ÷ 年率ボラ。'
        html += '<strong>1.0 以上で優秀 / 2.0 超で卓越 / マイナスはリスクに見合わないリターン</strong>。'
        html += 'この値が高いほど「効率的に儲かる銘柄」と言えます。</dd>'
        html += '<dt>最大DD (最大ドローダウン)</dt>'
        html += '<dd>過去の高値から最安値までの最大下落率。-50% なら「保有期間中に資産が半分になった瞬間があった」を意味します。'
        html += '<strong>長期保有する勇気が必要な深さ</strong>を表す心理的指標です。'
        html += 'NVDA や TSLA のような成長株は最大DDが -70% を超えることもあります。</dd>'
        html += '</dl>'
        html += '<p class="explainer-takeaway">💡 <strong>長期投資家の見方</strong>: '
        html += 'Sharpe が 1.0 以上 + 年率リターン > 10% + 最大DD < -60% 程度なら「歴史的に長期保有に向いた銘柄」と言えます。'
        html += 'ボラが高くても Sharpe が高ければ「振れ幅は大きいが、結局報われた」を意味します。</p>'
        html += '</div>'
        html += '<div class="port-table-section"><div class="port-table-wrap"><table class="port-table">'
        html += '<thead><tr><th>銘柄</th><th>履歴年数</th><th>年率リターン</th><th>年率ボラ</th><th>Sharpe</th><th>最大DD</th></tr></thead><tbody>'
        sorted_tickers = sorted(ticker_stats.items(), key=lambda x: -x[1].get("sharpe", 0))
        for t, s in sorted_tickers:
            ar = s.get("annual_return", 0) * 100
            av = s.get("annual_volatility", 0) * 100
            sh = s.get("sharpe", 0)
            dd = s.get("max_drawdown", 0) * 100
            sh_color = "var(--accent-green)" if sh > 1.0 else "var(--accent-yellow)" if sh > 0.5 else "var(--accent-red)"
            html += f'<tr><td><strong>{t}</strong></td><td>{s.get("history_years", 0):.1f}年</td>'
            html += f'<td style="color:{"var(--accent-green)" if ar > 0 else "var(--accent-red)"}">{ar:+.1f}%</td>'
            html += f'<td>{av:.1f}%</td>'
            html += f'<td style="color:{sh_color}">{sh:+.2f}</td>'
            html += f'<td style="color:var(--accent-red)">{dd:.1f}%</td></tr>'
        html += '</tbody></table></div></div>'

    html += '</div>'
    return html


def build_strategy_lab_html(results: dict) -> str:
    """🚀 Strategy Lab タブ — 学習加速 + 12項目強化群の状態."""
    enh = results.get("enhancements", {}) or {}
    if not enh:
        return ('<div class="section"><p style="color:var(--text-muted);text-align:center;padding:40px">'
                'Strategy Lab データがまだありません。次回 daily_evolution 実行後に表示されます。</p></div>')

    html = '<div class="section"><div class="section-title"><span class="icon">🚀</span> Strategy Lab — 学習加速 + 12項目強化</div>'

    # === Strategy Lab 解説 ===
    html += '<div class="explainer-box">'
    html += '<div class="explainer-title">🚀 Strategy Lab とは何か</div>'
    html += '<p>9ファクター予測 (Master Wisdom) <strong>の上に乗せる</strong>、'
    html += '機械学習・統計学・行動ファイナンスの<strong>追加レイヤー</strong>群です。'
    html += 'バフェット流の基本判断を、現代の数学的手法で<strong>補強・検証</strong>する役割を担います。</p>'
    html += '<div class="explainer-title" style="margin-top:14px;font-size:14px">主要な強化項目</div>'
    html += '<dl class="explainer-dl">'
    html += '<dt>📊 Walk-Forward 検証</dt>'
    html += '<dd>過去の予測を「実際にトレードしたら」と仮定して累積リターンを計算する手法。'
    html += '<strong>「本当に儲かるか」を統計的に裏付ける</strong>もので、シャープレシオ・年率リターン・勝率を算出。'
    html += '日々の予測の信頼度を数字で評価。</dd>'
    html += '<dt>⚠️ 異常検知 (Anomaly Detection)</dt>'
    html += '<dd>VIX、リターン、ボラティリティなど複数指標で <strong>「黒い白鳥イベント」</strong>の予兆を検出。'
    html += '深刻度 (severity) が 70% 超なら防御ポジション推奨。'
    html += '2008年・2020年のような異常局面を統計的に「平常時から逸脱」として捉える。</dd>'
    html += '<dt>🔥 セクターローテーション</dt>'
    html += '<dd>11業種別 ETF (XLK/XLF など) のリターン比較から、'
    html += '<strong>勢いのあるセクター</strong>と<strong>失速しているセクター</strong>を抽出。'
    html += '長期保有でも「業界の構造変化」は重要なシグナル。</dd>'
    html += '<dt>🛡️ ストレステスト</dt>'
    html += '<dd>「リーマンショック級の暴落が来たらポートフォリオは何%下落するか」を仮想シミュレーション。'
    html += '保有銘柄ごとに過去の暴落時の値動きを当てはめ、最悪ケースの想定損失額を算出。</dd>'
    html += '<dt>💰 税効率推奨 (Tax-Aware)</dt>'
    html += '<dd>NISA口座と特定口座 (税繰延) の使い分けを最適化。'
    html += '高配当・高成長銘柄を NISA に、低成長・配当少を特定に振り分ける推奨。'
    html += '長期保有では税繰延効果が複利で効くため重要。</dd>'
    html += '<dt>📊 実 P&amp;L フィードバック</dt>'
    html += '<dd>実際の含み損益を AI に「成績表」として渡し、'
    html += '当たった銘柄・外した銘柄から学習。「自分が何を間違えたか」を継続的に把握。</dd>'
    html += '</dl>'
    html += '<p class="explainer-takeaway">💡 <strong>長期保有との関係</strong>: '
    html += 'Strategy Lab の出力は「日々のトレード推奨」ではなく、<strong>「保有方針の妥当性確認」</strong>です。'
    html += '異常検知で警告が出ても全売却するのではなく、「新規買いの慎重化」「リバランスの慎重化」に使います。</p>'
    html += '<p class="explainer-source">📂 計算: <code>prediction_enhancements.py</code> + <code>portfolio_advisor_pro.py</code> / '
    html += '出力: <code>latest_evolution_results.enhancements</code> セクション</p>'
    html += '</div>'

    # ========== 1. Walk-Forward Sharpe ==========
    wf = enh.get("walk_forward", {}) or {}
    if wf and wf.get("n_trades", 0) > 0:
        sharpe = wf.get("sharpe", 0)
        ann_ret = wf.get("annual_return", 0) * 100
        win = wf.get("win_rate", 0) * 100
        dd = wf.get("max_dd", 0) * 100
        n_tr = wf.get("n_trades", 0)
        sharpe_color = "var(--accent-green)" if sharpe >= 1.0 else "var(--accent-yellow)" if sharpe >= 0.5 else "var(--accent-red)"
        html += '<div class="port-summary-grid">'
        html += f'<div class="port-metric"><div class="port-metric-label">Sharpe Ratio</div>'
        html += f'<div class="port-metric-value" style="color:{sharpe_color}">{sharpe:+.2f}</div>'
        html += f'<div class="port-metric-sub">{n_tr} trades</div></div>'
        html += f'<div class="port-metric"><div class="port-metric-label">年率リターン</div>'
        html += f'<div class="port-metric-value">{ann_ret:+.1f}%</div>'
        html += f'<div class="port-metric-sub">仮想実戦</div></div>'
        html += f'<div class="port-metric"><div class="port-metric-label">勝率</div>'
        html += f'<div class="port-metric-value">{win:.0f}%</div>'
        html += f'<div class="port-metric-sub">予測の的中率</div></div>'
        html += f'<div class="port-metric"><div class="port-metric-label">最大DD</div>'
        html += f'<div class="port-metric-value" style="color:var(--accent-red)">{dd:.1f}%</div>'
        html += f'<div class="port-metric-sub">仮想資産</div></div>'
        html += '</div>'

    # ========== 2. Meta-Learner ==========
    meta = enh.get("meta_learner", {}) or {}
    cal_size = enh.get("calibration_table_size", 0)
    if meta or cal_size:
        html += '<div class="sector-section"><div class="sector-title">🤖 メタ学習器 + 信頼度較正</div>'
        html += '<div class="sector-bars">'
        if meta.get("n_samples", 0) > 0:
            acc = meta.get("accuracy", 0) * 100
            html += f'<div class="sector-bar-row">'
            html += f'<div class="sector-bar-label">Stacking 精度</div>'
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{acc:.0f}%;background:var(--accent)"></div></div>'
            html += f'<div class="sector-bar-value">{acc:.1f}% ({meta.get("n_samples", 0)} 件)</div>'
            html += '</div>'
        if cal_size:
            html += f'<div class="sector-bar-row">'
            html += f'<div class="sector-bar-label">較正テーブル</div>'
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{min(100, cal_size*15):.0f}%;background:var(--color-success)"></div></div>'
            html += f'<div class="sector-bar-value">{cal_size} bins</div>'
            html += '</div>'
        html += '</div></div>'

    # ========== 3. Anomaly Detection ==========
    anomaly = enh.get("anomaly", {}) or {}
    if anomaly:
        is_anom = anomaly.get("is_anomaly", False)
        sev = anomaly.get("severity", 0) * 100
        triggers = anomaly.get("triggers", [])
        bg = "var(--accent-red-bg)" if is_anom else "var(--accent-green-bg)"
        border = "var(--accent-red)" if is_anom else "var(--accent-green)"
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">⚠️</span> 異常検知 (黒い白鳥)</div>'
        html += f'<div class="action-card" style="border-left:4px solid {border};background:{bg}">'
        if is_anom:
            html += f'<div class="card-header"><strong>🔴 異常検知 (severity {sev:.0f}%)</strong></div>'
            for t in triggers:
                html += f'<div class="reason" style="border-top:none;padding-top:0">• {html_mod.escape(t)}</div>'
        else:
            html += f'<div class="card-header"><strong>✅ マーケット正常</strong></div>'
            html += '<div class="reason" style="border-top:none;padding-top:0">VIX / 30日リターン / 直近5日 とも安定範囲</div>'
        html += '</div>'

    # ========== 4. Sector Rotation ==========
    sr = enh.get("sector_rotation", {}) or {}
    if sr.get("hot_sectors") or sr.get("cold_sectors"):
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">🔥</span> セクターローテーション</div>'
        html += '<div class="action-grid" style="grid-template-columns:1fr 1fr">'
        if sr.get("hot_sectors"):
            html += '<div class="action-card buy"><div class="card-header"><strong style="color:var(--accent-green)">🔥 Hot Sectors</strong></div>'
            html += f'<div class="reason" style="border-top:none;padding-top:0">{", ".join(sr["hot_sectors"])}</div></div>'
        if sr.get("cold_sectors"):
            html += '<div class="action-card sell"><div class="card-header"><strong style="color:var(--accent-red)">❄️ Cold Sectors</strong></div>'
            html += f'<div class="reason" style="border-top:none;padding-top:0">{", ".join(sr["cold_sectors"])}</div></div>'
        html += '</div>'

    # ========== 5. Stress Test ==========
    stress = enh.get("stress_test", {}) or {}
    scenarios = stress.get("scenarios", [])
    if scenarios:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">🛡️</span> ストレステスト</div>'
        html += '<div class="port-table-section"><div class="port-table-wrap"><table class="port-table">'
        html += '<thead><tr><th>シナリオ</th><th>想定損益</th><th>評価</th></tr></thead><tbody>'
        for sc in scenarios:
            pnl = sc.get("portfolio_pnl_pct", 0)
            pnl_color = "var(--accent-red)" if pnl < -5 else "var(--accent-yellow)" if pnl < 0 else "var(--accent-green)"
            html += f'<tr><td style="white-space:normal">{html_mod.escape(sc.get("scenario", ""))}</td>'
            html += f'<td style="color:{pnl_color}">{pnl:+.2f}%</td>'
            html += f'<td>{sc.get("rating", "")}</td></tr>'
        html += '</tbody></table></div></div>'

    # ========== 6. Tax Recommendations ==========
    tax_recs = enh.get("tax_recommendations", []) or []
    if tax_recs:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">💰</span> 税効率推奨 (NISA/特定口座)</div>'
        html += '<div class="action-grid">'
        for r in tax_recs:
            html += f'<div class="action-card">'
            html += f'<div class="card-header"><strong>{r.get("ticker", "")} [{r.get("account", "").upper()}]</strong></div>'
            html += f'<div class="holdings"><span>{r.get("shares", 0)}株</span><span>P&L ${r.get("pnl_usd", 0):+,.0f}</span></div>'
            html += f'<div class="reason">{html_mod.escape(r.get("tax_priority", ""))}</div></div>'
        html += '</div>'

    # ========== 7. Realized P&L Feedback ==========
    pnl_fb = enh.get("pnl_feedback", {}) or {}
    if pnl_fb.get("matched_predictions", 0) > 0:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">📊</span> 実P&L フィードバック</div>'
        html += '<div class="port-summary-grid">'
        html += f'<div class="port-metric"><div class="port-metric-label">マッチ件数</div>'
        html += f'<div class="port-metric-value">{pnl_fb.get("matched_predictions", 0)}</div>'
        html += f'<div class="port-metric-sub">実購入×予測の照合</div></div>'
        avg_r = pnl_fb.get("avg_realized_return_30d", 0) * 100
        html += f'<div class="port-metric"><div class="port-metric-label">平均30日リターン</div>'
        html += f'<div class="port-metric-value">{avg_r:+.1f}%</div>'
        html += f'<div class="port-metric-sub">実体験ベース</div></div>'
        win = pnl_fb.get("win_rate", 0) * 100
        html += f'<div class="port-metric"><div class="port-metric-label">勝率</div>'
        html += f'<div class="port-metric-value">{win:.0f}%</div>'
        html += f'<div class="port-metric-sub">実購入の的中率</div></div>'
        buy_acc = pnl_fb.get("buy_signal_accuracy", 0) * 100
        html += f'<div class="port-metric"><div class="port-metric-label">BUY 精度</div>'
        html += f'<div class="port-metric-value">{buy_acc:.0f}%</div>'
        html += f'<div class="port-metric-sub">予測BUY → 実際UP</div></div>'
        html += '</div>'

    html += '</div>'
    return html


def build_master_wisdom_html(results: dict) -> str:
    """🎯 Master Wisdom タブ — Buffett 級9ファクター予測.

    各銘柄ごとに 9ファクターのスコア + 重み + verdict を表示。
    上部に Master Learning サマリー + ファクター別精度。
    """
    master_signals = results.get("master_signals", []) or []
    master_learning = results.get("master_learning", {}) or {}

    factor_jp = {
        "quality_roe": "ROE品質",
        "quality_margin": "利益率品質",
        "value_earnings_yield": "Earnings Yield",
        "value_fcf_yield": "FCF Yield",
        "value_margin_of_safety": "Margin of Safety",
        "momentum_composite": "テクニカル",
        "contrarian_fear_greed": "Fear-Greed逆張り",
        "contrarian_insider_pulse": "インサイダー脈動",
        "risk_kelly": "Kelly基準",
    }

    if not master_signals and not master_learning:
        return ('<div class="section"><p style="color:var(--text-muted);text-align:center;padding:40px">'
                'Master Wisdom データはまだありません。次回 daily_evolution 実行後に表示されます。</p></div>')

    html = '<div class="section"><div class="section-title"><span class="icon">🎯</span> Master Wisdom — Buffett 級9ファクター予測</div>'

    # === Master Wisdom 解説 ===
    html += '<div class="explainer-box">'
    html += '<div class="explainer-title">🎯 Master Wisdom とは何か</div>'
    html += '<p><strong>Master Wisdom</strong> は、バフェット流の投資哲学を <strong>9つの定量指標 (ファクター)</strong> に分解し、'
    html += 'それぞれを 0〜100 のスコアで採点して合成する予測エンジンです。'
    html += '人間が「なんとなく良い銘柄」と感じる判断を、再現可能な数式に落としています。</p>'
    html += '<div class="explainer-title" style="margin-top:14px;font-size:14px">9つのファクター (3つの大カテゴリ)</div>'
    html += '<dl class="explainer-dl">'
    html += '<dt>① Quality (経営の質) — 2項目</dt>'
    html += '<dd><strong>ROE (株主資本利益率)</strong>: 「株主のお金を使ってどれだけ稼げているか」。15%以上が優良。'
    html += '<br><strong>営業利益率の質</strong>: 利益率の水準と安定性。高くて安定 = 経済的堀あり。</dd>'
    html += '<dt>② Value (株価の妥当性) — 3項目</dt>'
    html += '<dd><strong>Earnings Yield</strong>: 1÷PER。「株の利回り」を国債利回りと比較して割安度を測る。'
    html += '<br><strong>FCF Yield</strong>: 自由に使えるキャッシュ÷時価総額。5%以上で割安。'
    html += '<br><strong>Margin of Safety (安全域)</strong>: DCF (将来キャッシュフロー割引法) で算出した「本来価値」と現在株価の差。'
    html += '20%以上安く買えればバフェット流の合格ライン。</dd>'
    html += '<dt>③ Momentum + Contrarian + Risk — 4項目</dt>'
    html += '<dd><strong>テクニカル統合</strong>: トレンド・移動平均・出来高など複数のテクニカル指標を統合。'
    html += '<br><strong>Fear-Greed逆張り</strong>: VIX や Put/Call比率から市場の極端を検出。'
    html += '<br><strong>インサイダー脈動</strong>: 経営陣の自社株買い動向 (内部者は将来情報を持つ)。'
    html += '<br><strong>Kelly基準</strong>: 期待リターンとリスクから「適正な投資サイズ」を数学的に算出。</dd>'
    html += '</dl>'
    html += '<p class="explainer-takeaway">💡 <strong>9ファクターの合成方法</strong>: '
    html += '各ファクターを [-1, +1] のスコアに正規化し、'
    html += '<strong>市場局面 (低ボラ/移行期/危機) ごとに動的なウェイト</strong>を掛けて加重平均します。'
    html += '例えば危機局面では「安全域」と「インサイダー脈動」が重く、'
    html += '低ボラ局面では「テクニカル」と「Quality」が重くなります。</p>'
    html += '<p class="explainer-source">📂 計算: <code>master_predictor.py</code> の各 <code>*_score()</code> 関数。'
    html += 'ウェイト: <code>data/master_factor_weights.json</code> (Bayesian 学習で自動更新)。'
    html += '学習履歴: <code>data/master_learning_journal.json</code></p>'
    html += '</div>'

    # 学習サマリー
    n_total = master_learning.get("n_total_evaluated", 0)
    n_today = master_learning.get("n_evaluated", 0)
    factor_accs = master_learning.get("factor_accuracies", {}) or {}
    weights = master_learning.get("weights", {}) or {}

    if factor_accs:
        avg_acc = sum(factor_accs.values()) / len(factor_accs) if factor_accs else 0

        html += '<div class="port-summary-grid">'
        html += f'<div class="port-metric"><div class="port-metric-label">平均ファクター精度</div>'
        html += f'<div class="port-metric-value">{avg_acc*100:.0f}%</div>'
        html += f'<div class="port-metric-sub">9ファクター平均</div></div>'

        html += f'<div class="port-metric"><div class="port-metric-label">累計評価</div>'
        html += f'<div class="port-metric-value">{n_total}</div>'
        html += f'<div class="port-metric-sub">本日 +{n_today} 新規</div></div>'

        n_signals = len(master_signals)
        n_strong = sum(1 for s in master_signals if "STRONG" in s.get("signal", ""))
        html += f'<div class="port-metric"><div class="port-metric-label">予測銘柄</div>'
        html += f'<div class="port-metric-value">{n_signals}</div>'
        html += f'<div class="port-metric-sub">うちSTRONG: {n_strong}</div></div>'

        regime_value = master_signals[0].get("regime", "?") if master_signals else "?"
        regime_jp = {"low_vol": "低ボラ", "transition": "移行期", "crisis": "危機"}.get(regime_value, regime_value)
        html += f'<div class="port-metric"><div class="port-metric-label">レジーム</div>'
        html += f'<div class="port-metric-value" style="font-size:20px">{regime_jp}</div>'
        html += f'<div class="port-metric-sub">適応的ウェイト</div></div>'
        html += '</div>'

    # ファクター別精度 + 重み バー
    if factor_accs:
        html += '<div class="explainer-box explainer-sub" style="margin-top:20px">'
        html += '<div class="explainer-title">⚙️ 「学習状態」の読み方</div>'
        html += '<p>各ファクターの右側に表示される指標:</p>'
        html += '<ul class="explainer-list">'
        html += '<li><strong>精度 X%</strong>: 過去の予測の方向 (上がる/下がる) が実際の値動きと一致した割合。'
        html += '50%が偶然、60%超で「市場で本当に効いている」、70%超で「卓越」。</li>'
        html += '<li><strong>W X.X%</strong>: そのファクターに与えられたウェイト (重み)。'
        html += '合計100%。精度が高いファクターほど重みが大きくなり、外れ続けるとウェイトが下がる。</li>'
        html += '<li><strong>decay X.XX (赤表示)</strong>: 直近の予測が外れているため、ウェイトが減衰中。'
        html += '0.5 だと「半分まで縮小」、0.99 以上は通常状態。</li>'
        html += '</ul>'
        html += '<p class="explainer-takeaway">💡 これが <strong>「自己進化」の正体</strong>です。'
        html += '人間がパラメータを調整するのではなく、<strong>当たれば信頼度を上げ、外れれば信頼度を下げる</strong>'
        html += 'という Bayesian 更新を毎日繰り返しています。</p>'
        html += '</div>'
        html += '<div class="sector-section"><div class="sector-title">⚙️ 9ファクターの学習状態</div>'
        html += '<div class="sector-bars">'
        for name in factor_accs.keys():
            jp = factor_jp.get(name, name)
            acc = factor_accs.get(name, 0) * 100
            w = weights.get(name, 0)
            decay = master_learning.get("factor_decay", {}).get(name, 1.0)
            decay_str = "" if decay >= 0.99 else f' <span style="color:var(--accent-red)">decay {decay:.2f}</span>'
            html += f'<div class="sector-bar-row">'
            html += f'<div class="sector-bar-label">{jp}</div>'
            html += f'<div class="sector-bar-track"><div class="sector-bar-fill" style="width:{min(100,w*400):.0f}%;background:var(--accent)"></div></div>'
            html += f'<div class="sector-bar-value">精度 {acc:.0f}% / W {w:.2%}{decay_str}</div>'
            html += '</div>'
        html += '</div></div>'

    # 本日の発見
    findings = master_learning.get("notable_findings", [])
    if findings:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">💡</span> 本日の発見</div>'
        html += '<div class="action-grid" style="grid-template-columns:1fr">'
        for f in findings[:8]:
            html += f'<div class="action-card" style="border-left:3px solid var(--accent)">'
            html += f'<div class="reason" style="border-top:none;padding-top:0">{html_mod.escape(f)}</div>'
            html += '</div>'
        html += '</div>'

    # 個別予測カード
    if master_signals:
        html += '<div class="section-title" style="margin-top:20px"><span class="icon">📊</span> 個別銘柄スコア</div>'
        html += '<div class="adv-grid">'
        for s in master_signals[:18]:
            sig = s.get("signal", "HOLD")
            color, emoji = {
                "STRONG_BUY":  ("var(--accent-green)", "🟢🟢"),
                "BUY":         ("var(--accent-green)", "🟢"),
                "HOLD":        ("var(--accent-yellow)", "🟡"),
                "SELL":        ("var(--accent-red)", "🔴"),
                "STRONG_SELL": ("var(--accent-red)", "🔴🔴"),
            }.get(sig, ("var(--text-muted)", "⚪"))

            ticker = s.get("ticker", "?")
            score = s.get("composite_score", 0)
            conf = s.get("confidence", 0)
            kelly_pct = s.get("kelly_recommended_pct", 0) * 100
            mos = s.get("margin_of_safety", 0)
            url = yahoo_chart_url(ticker)

            html += f'<div class="adv-card" style="border-left:4px solid {color}">'
            html += f'<div class="adv-header"><a href="{url}" target="_blank" class="adv-ticker">{ticker} 📈</a>'
            html += f'<span class="adv-signal" style="color:{color}">{emoji} {sig}</span></div>'
            html += f'<div class="adv-score">合成: <strong>{score:+.3f}</strong> / 確信度 {conf:.0%} / Kelly {kelly_pct:+.1f}%</div>'
            if mos:
                mos_color = "var(--accent-green)" if mos > 0 else "var(--accent-red)"
                html += f'<div class="adv-conf" style="color:{mos_color}">Margin of Safety: {mos*100:+.0f}%</div>'

            # ファクター別ミニバー
            html += '<div class="adv-subbars">'
            fs = s.get("factor_scores", {})
            for name in factor_accs.keys() if factor_accs else fs.keys():
                f_data = fs.get(name, {})
                if not f_data:
                    continue
                f_score = f_data.get("score", 0)
                v_pct = abs(f_score) * 50
                f_color = "var(--accent-green)" if f_score > 0 else "var(--accent-red)" if f_score < 0 else "var(--text-muted)"
                jp = factor_jp.get(name, name)[:8]
                html += f'<div class="adv-subbar">'
                html += f'<span class="adv-sublabel">{jp}</span>'
                html += f'<span class="adv-subval" style="color:{f_color}">{f_score:+.2f}</span>'
                if f_score >= 0:
                    html += f'<div class="adv-bar-track"><div class="adv-bar-pos" style="width:{v_pct:.0f}%"></div></div>'
                else:
                    html += f'<div class="adv-bar-track"><div class="adv-bar-neg" style="width:{v_pct:.0f}%"></div></div>'
                html += '</div>'
            html += '</div></div>'
        html += '</div>'

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

    # === Advanced AI 解説 ===
    html += '<div class="explainer-box">'
    html += '<div class="explainer-title">🔮 アンサンブル予測とは何か</div>'
    html += '<p>1つの予測モデルだけに頼ると、そのモデルの<strong>得意/不得意</strong>に成績が左右されます。'
    html += '「複数のモデルの予測を統合する」ことで <strong>「個別モデルの弱点を平均化し、当たりやすくする」</strong>のが'
    html += 'アンサンブル (Ensemble) の発想です。</p>'
    html += '<div class="explainer-title" style="margin-top:14px;font-size:14px">統合される5つのモデル</div>'
    html += '<dl class="explainer-dl">'
    html += '<dt>① Kalman Filter (カルマンフィルタ)</dt>'
    html += '<dd>株価のノイズを除去して<strong>「真のトレンド」</strong>を抽出する状態空間モデル。'
    html += '宇宙工学やGPSにも使われる名門アルゴリズム。'
    html += '日々の細かい値動きに惑わされず大きな流れを掴む。</dd>'
    html += '<dt>② Hurst指数</dt>'
    html += '<dd>株価時系列の<strong>「持続性 vs 平均回帰性」</strong>を測る指標。'
    html += '<strong>0.5超でトレンド継続 (順張り有利)</strong>、'
    html += '<strong>0.5未満で平均回帰 (逆張り有利)</strong>。'
    html += '今の銘柄が「上がり続けるタイプ」か「振動するタイプ」かを判定。</dd>'
    html += '<dt>③ Cross-Sectional Momentum</dt>'
    html += '<dd>同時点で複数銘柄を比較し、<strong>相対的に強い銘柄を買う</strong>戦略。'
    html += '「絶対的に上がっているか」ではなく「他より上がっているか」で判断。'
    html += '学術研究で長期に渡って効果が確認されている古典的アノマリー。</dd>'
    html += '<dt>④ Mean Reversion (平均回帰)</dt>'
    html += '<dd>「価格は長期平均に戻る」という統計的傾向を利用。'
    html += '直近で<strong>下がりすぎた銘柄</strong>に小さく賭けるシグナル。'
    html += '短期向けで、長期保有方針では「買いタイミング」の参考に使う。</dd>'
    html += '<dt>⑤ Stacking Meta-Learner</dt>'
    html += '<dd>上記4モデルの予測を入力として受け取り、'
    html += '<strong>「どのモデルを今信じるべきか」</strong>を機械学習で学習する高位モデル。'
    html += '局面ごとに最適なモデル組み合わせを動的に選択。</dd>'
    html += '</dl>'
    html += '<p class="explainer-takeaway">💡 <strong>9ファクター (Master Wisdom) との違い</strong>: '
    html += '9ファクターは <strong>「企業の本質的な価値」</strong>を見る (財務指標中心)、'
    html += 'アンサンブルは <strong>「価格時系列の数学的パターン」</strong>を見る (テクニカル中心)。'
    html += '両方を組み合わせて最終判断を出します。</p>'
    html += '<p class="explainer-source">📂 計算: <code>prediction_enhancements.py</code> の <code>train_stacking_meta()</code> など / '
    html += '出力: <code>latest_evolution_results.advanced_signals</code></p>'
    html += '</div>'
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
    daily_brief_html = build_daily_brief_html(results, track, holdings)
    weekly_brief_html = build_weekly_brief_html(results, track, holdings)
    monthly_brief_html = build_monthly_brief_html(results, track, holdings)
    portfolio_html = build_portfolio_html(holdings, results)
    advanced_html = build_advanced_signals_html(results)
    learning_html = build_learning_html(results)
    master_wisdom_html = build_master_wisdom_html(results)
    strategy_lab_html = build_strategy_lab_html(results)
    history_html = build_history_html(results)
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
  /* === Notion warm-neutral palette === */
  --bg-primary:    #ffffff;    /* Pure white canvas */
  --bg-panel:      #f6f5f4;    /* Warm white surface (yellow undertone) */
  --bg-elevated:   #ffffff;    /* Cards stay pure white */
  --bg-secondary:  #efeeec;    /* Subtle emphasis */
  --bg-card:       #ffffff;
  --bg-card-hover: #faf9f8;
  --bg-button:     #ffffff;
  --bg-button-hover: #f6f5f4;

  /* === Notion text hierarchy (warm near-black) === */
  --text-primary:   rgba(0, 0, 0, 0.95);    /* Near-black (micro warmth) */
  --text-secondary: #31302e;                /* Warm dark */
  --text-muted:     #615d59;                /* Warm gray 500 */
  --text-subtle:    #a39e98;                /* Warm gray 300 */

  /* === Notion whisper borders === */
  --border:         rgba(0, 0, 0, 0.1);     /* Standard whisper border */
  --border-subtle:  rgba(0, 0, 0, 0.06);    /* Ultra subtle */
  --border-solid:   #31302e;                /* Solid for emphasis */
  --line-tint:      rgba(0, 0, 0, 0.04);

  /* === Brand: Notion Blue (singular accent) === */
  --accent:         #0075de;    /* Notion Blue (primary CTA) */
  --accent-bright:  #097fe8;    /* Focus / hover accent */
  --accent-hover:   #005bab;    /* Button pressed */
  --accent-navy:    #213183;    /* Deep brand navy */
  --accent-soft:    #f2f9ff;    /* Badge blue bg */
  --accent-soft-hover: #e6f1fc;
  --accent-text:    #097fe8;    /* Badge blue text */

  /* === Notion semantic accents === */
  --color-success:  #1aae39;    /* Green (confirmation) */
  --color-success-alt: #2a9d99; /* Teal (positive indicator) */
  --color-error:    #dd5b00;    /* Orange (warning) */
  --color-warn:     #dd5b00;
  --color-pink:     #ff64c8;
  --color-purple:   #391c57;
  --color-brown:    #523410;

  /* === P&L / chart semantic (Notion palette) === */
  --accent-green:   #1aae39;
  --accent-green-bg:rgba(26, 174, 57, 0.08);
  --accent-red:     #dd5b00;
  --accent-red-bg:  rgba(221, 91, 0, 0.08);
  --accent-yellow:  #c08532;
  --accent-yellow-bg:rgba(192, 133, 50, 0.08);
  --accent-blue:    #0075de;
  --accent-blue-bg: #f2f9ff;
  --accent-purple:  #391c57;
  --accent-purple-bg:rgba(57, 28, 87, 0.06);

  /* === Radius (Notion: subtle 4px) === */
  --radius-sm:     2px;
  --radius-md:     4px;          /* Standard Notion radius */
  --radius-lg:     6px;
  --radius-xl:     8px;
  --radius-panel:  10px;
  --radius-pill:   9999px;
  --radius-circle: 50%;

  /* === Spacing (8px base) === */
  --space-half: 4px;
  --space-1: 8px;  --space-2: 12px; --space-3: 16px;
  --space-4: 24px; --space-5: 32px; --space-6: 48px;
  --space-7: 64px; --space-8: 96px;

  /* === Notion Shadow: multi-layer sub-0.05 opacity === */
  --shadow-whisper: rgba(0,0,0,0.04) 0px 4px 18px, rgba(0,0,0,0.027) 0px 2.025px 7.84688px, rgba(0,0,0,0.02) 0px 0.8px 2.925px, rgba(0,0,0,0.01) 0px 0.175px 1.04062px;
  --shadow-deep:    rgba(0,0,0,0.01) 0px 1px 3px, rgba(0,0,0,0.02) 0px 3px 7px, rgba(0,0,0,0.02) 0px 7px 15px, rgba(0,0,0,0.04) 0px 14px 28px, rgba(0,0,0,0.05) 0px 23px 52px;
  --shadow-focus:   rgba(9, 127, 232, 0.25) 0px 0px 0px 3px;

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
  /* NotionInter → Inter + 日本語 fallback 必須 */
  font-family:'Inter','Noto Sans JP','Hiragino Sans','Yu Gothic UI',-apple-system,system-ui,'Segoe UI',sans-serif;
  background:var(--bg-primary);
  color:var(--text-primary);
  min-height:100vh;
  line-height:1.5;                  /* Notion Body line-height */
  font-weight:400;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}}
/* Notion は display/heading に lnum + locl を適用 */
h1, h2, h3, .section-title, .header h1, .stat-value, .port-metric-value{{
  font-feature-settings:'lnum','locl';
}}
/* Notion weight system: 400 body / 500 UI / 600 emphasis / 700 display */
.tab-btn,
.stat-label,
.port-metric-label,
.port-table th,
.adv-sublabel{{
  font-weight:500;
}}
.acct-badge,
.action-badge,
.adv-signal,
.pick-sector{{
  font-weight:600;
  letter-spacing:0.01em;    /* Notion badge micro-tracking */
}}
.container{{max-width:1440px;margin:0 auto;padding:var(--space-5)}}

/* Focus-visible: Notion blue ring */
*:focus{{outline:none}}
*:focus-visible{{
  outline:none;
  box-shadow:var(--shadow-focus);
  border-radius:var(--radius-md);
}}
button:focus-visible,
a:focus-visible{{
  outline:none;
  box-shadow:var(--shadow-focus);
}}
/* Selection color */
::selection{{background:var(--accent-soft);color:var(--text-primary)}}

/* Scrollbar (webkit) — thin Linear style */
::-webkit-scrollbar{{width:10px;height:10px}}
::-webkit-scrollbar-track{{background:var(--bg-panel)}}
::-webkit-scrollbar-thumb{{background:rgba(0,0,0,0.15);border-radius:var(--radius-pill);border:2px solid var(--bg-panel)}}
::-webkit-scrollbar-thumb:hover{{background:rgba(0,0,0,0.25)}}

/* Header — Notion minimal white */
.header{{
  display:flex;justify-content:space-between;align-items:center;
  padding:var(--space-4) var(--space-5);
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  margin-bottom:var(--space-4);
  box-shadow:var(--shadow-whisper);
}}
.header h1{{
  font-family:inherit;
  font-size:32px;
  font-weight:700;                 /* Notion display weight */
  letter-spacing:-0.028em;          /* Progressive: ~-0.9px at 32px */
  color:var(--text-primary);
  line-height:1.1;
}}
.header h1 span{{
  color:var(--accent);
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
  border:1px solid rgba(0, 117, 222, 0.15);
  border-radius:var(--radius-pill);
  padding:3px 10px;
  font-size:12px;font-weight:600;
  color:var(--accent-text);
  margin-top:var(--space-1);
  letter-spacing:0.01em;
}}
.header-meta .confidence .dot{{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 2.4s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}

/* Stats — Notion white cards w/ whisper shadow */
.stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-2);margin-bottom:var(--space-4)}}
.stat-card{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3);
  transition:box-shadow var(--transition-shadow);
}}
.stat-card:hover{{box-shadow:var(--shadow-whisper)}}
.stat-label{{
  font-size:11px;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:0.06em;font-weight:500;
  font-family:inherit;
}}
.stat-value{{
  font-size:28px;font-weight:700;margin-top:var(--space-1);
  font-family:inherit;
  color:var(--text-primary);
  letter-spacing:-0.025em;
  font-variant-numeric:tabular-nums lnum;
}}
.stat-sub{{font-size:13px;color:var(--text-muted);margin-top:var(--space-half);line-height:1.5}}

/* Tabs — Notion pill segmented */
.tab-nav{{
  display:flex;gap:2px;
  margin-bottom:var(--space-4);
  overflow-x:auto;
  padding:var(--space-half);
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
}}
.tab-btn{{
  padding:6px 14px;background:transparent;border:none;
  border-radius:var(--radius-md);
  color:var(--text-muted);cursor:pointer;
  font-size:14px;font-weight:500;white-space:nowrap;
  font-family:inherit;
  transition:color var(--transition-color), background var(--transition-bg);
}}
.tab-btn:hover{{background:var(--bg-button-hover);color:var(--text-primary)}}
.tab-btn.active{{background:var(--accent);color:#fff;font-weight:600}}
.tab-btn.active:hover{{color:#fff;background:var(--accent-hover)}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}

/* Strategy Banner — Notion warm white */
/* === Trading Hero — ポートフォリオ総括 (Robinhood/Yahoo風) === */
.trading-hero{{
  background:linear-gradient(135deg,var(--bg-card) 0%,var(--bg-panel) 100%);
  border-radius:var(--radius-lg);
  padding:20px 24px;
  margin-bottom:var(--space-3);
  border:1px solid var(--border);
  border-left:4px solid var(--text-muted);
}}
.trading-hero.hero-profit{{
  border-left-color:var(--accent-green);
  background:linear-gradient(135deg,rgba(26,174,57,0.08) 0%,var(--bg-card) 60%);
}}
.trading-hero.hero-loss{{
  border-left-color:var(--accent-red);
  background:linear-gradient(135deg,rgba(221,91,0,0.08) 0%,var(--bg-card) 60%);
}}
.trading-hero-label{{
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:0.08em;
  color:var(--text-muted);
  font-weight:600;
  margin-bottom:6px;
}}
.trading-hero-row{{
  display:flex;align-items:baseline;gap:10px;
  font-family:'JetBrains Mono',monospace;
  letter-spacing:-0.02em;
}}
.trading-hero-arrow{{
  font-size:24px;
  font-weight:700;
  line-height:1;
}}
.trading-hero.hero-profit .trading-hero-arrow,
.trading-hero.hero-profit .trading-hero-value,
.trading-hero.hero-profit .trading-hero-pct{{color:var(--accent-green)}}
.trading-hero.hero-loss .trading-hero-arrow,
.trading-hero.hero-loss .trading-hero-value,
.trading-hero.hero-loss .trading-hero-pct{{color:var(--accent-red)}}
.trading-hero-value{{
  font-size:36px;
  font-weight:700;
  line-height:1;
}}
.trading-hero-pct{{
  font-size:18px;
  font-weight:600;
  padding:4px 10px;
  border-radius:var(--radius-pill);
  background:rgba(255,255,255,0.04);
}}
.trading-hero.hero-profit .trading-hero-pct{{background:rgba(26,174,57,0.12)}}
.trading-hero.hero-loss .trading-hero-pct{{background:rgba(221,91,0,0.12)}}
.trading-hero-sub{{
  margin-top:10px;
  font-size:13px;
  color:var(--text-secondary);
  display:flex;flex-wrap:wrap;gap:8px;
  align-items:center;
}}
.trading-hero-sub strong{{color:var(--text-primary);font-family:'JetBrains Mono',monospace}}
.trading-hero-sub .dot-sep{{color:var(--text-subtle)}}

.strategy-banner{{
  background:var(--bg-panel);
  border:1px solid var(--border);
  border-left:3px solid var(--accent);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-4);
  font-size:15px;
  color:var(--text-secondary);
  line-height:1.6;
}}
.strategy-banner strong{{color:var(--text-primary);font-weight:600}}

/* Sections & Cards */
.section{{margin-bottom:var(--space-5)}}

/* === Explainer (読み物・解説) === */
.explainer-box{{
  background:linear-gradient(135deg,var(--accent-soft) 0%,var(--bg-panel) 100%);
  border:1px solid var(--border);
  border-left:4px solid var(--accent);
  border-radius:var(--radius-md);
  padding:18px 22px;
  margin:var(--space-3) 0;
  font-size:14px;
  line-height:1.85;
  color:var(--text-secondary);
}}
.explainer-box.explainer-sub{{
  border-left-color:var(--accent-bright);
  padding:14px 18px;
  margin:var(--space-2) 0 var(--space-3) 0;
  font-size:13.5px;
}}
.explainer-title{{
  font-size:15px;
  font-weight:700;
  color:var(--text-primary);
  margin-bottom:8px;
}}
.explainer-box p{{margin:8px 0;color:var(--text-secondary)}}
.explainer-box strong{{color:var(--text-primary);font-weight:700}}
.explainer-box code{{
  font-family:'JetBrains Mono',monospace;
  font-size:12px;
  background:var(--bg-panel);
  padding:1px 6px;
  border-radius:4px;
  color:var(--accent);
}}
.explainer-quote{{
  margin:12px 0;
  padding:10px 16px;
  border-left:3px solid var(--accent);
  background:var(--bg-card);
  font-style:italic;
  color:var(--text-primary);
  font-size:14.5px;
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;
}}
.explainer-list{{
  margin:8px 0;
  padding-left:22px;
  line-height:1.85;
}}
.explainer-list li{{margin-bottom:6px;color:var(--text-secondary)}}
.explainer-dl{{
  margin:10px 0;
  padding:10px 0;
  border-top:1px dashed var(--border);
  border-bottom:1px dashed var(--border);
}}
.explainer-dl dt{{
  font-weight:700;
  color:var(--accent);
  margin-top:10px;
  font-family:'JetBrains Mono',monospace;
  font-size:13px;
  letter-spacing:-0.01em;
}}
.explainer-dl dd{{
  margin:3px 0 0 16px;
  padding-left:12px;
  border-left:2px solid var(--border);
  color:var(--text-secondary);
  line-height:1.7;
}}
.explainer-takeaway{{
  margin-top:12px !important;
  padding:10px 14px;
  background:rgba(26,174,57,0.08);
  border-left:3px solid var(--accent-green);
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;
  font-size:13.5px;
}}
.explainer-source{{
  margin-top:12px !important;
  padding-top:8px;
  border-top:1px solid var(--border);
  font-size:12px;
  color:var(--text-muted);
  font-style:italic;
}}
.explainer-text{{
  margin:8px 0 12px 0;
  font-size:13.5px;
  color:var(--text-secondary);
  line-height:1.8;
}}
.explainer-text strong{{color:var(--text-primary);font-weight:700}}
.section-title{{
  font-family:inherit;
  font-size:22px;font-weight:700;
  margin-bottom:var(--space-3);
  display:flex;align-items:center;gap:var(--space-1);
  color:var(--text-primary);
  letter-spacing:-0.011em;
  line-height:1.27;
}}
.section-title .icon{{font-size:22px}}

.action-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:var(--space-2)}}
.action-card{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  display:flex;flex-direction:column;gap:var(--space-1);
  transition:box-shadow var(--transition-shadow);
}}
.action-card:hover{{box-shadow:var(--shadow-whisper)}}
.action-card.buy{{border-left:3px solid var(--accent-green)}}
.action-card.hold{{border-left:3px solid var(--accent-yellow)}}
.action-card.sell{{border-left:3px solid var(--accent-red)}}
.action-card .card-header{{display:flex;justify-content:space-between;align-items:center}}
.action-card .ticker{{
  font-size:22px;font-weight:700;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-primary);
  letter-spacing:-0.012em;
}}
.action-badge{{
  font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.04em;
  padding:2px 10px;
  border-radius:var(--radius-pill);
  font-family:inherit;
}}
.action-badge.buy{{background:var(--accent-green-bg);color:var(--accent-green);border:1px solid rgba(26, 174, 57, 0.2)}}
.action-badge.hold{{background:var(--accent-yellow-bg);color:var(--accent-yellow);border:1px solid rgba(192, 133, 50, 0.2)}}
.action-badge.sell{{background:var(--accent-red-bg);color:var(--accent-red);border:1px solid rgba(221, 91, 0, 0.2)}}
.holdings{{
  display:flex;gap:var(--space-3);
  font-size:12px;color:var(--text-muted);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:-0.01em;
}}
.pnl-positive{{color:var(--accent-green)}}
.pnl-negative{{color:var(--accent-red)}}

/* === Trading-App 風 保有情報ブロック === */
.holdings-pro{{
  background:var(--bg-panel);
  border-radius:var(--radius-sm);
  padding:8px 10px;
  margin:6px 0;
  font-family:'JetBrains Mono',monospace;
  border-left:3px solid var(--text-muted);
}}
.holdings-pro.pnl-positive{{
  border-left-color:var(--accent-green);
  background:linear-gradient(90deg,rgba(26,174,57,0.08) 0%,var(--bg-panel) 60%);
}}
.holdings-pro.pnl-negative{{
  border-left-color:var(--accent-red);
  background:linear-gradient(90deg,rgba(221,91,0,0.08) 0%,var(--bg-panel) 60%);
}}
.holdings-row{{
  display:flex;justify-content:space-between;
  font-size:11px;color:var(--text-muted);
  margin-bottom:3px;
}}
.holdings-row .hold-label{{color:var(--text-subtle);font-weight:500}}
.holdings-row .hold-value{{color:var(--text-primary);font-weight:600}}
.holdings-pnl{{
  display:flex;align-items:baseline;gap:6px;
  margin-top:6px;padding-top:6px;
  border-top:1px solid var(--border);
}}
.holdings-pro.pnl-positive .holdings-pnl{{color:var(--accent-green)}}
.holdings-pro.pnl-negative .holdings-pnl{{color:var(--accent-red)}}
.pnl-arrow{{
  font-size:14px;
  font-weight:700;
  display:inline-block;
  line-height:1;
}}
.pnl-amount{{
  font-size:16px;
  font-weight:700;
  letter-spacing:-0.02em;
}}
.pnl-pct{{
  font-size:12px;
  font-weight:600;
  opacity:0.85;
}}
.reason{{
  font-size:13px;color:var(--text-secondary);
  line-height:1.6;
  border-top:1px solid var(--border);
  padding-top:var(--space-1);
  word-break:normal;overflow-wrap:anywhere;
}}
.alloc-bar{{height:3px;border-radius:var(--radius-pill);background:var(--bg-panel);margin-top:auto;overflow:hidden}}
.alloc-bar-fill{{height:100%;border-radius:var(--radius-pill);background:var(--accent)}}
.alloc-label{{
  font-size:11px;color:var(--text-subtle);
  margin-top:var(--space-half);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:0.02em;
}}

.picks-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:var(--space-2)}}
.pick-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  border-left:3px solid var(--accent);
  transition:box-shadow var(--transition-shadow);
}}
.pick-card:hover{{box-shadow:var(--shadow-whisper)}}
.pick-ticker{{font-size:18px;font-weight:590;font-family:'JetBrains Mono',monospace;color:var(--text-primary);letter-spacing:-0.013em}}
.pick-sector{{
  display:inline-block;margin-top:var(--space-half);
  font-size:10px;color:var(--accent);
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
  border-radius:var(--radius-md);
  padding:var(--space-4) var(--space-3);
  text-align:center;
  transition:box-shadow var(--transition-shadow);
}}
.risk-card:hover{{box-shadow:var(--shadow-whisper)}}
.risk-card.bull{{border-top:3px solid var(--accent-green)}}
.risk-card.base{{border-top:3px solid var(--accent)}}
.risk-card.bear{{border-top:3px solid var(--accent-red)}}
.risk-prob{{
  font-size:40px;font-weight:700;
  font-family:inherit;
  letter-spacing:-0.028em;
  font-variant-numeric:tabular-nums lnum;
}}
.risk-card.bull .risk-prob{{color:var(--accent-green)}}
.risk-card.base .risk-prob{{color:var(--accent)}}
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
  font-weight:510;color:var(--accent);
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
.footer a{{color:var(--accent);text-decoration:none;transition:color var(--transition-color)}}
.footer a:hover{{color:var(--accent-hover);text-decoration:underline}}
.fade-in{{animation:fadeIn 0.4s ease forwards;opacity:0}}
@keyframes fadeIn{{to{{opacity:1}}}}

/* Portfolio Tab — Linear dense dashboard */
.port-summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-2);margin-bottom:var(--space-4)}}
.port-metric{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3);
  transition:box-shadow var(--transition-shadow);
}}
.port-metric:hover{{box-shadow:var(--shadow-whisper)}}
.port-metric-label{{
  font-size:11px;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:0.06em;font-weight:500;
  font-family:inherit;
}}
.port-metric-value{{
  font-size:26px;font-weight:700;
  font-family:inherit;
  margin-top:var(--space-1);
  color:var(--text-primary);
  letter-spacing:-0.025em;
  font-variant-numeric:tabular-nums lnum;
}}
.port-metric-sub{{font-size:12px;color:var(--text-muted);margin-top:var(--space-half);line-height:1.5;font-variant-numeric:tabular-nums}}

.sector-section{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);
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
  height:8px;background:var(--bg-panel);
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
  border-radius:var(--radius-md);
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
  font-weight:600;text-transform:uppercase;
  font-size:11px;letter-spacing:0.04em;
  cursor:pointer;user-select:none;
  background:var(--bg-panel);
  font-family:inherit;
  white-space:nowrap;
  transition:color var(--transition-color);
}}
.port-table th:hover{{color:var(--accent)}}
.port-table td{{
  padding:var(--space-2) var(--space-2);
  border-bottom:1px solid var(--border-subtle);
  color:var(--text-secondary);
  white-space:nowrap;
  font-family:'JetBrains Mono',monospace;
  letter-spacing:-0.005em;
  font-variant-numeric:tabular-nums;
}}
.port-table tr:hover{{background:var(--bg-card-hover)}}
.port-table tr.row-profit td.pnl-cell{{background:rgba(26,174,57,0.06)}}
.port-table tr.row-loss td.pnl-cell{{background:rgba(221,91,0,0.06)}}
.port-table .pnl-cell{{font-weight:700;font-size:13px}}
.port-table .pnl-arrow-sm{{display:inline-block;margin-right:3px;font-size:10px;line-height:1}}
.port-table .port-ticker strong{{color:var(--text-primary);font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700}}
.acct-badge{{
  display:inline-block;padding:2px 10px;
  border-radius:var(--radius-pill);
  font-size:11px;
  background:var(--accent-soft);
  color:var(--accent-text);
  font-weight:600;
  font-family:inherit;
  letter-spacing:0.02em;
  text-transform:uppercase;
}}
.chart-link{{text-decoration:none;font-size:14px;opacity:0.55;transition:opacity 0.2s}}
.chart-link:hover{{opacity:1}}

/* Advanced Predictor Tab — Linear data cards */
.adv-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:var(--space-2)}}
.adv-card{{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  transition:box-shadow var(--transition-shadow);
}}
.adv-card:hover{{box-shadow:var(--shadow-whisper)}}
.adv-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--space-1)}}
.adv-ticker{{
  font-size:20px;font-weight:700;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-primary);
  text-decoration:none;
  letter-spacing:-0.012em;
  transition:color var(--transition-color);
}}
.adv-ticker:hover{{color:var(--accent)}}
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
  height:6px;background:var(--bg-panel);
  border-radius:var(--radius-pill);overflow:hidden;
}}
.adv-bar-pos{{height:100%;background:var(--accent-green);border-radius:var(--radius-pill)}}
.adv-bar-neg{{height:100%;background:var(--accent-red);border-radius:var(--radius-pill)}}

/* === Daily Brief 記事形式スタイル === */
.daily-brief{{
  max-width:760px;
  margin:0 auto;
  padding:var(--space-4) 0;
  font-size:16px;
  line-height:1.8;
  color:var(--text-primary);
}}
.db-header{{
  border-bottom:1px solid var(--border);
  padding-bottom:var(--space-4);
  margin-bottom:var(--space-5);
}}
.db-eyebrow{{
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:0.12em;
  color:var(--accent);
  font-weight:600;
  margin-bottom:var(--space-2);
}}
.db-title{{
  font-size:34px;
  font-weight:700;
  letter-spacing:-0.025em;
  line-height:1.25;
  color:var(--text-primary);
  margin-bottom:var(--space-3);
  word-break:auto-phrase;
}}
.db-meta{{
  display:flex;flex-wrap:wrap;gap:var(--space-3);
  font-size:13px;color:var(--text-muted);
  font-family:'JetBrains Mono',monospace;
}}
.db-meta-item{{display:inline-flex;align-items:center;gap:var(--space-half)}}
.db-section{{
  margin-bottom:var(--space-5);
  padding-bottom:var(--space-4);
  border-bottom:1px solid var(--border-subtle);
}}
.db-section:last-of-type{{border-bottom:none}}
.db-section-title{{
  font-size:22px;font-weight:700;
  letter-spacing:-0.015em;
  color:var(--text-primary);
  margin-bottom:var(--space-3);
  line-height:1.3;
}}
.db-lead{{
  font-size:18px;
  line-height:1.85;
  color:var(--text-primary);
  margin-bottom:var(--space-2);
  font-weight:400;
}}
.db-body{{
  font-size:15px;
  line-height:1.85;
  color:var(--text-secondary);
  margin-bottom:var(--space-2);
}}
.db-body strong{{color:var(--text-primary);font-weight:600}}
.db-meta-line{{
  font-size:13px;
  color:var(--text-muted);
  font-style:italic;
}}
.db-callout{{
  padding:var(--space-3);
  border-radius:var(--radius-md);
  margin:var(--space-3) 0;
  border-left:3px solid var(--accent);
  background:var(--bg-panel);
}}
.db-callout-warn{{border-left-color:var(--accent-red);background:var(--accent-red-bg)}}
.db-action-block{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-3);
  border-left:3px solid var(--text-muted);
}}
.db-buy{{border-left-color:var(--accent-green)}}
.db-sell{{border-left-color:var(--accent-red)}}
.db-pick-block{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-3);
  border-left:3px solid var(--accent);
}}
.db-action-title{{
  font-size:18px;
  font-weight:600;
  margin-bottom:var(--space-2);
  display:flex;flex-wrap:wrap;align-items:center;gap:var(--space-2);
  line-height:1.4;
}}
.db-master-tag{{
  font-size:11px;
  font-family:'JetBrains Mono',monospace;
  color:var(--accent);
  background:var(--accent-soft);
  padding:2px 10px;
  border-radius:var(--radius-pill);
  font-weight:500;
  letter-spacing:0;
}}
.db-sector-tag{{
  font-size:11px;
  color:var(--text-muted);
  background:var(--bg-panel);
  padding:2px 10px;
  border-radius:var(--radius-pill);
  font-weight:500;
}}
.db-scenario{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  padding:var(--space-3);
  margin-bottom:var(--space-2);
}}
.db-bull{{border-left:3px solid var(--accent-green)}}
.db-base{{border-left:3px solid var(--accent)}}
.db-bear{{border-left:3px solid var(--accent-red)}}
.db-scenario-header{{
  display:flex;align-items:center;gap:var(--space-2);
  margin-bottom:var(--space-1);
}}
.db-scenario-emoji{{font-size:22px}}
.db-scenario-name{{
  font-size:16px;font-weight:600;
  color:var(--text-primary);
}}
.db-scenario-prob{{
  margin-left:auto;
  font-size:14px;
  font-weight:600;
  font-family:'JetBrains Mono',monospace;
  color:var(--text-secondary);
  padding:2px 10px;
  background:var(--bg-panel);
  border-radius:var(--radius-pill);
}}
.db-list{{
  margin:var(--space-2) 0;
  padding-left:var(--space-4);
  font-size:15px;
  line-height:1.85;
  color:var(--text-secondary);
}}
.db-list li{{margin-bottom:var(--space-1)}}
.db-footer{{
  margin-top:var(--space-5);
  padding-top:var(--space-3);
  border-top:1px solid var(--border);
  text-align:center;
}}
.db-source-list{{
  text-align:left;
  margin:var(--space-2) auto;
  max-width:680px;
  padding-left:var(--space-4);
  font-size:13px;
  color:var(--text-muted);
  line-height:1.85;
}}
.db-source-list li{{margin-bottom:4px}}
.db-source-list em{{font-style:normal;color:var(--accent);font-family:'JetBrains Mono',monospace;font-size:12px}}
.glossary-term{{
  text-decoration:underline dotted var(--accent);
  text-underline-offset:3px;
  cursor:help;
}}
.glossary-hint{{
  display:inline-block;
  margin-left:2px;
  font-size:10px;
  color:var(--accent);
  background:var(--accent-soft);
  padding:0 5px;
  border-radius:50%;
  vertical-align:super;
  font-weight:700;
}}
.db-glossary{{
  margin-top:var(--space-4);
  padding:var(--space-3);
  background:var(--bg-panel);
  border-radius:var(--radius-md);
  border:1px solid var(--border);
}}
.db-glossary-toggle{{
  font-weight:600;
  cursor:pointer;
  color:var(--text-primary);
  padding:var(--space-1) 0;
  list-style:none;
}}
.db-glossary-toggle::-webkit-details-marker{{display:none}}
.db-glossary-toggle::before{{content:"▶ ";font-size:11px;color:var(--accent)}}
.db-glossary[open] .db-glossary-toggle::before{{content:"▼ "}}
.db-glossary-list{{
  margin-top:var(--space-2);
  padding-left:0;
  font-size:14px;
}}
.db-glossary-list dt{{
  font-weight:700;
  color:var(--accent);
  margin-top:var(--space-2);
  font-family:'JetBrains Mono',monospace;
  font-size:13px;
}}
.db-glossary-list dd{{
  margin:2px 0 var(--space-1) 0;
  padding-left:var(--space-2);
  color:var(--text-secondary);
  line-height:1.7;
  border-left:2px solid var(--border);
}}
.db-factor-detail{{
  margin:var(--space-2) 0;
  padding:var(--space-2);
  background:var(--bg-panel);
  border-radius:var(--radius-sm);
  border:1px solid var(--border);
}}
.db-factor-detail summary{{
  cursor:pointer;
  font-weight:600;
  color:var(--accent);
  font-size:13px;
  padding:4px 0;
}}
.db-factor-table{{
  width:100%;
  border-collapse:collapse;
  font-size:13px;
  margin-top:var(--space-2);
  font-family:'JetBrains Mono',monospace;
}}
.db-factor-table th,.db-factor-table td{{
  padding:6px 10px;
  text-align:left;
  border-bottom:1px solid var(--border);
}}
.db-factor-table th{{
  background:var(--bg-panel);
  color:var(--text-muted);
  font-weight:600;
  font-size:12px;
}}
.db-factor-table tr:hover td{{background:var(--accent-soft)}}
.db-meta-line{{
  font-size:13px;
  color:var(--text-muted);
  font-style:italic;
  margin:var(--space-1) 0;
  line-height:1.6;
}}
.db-longterm-banner{{
  display:flex;
  align-items:flex-start;
  gap:var(--space-2);
  margin:var(--space-3) 0;
  padding:var(--space-3);
  background:linear-gradient(135deg,var(--accent-soft) 0%,var(--bg-panel) 100%);
  border-left:3px solid var(--accent);
  border-radius:var(--radius-md);
}}
.db-longterm-banner .icon{{font-size:22px;line-height:1}}
.db-longterm-banner strong{{color:var(--text-primary);display:block;margin-bottom:4px}}
.db-longterm-banner p{{margin:0;font-size:14px;color:var(--text-secondary);line-height:1.7}}

/* === レスポンシブ === */
@media(max-width:768px){{
  .stats-row,.metrics-row{{grid-template-columns:repeat(2,1fr)}}
  .risk-grid,.tsumitate-grid{{grid-template-columns:1fr}}
  .action-grid{{grid-template-columns:1fr}}
  .chart-row{{grid-template-columns:1fr}}
  .tab-nav{{gap:2px;padding:3px}}
  .tab-btn{{padding:8px 12px;font-size:12px}}
  .container{{padding:var(--space-3) var(--space-2)}}
  .header{{padding:var(--space-3);flex-direction:column;align-items:flex-start;gap:var(--space-2)}}
  .header h1{{font-size:20px}}
  .header-meta{{text-align:left}}
  /* iPhone Trading Hero 縮小 */
  .trading-hero{{padding:14px 16px}}
  .trading-hero-value{{font-size:28px}}
  .trading-hero-arrow{{font-size:20px}}
  .trading-hero-pct{{font-size:14px;padding:3px 8px}}
  .trading-hero-sub{{font-size:11px;gap:6px}}
  /* iPhone 大きいフォントを縮小 */
  .stat-value{{font-size:20px}}
  .stat-label{{font-size:10px}}
  .stat-sub{{font-size:11px}}
  .section-title{{font-size:15px}}
  .section-title .icon{{font-size:18px}}
  .port-metric-value{{font-size:18px !important}}
  /* Daily Brief モバイル最適化 */
  .daily-brief{{font-size:14px;padding:var(--space-2) 0}}
  .db-title{{font-size:20px;line-height:1.3}}
  .db-eyebrow{{font-size:10px;letter-spacing:0.5px}}
  .db-section-title{{font-size:16px}}
  .db-lead{{font-size:15px;line-height:1.8}}
  .db-body{{font-size:14px;line-height:1.75}}
  .db-action-title{{font-size:15px}}
  .db-meta{{flex-direction:column;gap:var(--space-1)}}
  .db-meta-item{{font-size:12px}}
  .db-longterm-banner p{{font-size:13px}}
  .db-longterm-banner strong{{font-size:14px}}
  .db-longterm-banner .icon{{font-size:18px}}
  .db-master-tag,.db-sector-tag{{font-size:10px;padding:1px 8px}}
  .db-meta-line{{font-size:12px}}
  .db-list{{font-size:14px;line-height:1.75}}
  /* テーブル横スクロール対応 */
  .port-table-section,.timing-table{{font-size:11px}}
  .db-factor-table{{font-size:11px}}
  .db-factor-table th,.db-factor-table td{{padding:4px 6px}}
}}
@media(min-width:769px){{
  .daily-brief{{font-size:17px}}
  .db-title{{font-size:38px}}
  .db-lead{{font-size:19px}}
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
    <button class="tab-btn active" data-tab="brief" onclick="switchTab(this,'brief')">🗞️ Daily</button>
    <button class="tab-btn" data-tab="weekly" onclick="switchTab(this,'weekly')">📅 Weekly</button>
    <button class="tab-btn" data-tab="monthly" onclick="switchTab(this,'monthly')">📊 Monthly</button>
    <button class="tab-btn" data-tab="portfolio" onclick="switchTab(this,'portfolio')">💼 Portfolio</button>
    <button class="tab-btn" data-tab="action" onclick="switchTab(this,'action')">🎯 Action</button>
    <button class="tab-btn" data-tab="master" onclick="switchTab(this,'master')">🎯 Master Wisdom</button>
    <button class="tab-btn" data-tab="lab" onclick="switchTab(this,'lab')">🚀 Strategy Lab</button>
    <button class="tab-btn" data-tab="history" onclick="switchTab(this,'history')">📚 History</button>
    <button class="tab-btn" data-tab="advanced" onclick="switchTab(this,'advanced')">🔮 Advanced AI</button>
    <button class="tab-btn" data-tab="learning" onclick="switchTab(this,'learning')">🧠 Learning</button>
    <button class="tab-btn" data-tab="overview" onclick="switchTab(this,'overview')">📊 Overview</button>
    <button class="tab-btn" data-tab="risk" onclick="switchTab(this,'risk')">🔥 Risk</button>
    <button class="tab-btn" data-tab="report" onclick="switchTab(this,'report')">🎯 AI Report Card</button>
    <button class="tab-btn" data-tab="prompt" onclick="switchTab(this,'prompt')">🧬 Meta-Prompt</button>
    <button class="tab-btn" data-tab="record" onclick="switchTab(this,'record')">📝 Record</button>
  </div>

  <!-- Tab: Portfolio (default) -->
  <div id="tab-brief" class="tab-content active">
    {daily_brief_html}
  </div>
  <div id="tab-weekly" class="tab-content">
    {weekly_brief_html}
  </div>
  <div id="tab-monthly" class="tab-content">
    {monthly_brief_html}
  </div>
  <div id="tab-portfolio" class="tab-content">
    {portfolio_html}
    {performance_chart_html}
  </div>

  <!-- Tab: Master Wisdom (Buffett 級 9ファクター) -->
  <div id="tab-master" class="tab-content">
    {master_wisdom_html}
  </div>

  <!-- Tab: Strategy Lab (12項目強化群) -->
  <div id="tab-lab" class="tab-content">
    {strategy_lab_html}
  </div>

  <!-- Tab: History (30年歴史パターン) -->
  <div id="tab-history" class="tab-content">
    {history_html}
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
    <div class="trading-hero {('hero-profit' if total_pnl >= 0 else 'hero-loss')}">
      <div class="trading-hero-label">ポートフォリオ含み損益 (USD換算)</div>
      <div class="trading-hero-row">
        <div class="trading-hero-arrow">{'▲' if total_pnl >= 0 else '▼'}</div>
        <div class="trading-hero-value">{'+' if total_pnl >= 0 else '-'}${abs(total_pnl):,.0f}</div>
        <div class="trading-hero-pct">{'+' if total_pnl >= 0 else '-'}{abs(pnl_pct):.2f}%</div>
      </div>
      <div class="trading-hero-sub">
        <span>総資産 <strong>${total_usd:,.0f}</strong></span>
        <span class="dot-sep">·</span>
        <span>{len(stocks)}銘柄</span>
        <span class="dot-sep">·</span>
        <span>{date_str}</span>
      </div>
    </div>
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
  const colors = ['#0075de','#1aae39','#dd5b00','#2a9d99','#ff64c8','#391c57','#523410','#c08532','#097fe8','#213183','#62aef0','#9a4a6b','#1aae39','#2a9d99','#dd5b00','#c08532','#8b6914','#ff64c8'];

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
        legend: {{ position: 'right', labels: {{ color: '#615d59', font: {{ size: 11 }} }} }}
      }}
    }}
  }});

  new Chart(document.getElementById('chartBar'), {{
    type: 'bar',
    data: {{
      labels: overviewData.tickers,
      datasets: [
        {{ label: 'Ensemble', data: overviewData.ensemble, backgroundColor: 'rgba(0,117,222,0.85)' }},
        {{ label: 'Black-Litterman', data: overviewData.bl, backgroundColor: 'rgba(33,49,131,0.75)' }},
        {{ label: 'Equal Weight', data: overviewData.equal, backgroundColor: 'rgba(97,93,89,0.35)' }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#615d59', font: {{ size: 10 }} }} }},
        y: {{ ticks: {{ color: '#615d59', callback: v => v+'%' }}, grid: {{ color: 'rgba(0, 0, 0, 0.06)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#615d59' }} }} }}
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
        backgroundColor: overviewData.ensemble.map(v => v > (100/overviewData.tickers.length) ? 'rgba(221,91,0,0.85)' : 'rgba(26,174,57,0.85)')
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {{
        x: {{ ticks: {{ color: '#615d59', callback: v => v+'%' }}, grid: {{ color: 'rgba(0, 0, 0, 0.06)' }} }},
        y: {{ ticks: {{ color: '#615d59', font: {{ family: 'JetBrains Mono', size: 11 }} }} }}
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
        borderColor: '#0075de',
        backgroundColor: 'rgba(0,117,222,0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: '#0075de'
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        x: {{ ticks: {{ color: '#615d59' }} }},
        y: {{ min: 0, max: 100, ticks: {{ color: '#615d59', callback: v => v+'%' }}, grid: {{ color: 'rgba(0, 0, 0, 0.06)' }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#615d59' }} }} }}
    }}
  }});

  const evals = trackData.evaluations;
  if (evals.length > 0) {{
    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px">';
    html += '<tr style="border-bottom:1px solid var(--border)"><th style="text-align:left;padding:8px;color:var(--text-muted)">日付</th><th style="padding:8px;color:var(--text-muted)">RMSE</th><th style="padding:8px;color:var(--text-muted)">方向性</th><th style="padding:8px;color:var(--text-muted)">較正</th></tr>';
    evals.forEach(e => {{
      html += `<tr style="border-bottom:1px solid var(--border)"><td style="padding:8px;color:var(--text-secondary);font-family:'JetBrains Mono',monospace">${{e.date}}</td><td style="padding:8px;text-align:center;color:#dd5b00;font-family:'JetBrains Mono',monospace">${{e.rmse}}</td><td style="padding:8px;text-align:center;color:#1aae39;font-family:'JetBrains Mono',monospace">${{e.direction_accuracy}}%</td><td style="padding:8px;text-align:center;color:#0075de;font-family:'JetBrains Mono',monospace">${{e.calibration_score}}</td></tr>`;
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
          borderColor: '#0075de',
          backgroundColor: 'rgba(0,117,222,0.08)',
          fill: true,
          tension: 0.3,
          yAxisID: 'y',
          pointRadius: 2,
        }},
        {{
          label: '含み損益 (USD)',
          data: perfData.pnl_usd,
          borderColor: '#1aae39',
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
        x: {{ ticks: {{ color: '#615d59', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(0, 0, 0, 0.04)' }} }},
        y: {{ position: 'left', ticks: {{ color: '#615d59', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: 'rgba(0, 0, 0, 0.06)' }} }},
        y1: {{ position: 'right', ticks: {{ color: '#1aae39', callback: v => '$'+v.toLocaleString() }}, grid: {{ display: false }} }}
      }},
      plugins: {{ legend: {{ labels: {{ color: '#615d59' }} }} }}
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
