#!/usr/bin/env python3
"""
market_data_enricher.py — 銘柄別の包括的マーケットデータを取得・要約

保有銘柄および推奨銘柄に対して以下を取得:
  - 20年分の価格データから算出したテクニカル指標
  - ファンダメンタルズ（PER, PBR, EPS成長率, 利益率等）
  - 市場センチメント（アナリスト推奨, 空売り, 機関投資家比率等）
  - 危機耐性（過去の最大ドローダウンと回復日数）

auto_prompt_cycle.py から呼び出され、プロンプトに1行サマリーとして挿入される。
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("market-enricher")


# ==============================================================================
# テクニカル指標
# ==============================================================================


def compute_cagr(prices: "np.ndarray", years: int) -> Optional[float]:
    """
    CAGR（年率複利成長率）を算出する。

    Args:
        prices: 日次終値の配列
        years: 対象年数

    Returns:
        CAGRのfloat値、またはデータ不足時None
    """
    trading_days = years * 252
    if len(prices) < trading_days:
        return None
    start = prices[-trading_days]
    end = prices[-1]
    if start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def compute_rsi(prices: "np.ndarray", period: int = 14) -> float:
    """
    RSI（相対力指数）を算出する。

    Args:
        prices: 日次終値の配列
        period: RSI計算期間（デフォルト14日）

    Returns:
        RSI値（0-100）
    """
    deltas = np.diff(prices[-period - 50:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_drawdowns(prices: "np.ndarray") -> list:
    """
    過去の主要ドローダウンを検出する。

    Args:
        prices: 日次終値の配列

    Returns:
        ドローダウン辞書のリスト（最大5件、深い順）
    """
    peak = prices[0]
    drawdowns = []
    current_dd_start = 0
    current_dd_max = 0.0

    for i in range(1, len(prices)):
        if prices[i] > peak:
            if current_dd_max < -0.15:
                # 回復完了 → ドローダウンを記録
                recovery_days = i - current_dd_start
                drawdowns.append({
                    "depth": current_dd_max,
                    "recovery_days": recovery_days,
                    "year": 2006 + int(current_dd_start / 252),
                })
            peak = prices[i]
            current_dd_start = i
            current_dd_max = 0.0
        else:
            dd = (prices[i] / peak) - 1
            if dd < current_dd_max:
                current_dd_max = dd

    # 未回復のドローダウン
    if current_dd_max < -0.15:
        drawdowns.append({
            "depth": current_dd_max,
            "recovery_days": -1,
            "year": 2006 + int(current_dd_start / 252),
        })

    drawdowns.sort(key=lambda x: x["depth"])
    return drawdowns[:5]


def compute_technicals(prices: "np.ndarray") -> dict:
    """
    テクニカル指標一式を算出する。

    Args:
        prices: 日次終値の配列（20年分）

    Returns:
        テクニカル指標の辞書
    """
    current = prices[-1]
    high_52w = np.max(prices[-252:])
    low_52w = np.min(prices[-252:])
    range_pct = (current - low_52w) / (high_52w - low_52w + 1e-10)

    ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else current
    ma200 = np.mean(prices[-200:]) if len(prices) >= 200 else current
    golden_cross = ma50 > ma200

    rsi = compute_rsi(prices)

    # ボリンジャーバンド
    bb_period = 20
    bb_std = np.std(prices[-bb_period:])
    bb_mean = np.mean(prices[-bb_period:])
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_position = (current - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # リターン
    def safe_return(days: int) -> Optional[float]:
        """指定日数前からのリターンを算出する。"""
        if len(prices) < days:
            return None
        return (current / prices[-days]) - 1

    # CAGR
    cagr_5y = compute_cagr(prices, 5)
    cagr_10y = compute_cagr(prices, 10)
    cagr_20y = compute_cagr(prices, 20)

    # ドローダウン
    drawdowns = compute_drawdowns(prices)

    # 出来高は別途取得が必要なので省略（infoから取得）

    return {
        "current": current,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "range_pct": range_pct,
        "ma50": ma50,
        "ma200": ma200,
        "golden_cross": golden_cross,
        "rsi": rsi,
        "bb_position": bb_position,
        "return_1m": safe_return(21),
        "return_3m": safe_return(63),
        "return_6m": safe_return(126),
        "return_ytd": safe_return(min(len(prices), 60)),
        "cagr_5y": cagr_5y,
        "cagr_10y": cagr_10y,
        "cagr_20y": cagr_20y,
        "drawdowns": drawdowns,
    }


# ==============================================================================
# ファンダメンタルズ + センチメント
# ==============================================================================


def fetch_fundamentals(ticker_obj: "yf.Ticker") -> dict:
    """
    yfinance の .info からファンダメンタルズとセンチメント指標を取得する。

    Args:
        ticker_obj: yfinance の Ticker オブジェクト

    Returns:
        ファンダメンタルズ指標の辞書
    """
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    def g(key: str, default=None):
        """info辞書から安全に値を取得する。"""
        return info.get(key, default)

    # ファンダ
    per = g("trailingPE")
    forward_per = g("forwardPE")
    pbr = g("priceToBook")
    market_cap = g("marketCap")
    revenue_growth = g("revenueGrowth")
    earnings_growth = g("earningsGrowth")
    profit_margin = g("profitMargins")
    operating_margin = g("operatingMargins")
    dividend_yield = g("dividendYield")

    # センチメント
    target_mean = g("targetMeanPrice")
    target_low = g("targetLowPrice")
    target_high = g("targetHighPrice")
    recommend = g("recommendationKey")
    recommend_mean = g("recommendationMean")
    num_analysts = g("numberOfAnalystOpinions")
    short_ratio = g("shortPercentOfFloat") or g("shortRatio")
    institutional_pct = g("heldPercentInstitutions")

    # 次回決算
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict):
            next_earnings = cal.get("Earnings Date", [None])[0]
        else:
            next_earnings = None
    except Exception:
        next_earnings = None

    # 直近決算サプライズ
    try:
        earnings_hist = ticker_obj.earnings_dates
        if earnings_hist is not None and len(earnings_hist) > 0:
            last_surprise = earnings_hist.iloc[0].get("Surprise(%)", None)
        else:
            last_surprise = None
    except Exception:
        last_surprise = None

    return {
        "per": per,
        "forward_per": forward_per,
        "pbr": pbr,
        "market_cap": market_cap,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
        "dividend_yield": dividend_yield,
        "target_mean": target_mean,
        "target_low": target_low,
        "target_high": target_high,
        "recommend": recommend,
        "recommend_mean": recommend_mean,
        "num_analysts": num_analysts,
        "short_ratio": short_ratio,
        "institutional_pct": institutional_pct,
        "next_earnings": str(next_earnings) if next_earnings else None,
        "last_surprise": last_surprise,
    }


# ==============================================================================
# メイン: 全銘柄データ取得
# ==============================================================================


def enrich_ticker(ticker: str) -> Optional[dict]:
    """
    1銘柄の包括的データを取得する。

    Args:
        ticker: ティッカーシンボル

    Returns:
        全指標を含む辞書、またはNone
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="20y")
        if hist.empty or len(hist) < 50:
            logger.warning("⚠️ %s: データ不足 (%d日)", ticker, len(hist))
            return None

        prices = hist["Close"].values
        technicals = compute_technicals(prices)
        fundamentals = fetch_fundamentals(t)

        return {
            "ticker": ticker,
            "data_days": len(prices),
            **technicals,
            **fundamentals,
        }
    except Exception as e:
        logger.error("❌ %s: データ取得失敗 — %s", ticker, e)
        return None


def enrich_all(tickers: list, new_picks: list = None) -> dict:
    """
    全銘柄のデータを取得する。

    Args:
        tickers: 保有銘柄リスト
        new_picks: 推奨新規銘柄リスト

    Returns:
        銘柄→データの辞書
    """
    all_tickers = list(set(tickers + (new_picks or [])))
    logger.info("📊 %d銘柄のデータ取得開始...", len(all_tickers))

    results = {}
    for ticker in all_tickers:
        data = enrich_ticker(ticker)
        if data:
            results[ticker] = data
            logger.info("✅ %s: %d日分取得", ticker, data["data_days"])

    logger.info("📊 %d/%d銘柄のデータ取得完了", len(results), len(all_tickers))
    return results


# ==============================================================================
# プロンプト用テキスト生成
# ==============================================================================


def format_for_prompt(enriched: dict) -> str:
    """
    全銘柄の要約データをプロンプト用テキストに変換する。

    Args:
        enriched: enrich_all() の戻り値

    Returns:
        プロンプトに挿入するテキスト
    """
    lines = ["═══ SECTION A2: 銘柄別マーケットデータ（20年分析） ═══\n"]

    for ticker, d in sorted(enriched.items()):
        # テクニカル行
        trend = "↗" if d.get("golden_cross") else "↘"
        cagr_parts = []
        for label, key in [("20Y", "cagr_20y"), ("10Y", "cagr_10y"), ("5Y", "cagr_5y")]:
            v = d.get(key)
            if v is not None:
                cagr_parts.append(f"{label}={v:+.0%}")
        cagr_str = ", ".join(cagr_parts) if cagr_parts else "N/A"

        line1 = (
            f"{ticker}: CAGR({cagr_str}) | "
            f"52w: ${d.get('low_52w', 0):.0f}-${d.get('high_52w', 0):.0f} "
            f"(現在: {d.get('range_pct', 0):.0%}) | "
            f"MA50{'>' if d.get('golden_cross') else '<'}MA200 {trend} | "
            f"RSI={d.get('rsi', 0):.0f}"
        )

        # リターン
        ret_parts = []
        for label, key in [("1M", "return_1m"), ("3M", "return_3m"), ("6M", "return_6m")]:
            v = d.get(key)
            if v is not None:
                ret_parts.append(f"{label}={v:+.1%}")
        ret_str = " | ".join(ret_parts) if ret_parts else ""

        # ファンダ行
        funda_parts = []
        if d.get("per"):
            funda_parts.append(f"PER={d['per']:.0f}")
        if d.get("forward_per"):
            funda_parts.append(f"fPER={d['forward_per']:.0f}")
        if d.get("pbr"):
            funda_parts.append(f"PBR={d['pbr']:.1f}")
        if d.get("earnings_growth") is not None:
            funda_parts.append(f"EPS成長{d['earnings_growth']:+.0%}")
        if d.get("revenue_growth") is not None:
            funda_parts.append(f"売上成長{d['revenue_growth']:+.0%}")
        if d.get("operating_margin") is not None:
            funda_parts.append(f"営業利益率{d['operating_margin']:.0%}")
        if d.get("dividend_yield") and d["dividend_yield"] > 0:
            funda_parts.append(f"配当{d['dividend_yield']:.1%}")
        funda_str = " | ".join(funda_parts) if funda_parts else "N/A"

        # センチメント行
        sent_parts = []
        if d.get("recommend"):
            sent_parts.append(f"推奨={d['recommend']}")
        if d.get("target_mean") and d.get("current"):
            upside = (d["target_mean"] / d["current"] - 1)
            sent_parts.append(f"目標${d['target_mean']:.0f}({upside:+.0%})")
        if d.get("num_analysts"):
            sent_parts.append(f"アナリスト{d['num_analysts']}名")
        if d.get("short_ratio"):
            sent_parts.append(f"空売り{d['short_ratio']:.1%}")
        if d.get("institutional_pct"):
            sent_parts.append(f"機関{d['institutional_pct']:.0%}")
        if d.get("next_earnings"):
            sent_parts.append(f"次決算={d['next_earnings'][:10]}")
        if d.get("last_surprise") is not None:
            sent_parts.append(f"前回サプライズ{d['last_surprise']:+.1%}")
        sent_str = " | ".join(sent_parts) if sent_parts else "N/A"

        # 危機耐性行
        dd_parts = []
        for dd in d.get("drawdowns", [])[:3]:
            depth = dd["depth"]
            rec = dd["recovery_days"]
            yr = dd.get("year", "?")
            if rec > 0:
                dd_parts.append(f"{yr}年:{depth:.0%}→{rec}日回復")
            else:
                dd_parts.append(f"{yr}年:{depth:.0%}(未回復)")
        dd_str = " | ".join(dd_parts) if dd_parts else "大幅下落なし"

        lines.append(f"  {line1}")
        if ret_str:
            lines.append(f"     リターン: {ret_str}")
        lines.append(f"     ファンダ: {funda_str}")
        lines.append(f"     センチメント: {sent_str}")
        lines.append(f"     危機耐性: {dd_str}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # テスト: 3銘柄だけ取得
    test_tickers = ["NVDA", "GLD", "VYM"]
    results = enrich_all(test_tickers)
    print(format_for_prompt(results))
