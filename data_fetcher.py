# -*- coding: utf-8 -*-
"""
Portfolio Data Fetcher — 堅牢なマーケットデータ収集モジュール
==========================================================
yfinance による IPブロック耐性＆エラー耐性データ収集。

yfinance 1.2.0+ は curl_cffi を内部使用するため、
カスタムSession (requests_cache/ratelimiter) は注入不可。
代わりに time.sleep() でリクエスト間隔を制御する。

すべての個別指標取得に try-except を適用し、
取得失敗時は None/"N/A" で処理を続行する。
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("portfolio-bot.data")

# リクエスト間隔 (秒) — IPブロック回避
REQUEST_INTERVAL = 0.5


def _rate_limit() -> None:
    """IPブロック回避のためリクエスト間に待機する。"""
    time.sleep(REQUEST_INTERVAL)


# ==============================================================================
# ヘルパー関数
# ==============================================================================


def _safe_get(info: dict, key: str, default: Any = None) -> Any:
    """
    辞書から安全に値を取得する。

    Args:
        info: yfinance の info 辞書
        key: 取得するキー
        default: デフォルト値

    Returns:
        取得した値、またはデフォルト値
    """
    try:
        val = info.get(key, default)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default


def _calc_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    RSI (Relative Strength Index) を pandas で独自計算する。

    Args:
        prices: 終値の時系列
        period: RSI 計算期間

    Returns:
        RSI 値 (0-100)、データ不足時は None
    """
    if prices is None or len(prices) < period + 1:
        return None
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return round(float(val), 1) if pd.notna(val) else None
    except Exception as e:
        logger.warning("RSI 計算失敗: %s", e)
        return None


def _calc_sma_deviation(prices: pd.Series, period: int) -> Optional[float]:
    """
    SMA (単純移動平均) からの乖離率を計算する。

    Args:
        prices: 終値の時系列
        period: SMA 期間

    Returns:
        乖離率 (%)、データ不足時は None
    """
    if prices is None or len(prices) < period:
        return None
    try:
        sma = prices.rolling(window=period).mean().iloc[-1]
        current = prices.iloc[-1]
        if pd.notna(sma) and sma != 0:
            return round(float((current - sma) / sma * 100), 2)
        return None
    except Exception as e:
        logger.warning("SMA乖離率計算失敗 (期間=%d): %s", period, e)
        return None


# ==============================================================================
# マクロ指標
# ==============================================================================


def fetch_macro_data(macro_tickers: list[str]) -> dict[str, Any]:
    """
    マクロ指標（ドル円・米10年債・VIX）を取得する。

    Args:
        macro_tickers: マクロ指標のティッカーリスト

    Returns:
        マクロデータ辞書
    """
    result = {}
    for ticker_str in macro_tickers:
        try:
            _rate_limit()
            ticker = yf.Ticker(ticker_str)
            hist = ticker.history(period="1mo")

            if hist.empty:
                result[ticker_str] = {"current": None, "month_change_pct": None}
                continue

            current = float(hist["Close"].iloc[-1])
            first = float(hist["Close"].iloc[0])
            change_pct = round((current - first) / first * 100, 2) if first else None

            result[ticker_str] = {
                "current": round(current, 4),
                "month_change_pct": change_pct,
            }
            logger.info("  マクロ取得: %s = %.4f (1M: %s%%)", ticker_str, current, change_pct)

        except Exception as e:
            logger.warning("  マクロ取得失敗 %s: %s", ticker_str, e)
            result[ticker_str] = {"current": None, "month_change_pct": None}

    return result


# ==============================================================================
# 銘柄個別データ
# ==============================================================================


def fetch_stock_data(ticker_str: str, config: dict) -> dict[str, Any]:
    """
    個別銘柄の全指標を取得する。

    取得項目: 基本指標、テクニカル、アナリスト、ニュース、
    インサイダー、オプション、決算予定

    Args:
        ticker_str: ティッカーシンボル
        config: 解析パラメータ設定

    Returns:
        銘柄データ辞書
    """
    analysis_cfg = config.get("analysis", {})
    rsi_period = analysis_cfg.get("rsi_period", 14)
    sma_short = analysis_cfg.get("sma_short", 50)
    sma_long = analysis_cfg.get("sma_long", 200)
    news_count = analysis_cfg.get("news_count", 3)
    insider_months = analysis_cfg.get("insider_lookback_months", 3)
    earnings_days = analysis_cfg.get("earnings_alert_days", 14)

    data: dict[str, Any] = {"ticker": ticker_str}

    _rate_limit()
    ticker = yf.Ticker(ticker_str)

    # --- 基本情報 (info) ---
    info: dict = {}
    try:
        info = ticker.info or {}
    except Exception as e:
        logger.warning("  info取得失敗 %s: %s", ticker_str, e)

    data["company_name"] = _safe_get(info, "shortName", ticker_str)
    data["market_cap"] = _safe_get(info, "marketCap")
    data["trailing_pe"] = _safe_get(info, "trailingPE")
    data["forward_pe"] = _safe_get(info, "forwardPE")
    data["pb_ratio"] = _safe_get(info, "priceToBook")
    data["eps"] = _safe_get(info, "trailingEps")
    data["dividend_yield"] = _safe_get(info, "dividendYield")

    # --- 1ヶ月騰落率 ---
    try:
        _rate_limit()
        hist_1m = ticker.history(period="1mo")
        if not hist_1m.empty and len(hist_1m) >= 2:
            first_p = float(hist_1m["Close"].iloc[0])
            last_p = float(hist_1m["Close"].iloc[-1])
            data["month_return_pct"] = round((last_p - first_p) / first_p * 100, 2) if first_p else None
        else:
            data["month_return_pct"] = None
    except Exception as e:
        logger.warning("  1M騰落率取得失敗 %s: %s", ticker_str, e)
        data["month_return_pct"] = None

    # --- テクニカル指標 ---
    try:
        _rate_limit()
        hist_long = ticker.history(period="1y")
        closes = hist_long["Close"] if not hist_long.empty else pd.Series(dtype=float)
        data["rsi_14"] = _calc_rsi(closes, rsi_period)
        data["sma_50_dev_pct"] = _calc_sma_deviation(closes, sma_short)
        data["sma_200_dev_pct"] = _calc_sma_deviation(closes, sma_long)
    except Exception as e:
        logger.warning("  テクニカル取得失敗 %s: %s", ticker_str, e)
        data["rsi_14"] = None
        data["sma_50_dev_pct"] = None
        data["sma_200_dev_pct"] = None

    # --- アナリストコンセンサス ---
    try:
        rec = info.get("recommendationKey", "N/A")
        target = _safe_get(info, "targetMeanPrice")
        current_price = _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice")

        target_dev = None
        if target and current_price and current_price > 0:
            target_dev = round((target - current_price) / current_price * 100, 2)

        buy_count = _safe_get(info, "numberOfAnalystOpinions", 0)
        data["analyst"] = {
            "recommendation": rec,
            "target_price": target,
            "target_deviation_pct": target_dev,
            "analyst_count": buy_count,
        }
    except Exception as e:
        logger.warning("  アナリスト取得失敗 %s: %s", ticker_str, e)
        data["analyst"] = {"recommendation": "N/A", "target_price": None,
                           "target_deviation_pct": None, "analyst_count": 0}

    # --- 決算予定日 ---
    try:
        cal = ticker.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if dates:
                next_date = dates[0] if isinstance(dates, list) else dates
                if hasattr(next_date, "date"):
                    next_date = next_date.date()
                days_until = (next_date - datetime.now().date()).days
                data["earnings"] = {
                    "next_date": str(next_date),
                    "days_until": days_until,
                    "imminent": 0 <= days_until <= earnings_days,
                }
            else:
                data["earnings"] = {"next_date": None, "days_until": None, "imminent": False}
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            data["earnings"] = {"next_date": "see_calendar", "days_until": None, "imminent": False}
        else:
            data["earnings"] = {"next_date": None, "days_until": None, "imminent": False}
    except Exception as e:
        logger.warning("  決算予定取得失敗 %s: %s", ticker_str, e)
        data["earnings"] = {"next_date": None, "days_until": None, "imminent": False}

    # --- ニュース (英語のまま) ---
    try:
        news_items = ticker.news or []
        headlines = []
        for item in news_items[:news_count]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
        data["news_headlines"] = headlines
    except Exception as e:
        logger.warning("  ニュース取得失敗 %s: %s", ticker_str, e)
        data["news_headlines"] = []

    # --- インサイダー取引 ---
    try:
        insider_df = ticker.insider_transactions
        if insider_df is not None and not insider_df.empty:
            cutoff = datetime.now() - timedelta(days=insider_months * 30)
            if "Start Date" in insider_df.columns:
                date_col = "Start Date"
            elif "startDate" in insider_df.columns:
                date_col = "startDate"
            else:
                date_col = insider_df.columns[0]

            insider_df[date_col] = pd.to_datetime(insider_df[date_col], errors="coerce")
            recent = insider_df[insider_df[date_col] >= cutoff]

            buy_count = 0
            sell_count = 0
            if "Transaction" in recent.columns:
                txn_col = "Transaction"
            elif "Text" in recent.columns:
                txn_col = "Text"
            else:
                txn_col = None

            if txn_col:
                for _, row in recent.iterrows():
                    txn = str(row.get(txn_col, "")).lower()
                    if "buy" in txn or "purchase" in txn:
                        buy_count += 1
                    elif "sell" in txn or "sale" in txn:
                        sell_count += 1

            data["insider"] = {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "net": buy_count - sell_count,
            }
        else:
            data["insider"] = {"buy_count": 0, "sell_count": 0, "net": 0}
    except Exception as e:
        logger.warning("  インサイダー取得失敗 %s: %s", ticker_str, e)
        data["insider"] = {"buy_count": 0, "sell_count": 0, "net": 0}

    # --- オプション: プット/コール・レシオ ---
    try:
        _rate_limit()
        exp_dates = ticker.options
        if exp_dates:
            nearest = exp_dates[0]
            opt = ticker.option_chain(nearest)
            call_vol = opt.calls["volume"].sum() if "volume" in opt.calls.columns else 0
            put_vol = opt.puts["volume"].sum() if "volume" in opt.puts.columns else 0
            pc_ratio = round(float(put_vol / call_vol), 3) if call_vol > 0 else None
            data["options"] = {
                "expiry": nearest,
                "put_volume": int(put_vol) if pd.notna(put_vol) else 0,
                "call_volume": int(call_vol) if pd.notna(call_vol) else 0,
                "put_call_ratio": pc_ratio,
            }
        else:
            data["options"] = {"expiry": None, "put_volume": 0, "call_volume": 0, "put_call_ratio": None}
    except Exception as e:
        logger.warning("  オプション取得失敗 %s: %s", ticker_str, e)
        data["options"] = {"expiry": None, "put_volume": 0, "call_volume": 0, "put_call_ratio": None}

    return data


# ==============================================================================
# 相関マトリックス
# ==============================================================================


def calc_correlation_matrix(
    tickers: list[str],
    period_days: int = 252,
) -> dict[str, Any]:
    """
    銘柄間の価格相関マトリックスを計算する。

    Args:
        tickers: ティッカーリスト
        period_days: 相関計算期間（営業日数）

    Returns:
        {ticker_pair: correlation} 形式の辞書
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=int(period_days * 1.5))

        price_data = {}
        for t in tickers:
            try:
                _rate_limit()
                hist = yf.Ticker(t).history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                if not hist.empty:
                    price_data[t] = hist["Close"]
            except Exception as e:
                logger.warning("  相関データ取得失敗 %s: %s", t, e)

        if len(price_data) < 2:
            return {"note": "相関計算に必要なデータが不足"}

        df = pd.DataFrame(price_data).dropna()
        if df.empty or len(df) < 20:
            return {"note": "相関計算に十分なデータなし"}

        corr = df.corr()
        pairs = {}
        for i, t1 in enumerate(corr.columns):
            for j, t2 in enumerate(corr.columns):
                if i < j:
                    val = corr.iloc[i, j]
                    if pd.notna(val):
                        pairs[f"{t1}/{t2}"] = round(float(val), 3)

        return pairs

    except Exception as e:
        logger.error("相関マトリックス計算エラー: %s", e)
        return {"error": str(e)}


# ==============================================================================
# メイン収集関数
# ==============================================================================


def collect_all_data(config: dict) -> dict[str, Any]:
    """
    全データを統合収集するメイン関数。

    config.json の設定に基づき、マクロ・個別銘柄・相関を収集し、
    トークン最適化のためサニタイズした辞書を返す。

    Args:
        config: config.json の内容

    Returns:
        AIに渡すサニタイズ済みデータ辞書
    """
    logger.info("=" * 50)
    logger.info("データ収集開始")
    logger.info("=" * 50)

    analysis_cfg = config.get("analysis", {})
    corr_days = analysis_cfg.get("correlation_period_days", 252)

    # --- マクロ指標 ---
    logger.info("[1/4] マクロ指標取得中...")
    macro_tickers = config.get("macro_tickers", ["JPY=X", "^TNX", "^VIX"])
    macro = fetch_macro_data(macro_tickers)

    # --- 個別銘柄 ---
    holdings = config.get("portfolio", {}).get("holdings", [])
    tickers_list = [h["ticker"] for h in holdings]

    logger.info("[2/4] 個別銘柄データ取得中 (%d銘柄)...", len(tickers_list))
    stocks = []
    for h in holdings:
        t = h["ticker"]
        logger.info("  取得中: %s", t)
        stock_data = fetch_stock_data(t, config)
        stock_data["shares_held"] = h.get("shares", 0)
        stock_data["sector"] = h.get("sector", "Unknown")
        stocks.append(stock_data)

    # --- 相関マトリックス ---
    logger.info("[3/4] 相関マトリックス計算中...")
    correlation = calc_correlation_matrix(tickers_list, corr_days)

    # --- サニタイズ (トークン最適化) ---
    logger.info("[4/4] データサニタイズ中...")
    result = {
        "timestamp": datetime.now().isoformat(),
        "macro": macro,
        "stocks": stocks,
        "correlation": correlation,
    }

    logger.info("データ収集完了 (銘柄数: %d)", len(stocks))
    return result
