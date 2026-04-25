#!/usr/bin/env python3
"""
historical_pattern_extractor.py — 過去の歴史的パターンから学ぶ統計エンジン

【哲学】
バフェット: "歴史は完全には繰り返されないが、韻を踏む"
30年の歴史から以下を学習し、現在の判断に「歴史的視点」を与える:
  - 1987 Black Monday (暴落-22%/1日)
  - 2000 Dot-com Bubble (-49%/2.5年)
  - 2008 GFC (-57%/1.4年)
  - 2020 COVID (-34%/1ヶ月、回復5ヶ月)
  - 2022 Tech Recession (-35%/1年)
  - その他の中規模調整 (-10〜20%)

【抽出する統計】
  1. 大暴落イベント (Drawdown > 15% / 60日以内)
     - 深さ / 期間 / 回復日数 / きっかけ
  2. ブル/ベア サイクル
     - 平均ブル相場長 / 平均ベア相場長
     - ブル中のドローダウン頻度
  3. VIX レジーム分布
     - 各レジーム (low_vol/transition/crisis) の歴史的頻度
     - 平均継続期間
  4. セクターリターン分布
     - 各セクターの長期年率 / Sharpe
  5. Buffett 逆張り効果の検証
     - VIX>30 で買った場合の N年後リターン
     - 過去最大DD 直前 vs 直後の動き

【現在判断への活用】
  - 「現在のVIX=22 は過去30年でX percentile」
  - 「現在のドローダウン -8% は過去のブル相場で平均的に発生した中規模調整」
  - 「現在のレジーム移行期は過去30年で平均X日継続後にY %発生」

使用:
    python historical_pattern_extractor.py
    # または import 経由で
    from historical_pattern_extractor import extract_all_patterns
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hist-patterns")

SCRIPT_DIR = Path(__file__).resolve().parent
PATTERNS_PATH = SCRIPT_DIR / "data" / "historical_patterns.json"


# ==============================================================================
# 1. 大暴落イベント検出
# ==============================================================================


def detect_major_crashes(
    prices: np.ndarray,
    threshold: float = -0.15,
    window_days: int = 60,
    min_recovery_days: int = 30,
) -> list[dict[str, Any]]:
    """過去の重大暴落 (60日以内に -15%以上) を検出する。

    Args:
        prices: 日次終値配列
        threshold: 暴落判定閾値（負値）
        window_days: 暴落検出ウィンドウ
        min_recovery_days: 最低回復ウィンドウ

    Returns:
        各暴落イベントの辞書リスト
    """
    if len(prices) < 100:
        return []

    crashes = []
    n = len(prices)
    i = window_days

    while i < n:
        # window_days 内での最大ドローダウン計算
        window_high = float(np.max(prices[max(0, i - window_days):i + 1]))
        current = float(prices[i])
        dd = (current - window_high) / window_high if window_high > 0 else 0

        if dd <= threshold:
            # 暴落検出 - 開始日 (window_high の位置)
            start_idx = int(np.argmax(prices[max(0, i - window_days):i + 1])) + max(0, i - window_days)
            bottom_idx = i
            # ボトムを更に追跡
            for j in range(i, min(n, i + window_days)):
                if prices[j] < prices[bottom_idx]:
                    bottom_idx = j
            bottom_price = float(prices[bottom_idx])
            bottom_dd = (bottom_price - window_high) / window_high

            # 回復日数: window_high まで戻る日
            recovery_days = None
            for j in range(bottom_idx, n):
                if prices[j] >= window_high:
                    recovery_days = j - bottom_idx
                    break

            crashes.append({
                "start_idx": start_idx,
                "bottom_idx": bottom_idx,
                "depth": round(float(bottom_dd), 4),
                "duration_days": int(bottom_idx - start_idx),
                "recovery_days": int(recovery_days) if recovery_days else None,
                "approx_year": _idx_to_year(start_idx, n),
            })
            # 次の暴落探索は回復後 or 90日後から
            i = (bottom_idx + (recovery_days or 90)) + 1
        else:
            i += 5  # 5日ステップ

    return crashes


def _idx_to_year(idx: int, total_n: int, end_year: int = None) -> int:
    """インデックスを概算年に変換。"""
    if end_year is None:
        end_year = datetime.now().year
    days_from_end = total_n - idx
    return int(end_year - days_from_end / 252)


# ==============================================================================
# 2. ブル/ベアサイクル分析
# ==============================================================================


def analyze_bull_bear_cycles(
    prices: np.ndarray,
    bull_threshold: float = 0.20,
    bear_threshold: float = -0.20,
) -> dict[str, Any]:
    """ブル/ベアサイクルを抽出し、平均期間とリターンを計算。

    ブル: 過去最高値から +20% 上昇開始時から次のベア入りまで
    ベア: 直近高値から -20% 下落時から次のブル入りまで

    Args:
        prices: 日次終値
        bull_threshold: ブル開始閾値
        bear_threshold: ベア開始閾値

    Returns:
        サイクル統計
    """
    if len(prices) < 252:
        return {"bull_cycles": [], "bear_cycles": [], "note": "データ不足"}

    bull_cycles = []
    bear_cycles = []

    state = "bull"  # 開始は楽観的に
    cycle_start = 0
    cycle_extreme = float(prices[0])

    for i in range(1, len(prices)):
        p = float(prices[i])
        if state == "bull":
            cycle_extreme = max(cycle_extreme, p)
            if (p - cycle_extreme) / cycle_extreme <= bear_threshold:
                bull_cycles.append({
                    "start_idx": cycle_start,
                    "end_idx": i,
                    "duration_days": i - cycle_start,
                    "return_pct": round((cycle_extreme - float(prices[cycle_start])) / float(prices[cycle_start]), 4),
                })
                state = "bear"
                cycle_start = i
                cycle_extreme = p
        else:  # bear
            cycle_extreme = min(cycle_extreme, p)
            if (p - cycle_extreme) / cycle_extreme >= bull_threshold:
                bear_cycles.append({
                    "start_idx": cycle_start,
                    "end_idx": i,
                    "duration_days": i - cycle_start,
                    "return_pct": round((cycle_extreme - float(prices[cycle_start])) / float(prices[cycle_start]), 4),
                })
                state = "bull"
                cycle_start = i
                cycle_extreme = p

    # 集計
    bull_durs = [c["duration_days"] for c in bull_cycles]
    bear_durs = [c["duration_days"] for c in bear_cycles]
    return {
        "bull_cycles": bull_cycles,
        "bear_cycles": bear_cycles,
        "n_bull": len(bull_cycles),
        "n_bear": len(bear_cycles),
        "avg_bull_duration_days": round(float(np.mean(bull_durs)) if bull_durs else 0, 1),
        "avg_bear_duration_days": round(float(np.mean(bear_durs)) if bear_durs else 0, 1),
        "median_bull_return": round(float(np.median([c["return_pct"] for c in bull_cycles])) if bull_cycles else 0, 4),
        "median_bear_return": round(float(np.median([c["return_pct"] for c in bear_cycles])) if bear_cycles else 0, 4),
    }


# ==============================================================================
# 3. VIX レジーム分布
# ==============================================================================


def analyze_vix_regimes(vix_arr: np.ndarray) -> dict[str, Any]:
    """VIX の歴史的分布から各レジームの頻度・継続期間を学習。

    Args:
        vix_arr: VIX 日次値

    Returns:
        レジーム統計
    """
    if vix_arr is None or len(vix_arr) < 100:
        return {"note": "VIX データ不足"}

    # 各日のレジーム
    regimes = []
    for v in vix_arr:
        if v < 15:
            regimes.append("low_vol")
        elif v < 25:
            regimes.append("transition")
        else:
            regimes.append("crisis")

    # 連続期間の集計
    durations = {"low_vol": [], "transition": [], "crisis": []}
    current = regimes[0]
    cur_len = 1
    for r in regimes[1:]:
        if r == current:
            cur_len += 1
        else:
            durations[current].append(cur_len)
            current = r
            cur_len = 1
    durations[current].append(cur_len)

    # 統計
    counts = {r: regimes.count(r) for r in ["low_vol", "transition", "crisis"]}
    total = len(regimes)

    # VIX 分位数
    pct = {
        "p25": round(float(np.percentile(vix_arr, 25)), 2),
        "p50": round(float(np.percentile(vix_arr, 50)), 2),
        "p75": round(float(np.percentile(vix_arr, 75)), 2),
        "p90": round(float(np.percentile(vix_arr, 90)), 2),
        "p95": round(float(np.percentile(vix_arr, 95)), 2),
        "p99": round(float(np.percentile(vix_arr, 99)), 2),
        "mean": round(float(np.mean(vix_arr)), 2),
        "max_observed": round(float(np.max(vix_arr)), 2),
    }

    return {
        "frequencies": {r: round(counts[r] / total, 4) for r in counts},
        "avg_duration_days": {
            r: round(float(np.mean(durations[r])) if durations[r] else 0, 1)
            for r in durations
        },
        "max_duration_days": {
            r: int(max(durations[r])) if durations[r] else 0 for r in durations
        },
        "vix_percentiles": pct,
        "current_vix_percentile": _percentile_of(vix_arr, float(vix_arr[-1])),
    }


def _percentile_of(arr: np.ndarray, val: float) -> int:
    """値が分布の何パーセンタイルかを返す。"""
    if len(arr) == 0:
        return 50
    return int(np.sum(arr <= val) / len(arr) * 100)


# ==============================================================================
# 4. Buffett 逆張りシグナルの歴史的検証
# ==============================================================================


def validate_buffett_contrarian(
    prices: np.ndarray,
    vix_arr: np.ndarray,
    horizons_days: list[int] = None,
    vix_threshold: float = 30,
) -> dict[str, Any]:
    """「VIX > 30 (恐怖時) に買ったら N年後リターンは？」をデータで検証。

    Args:
        prices: マーケット価格 (S&P 500 推奨)
        vix_arr: VIX 配列 (prices と同じ長さ想定)
        horizons_days: 検証ホライズン
        vix_threshold: 恐怖判定閾値

    Returns:
        各ホライズンの平均リターン + 勝率
    """
    if horizons_days is None:
        horizons_days = [30, 90, 252, 504, 1260]  # 1ヶ月 / 3ヶ月 / 1年 / 2年 / 5年

    if vix_arr is None or len(vix_arr) < 252 or prices is None:
        return {"note": "データ不足"}

    # 共通インデックスにアラインメント (末尾基準)
    n = min(len(prices), len(vix_arr))
    p = prices[-n:]
    v = vix_arr[-n:]

    fear_indices = np.where(v >= vix_threshold)[0]
    if len(fear_indices) == 0:
        return {"note": "VIX>30 の日が観測されず"}

    results = {}
    for h in horizons_days:
        rets = []
        for idx in fear_indices:
            if idx + h >= n:
                continue
            entry = float(p[idx])
            future = float(p[idx + h])
            if entry > 0:
                rets.append((future - entry) / entry)
        if rets:
            results[f"{h}d"] = {
                "n_observations": len(rets),
                "mean_return": round(float(np.mean(rets)), 4),
                "median_return": round(float(np.median(rets)), 4),
                "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
                "max": round(float(np.max(rets)), 4),
                "min": round(float(np.min(rets)), 4),
            }
    return {
        "vix_threshold": vix_threshold,
        "n_fear_days": int(len(fear_indices)),
        "by_horizon": results,
        "buffett_validation": "VIX>30で買えば歴史的に高勝率"
                              if any(v["win_rate"] > 0.7 for v in results.values())
                              else "勝率は中程度",
    }


# ==============================================================================
# 5. 統合実行 - 全パターン抽出
# ==============================================================================


def extract_all_patterns(
    spy_arr: Optional[np.ndarray],
    vix_arr: Optional[np.ndarray],
    ticker_prices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """全パターン抽出を実行し JSON にまとめる。

    Args:
        spy_arr: S&P 500 価格 (長期推奨)
        vix_arr: VIX 配列
        ticker_prices: 個別銘柄価格

    Returns:
        全パターン統合辞書
    """
    logger.info("📚 歴史的パターン抽出開始...")
    payload = {"executed_at": datetime.now().isoformat()}

    # 1. 大暴落
    if spy_arr is not None and len(spy_arr) > 252:
        crashes = detect_major_crashes(spy_arr, threshold=-0.15)
        payload["major_crashes"] = {
            "n_events": len(crashes),
            "events": crashes,
            "avg_depth": round(float(np.mean([c["depth"] for c in crashes])), 4) if crashes else 0,
            "avg_recovery_days": round(float(np.mean([c["recovery_days"] for c in crashes if c["recovery_days"]])), 1) if any(c["recovery_days"] for c in crashes) else 0,
        }
        logger.info("  💥 大暴落 %d件: 平均深さ %.1f%% / 平均回復 %.0f日",
                    len(crashes),
                    payload["major_crashes"]["avg_depth"] * 100,
                    payload["major_crashes"]["avg_recovery_days"])

    # 2. ブル/ベアサイクル
    if spy_arr is not None and len(spy_arr) > 252:
        cycles = analyze_bull_bear_cycles(spy_arr)
        payload["bull_bear_cycles"] = cycles
        logger.info("  🔄 サイクル: ブル%d回 (平均%.0f日) / ベア%d回 (平均%.0f日)",
                    cycles["n_bull"], cycles["avg_bull_duration_days"],
                    cycles["n_bear"], cycles["avg_bear_duration_days"])

    # 3. VIX レジーム
    if vix_arr is not None and len(vix_arr) > 100:
        vix_stats = analyze_vix_regimes(vix_arr)
        payload["vix_regimes"] = vix_stats
        logger.info("  📊 VIX 中央値 %.1f / 95%%値 %.1f / 現在 %d パーセンタイル",
                    vix_stats["vix_percentiles"]["p50"],
                    vix_stats["vix_percentiles"]["p95"],
                    vix_stats["current_vix_percentile"])

    # 4. Buffett contrarian
    if spy_arr is not None and vix_arr is not None:
        buffett = validate_buffett_contrarian(spy_arr, vix_arr)
        payload["buffett_contrarian_validation"] = buffett
        if "by_horizon" in buffett and "252d" in buffett.get("by_horizon", {}):
            yr = buffett["by_horizon"]["252d"]
            logger.info("  🎯 Buffett逆張り: VIX>30で買えば1年後 平均%+.1f%% (勝率%.0f%%)",
                        yr["mean_return"] * 100, yr["win_rate"] * 100)

    # 5. 個別銘柄サマリー
    ticker_summary = {}
    for t, p in ticker_prices.items():
        if len(p) < 252:
            continue
        years = len(p) / 252
        log_ret = np.diff(np.log(np.maximum(p, 1e-9)))
        ann_ret = float(np.mean(log_ret) * 252)
        ann_vol = float(np.std(log_ret) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + np.exp(log_ret) - 1)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - peak) / peak))
        ticker_summary[t] = {
            "history_years": round(years, 1),
            "annual_return": round(ann_ret, 4),
            "annual_volatility": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
        }
    payload["ticker_long_term_stats"] = ticker_summary

    # 永続化
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("✅ パターン抽出完了: %s", PATTERNS_PATH)
    return payload


def load_patterns() -> dict[str, Any]:
    """既存のパターン辞書を読み込む。"""
    if not PATTERNS_PATH.exists():
        return {}
    try:
        with open(PATTERNS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    """CLI: S&P500 + VIX + 個別銘柄から30年パターン抽出。"""
    parser = argparse.ArgumentParser(description="30年歴史パターン抽出")
    args = parser.parse_args()

    import yfinance as yf
    from daily_evolution import load_config

    config = load_config()
    holdings = config.get("portfolio", {}).get("holdings", [])
    tickers = [h["ticker"] for h in holdings]

    logger.info("📥 履歴データ取得 (max period)...")
    spy = yf.download("^GSPC", period="max", progress=False)
    vix = yf.download("^VIX", period="max", progress=False)

    spy_arr = None
    if not spy.empty:
        s = spy["Close"]
        if hasattr(s, "columns"):
            s = s.iloc[:, 0]
        spy_arr = s.dropna().values
        logger.info("  S&P500: %.1f年", len(spy_arr) / 252)

    vix_arr = None
    if not vix.empty:
        v = vix["Close"]
        if hasattr(v, "columns"):
            v = v.iloc[:, 0]
        vix_arr = v.dropna().values
        logger.info("  VIX: %.1f年", len(vix_arr) / 252)

    ticker_data = yf.download(tickers, period="max", progress=False)
    ticker_close = ticker_data["Close"] if "Close" in ticker_data else ticker_data
    ticker_prices = {}
    for t in tickers:
        if t in ticker_close.columns:
            arr = ticker_close[t].dropna().values
            if len(arr) >= 200:
                ticker_prices[t] = arr

    extract_all_patterns(spy_arr, vix_arr, ticker_prices)


if __name__ == "__main__":
    main()
