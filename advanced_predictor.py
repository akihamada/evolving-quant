#!/usr/bin/env python3
"""
advanced_predictor.py — 高精度マーケット予測アンサンブル

5つの独立予測器をベイズスタッキングで統合:
  1. Kalman Trend       : ノイズ除去した真のトレンド
  2. Hurst Regime       : トレンド/平均回帰の支配判定
  3. Cross-Sectional    : 銘柄間の相対モメンタム順位
  4. Volatility Regime  : ボラのクラスタリング(GARCH風)
  5. Mean-Reversion z   : ロバストz-score(MAD)

各予測器は -1.0(強売) ~ +1.0(強買) のスコアを返し、
過去精度に応じた事後ウェイトで統合する。

使用:
  from advanced_predictor import predict_all
  signals = predict_all(prices_dict)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("advanced-predictor")

SCRIPT_DIR = Path(__file__).resolve().parent
ACCURACY_PATH = SCRIPT_DIR / "data" / "advanced_predictor_accuracy.json"

# 各サブ予測器の事前ウェイト（同等を仮定）
PRIOR_WEIGHTS: dict[str, float] = {
    "kalman_trend":    1.0,
    "hurst_regime":    1.0,
    "cross_sectional": 1.0,
    "vol_regime":      1.0,
    "mean_reversion":  1.0,
}

# シグナル発生閾値（合成スコアの絶対値）
STRONG_SIGNAL_THRESHOLD = 0.45
WEAK_SIGNAL_THRESHOLD = 0.20


# ==============================================================================
# 1. Kalman Filter — ノイズ除去トレンド抽出
# ==============================================================================


def kalman_trend(prices: np.ndarray, q: float = 0.001, r: float = 0.05) -> dict[str, float]:
    """1次元Kalmanフィルタで価格の真のトレンドを推定する。

    状態モデル: x_t = x_{t-1} + w_t (w_t ~ N(0, q))
    観測モデル: y_t = x_t + v_t (v_t ~ N(0, r))

    Args:
        prices: 日次終値配列
        q: プロセスノイズ分散（小さいほどスムージング強）
        r: 観測ノイズ分散（大きいほど価格を信用しない）

    Returns:
        {"score": float, "filtered": float, "slope": float, "snr": float}
    """
    n = len(prices)
    if n < 30:
        return {"score": 0.0, "filtered": float(prices[-1]) if n else 0.0,
                "slope": 0.0, "snr": 0.0}

    log_p = np.log(np.maximum(prices, 1e-9))

    # 初期化
    x_hat = log_p[0]
    p_var = 1.0
    filtered = np.empty(n)

    for t in range(n):
        # 予測ステップ
        p_var = p_var + q
        # 更新ステップ
        k_gain = p_var / (p_var + r)
        x_hat = x_hat + k_gain * (log_p[t] - x_hat)
        p_var = (1 - k_gain) * p_var
        filtered[t] = x_hat

    # 直近20日のフィルタ済みトレンド傾き
    window = min(20, n - 1)
    slope = float((filtered[-1] - filtered[-window - 1]) / window)

    # SNR: フィルタ済み変動 / 観測ノイズ
    residuals = log_p - filtered
    noise_var = float(np.var(residuals[-60:])) if n >= 60 else float(np.var(residuals))
    signal_var = float(np.var(np.diff(filtered[-60:]))) if n >= 60 else 0.0
    snr = signal_var / (noise_var + 1e-10)

    # スコア化: 傾き * SNR^0.5 を tanh で [-1,1] に圧縮
    raw = slope * np.sqrt(max(snr, 0.0)) * 200
    score = float(np.tanh(raw))

    return {
        "score": score,
        "filtered": float(np.exp(filtered[-1])),
        "slope": slope,
        "snr": float(snr),
    }


# ==============================================================================
# 2. Hurst Exponent — トレンド/平均回帰レジーム判定
# ==============================================================================


def hurst_exponent(prices: np.ndarray, max_lag: int = 50) -> dict[str, float]:
    """Hurst指数を Rescaled Range 法で算出する。

    H > 0.55: 強いトレンド持続性 → モメンタム戦略
    H ≈ 0.50: ランダムウォーク
    H < 0.45: 平均回帰傾向 → 逆張り戦略

    Args:
        prices: 日次終値配列
        max_lag: 最大ラグ

    Returns:
        {"score": float, "hurst": float, "regime": str}
    """
    n = len(prices)
    if n < max_lag * 2:
        return {"score": 0.0, "hurst": 0.5, "regime": "unknown"}

    log_returns = np.diff(np.log(np.maximum(prices, 1e-9)))
    log_returns = log_returns[-max_lag * 4:] if len(log_returns) > max_lag * 4 else log_returns

    lags = np.arange(2, min(max_lag, len(log_returns) // 2))
    rs_values = []

    for lag in lags:
        # ラグ毎にデータを分割
        n_chunks = len(log_returns) // lag
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = log_returns[i * lag:(i + 1) * lag]
            mean_chunk = np.mean(chunk)
            adjusted = chunk - mean_chunk
            cumdev = np.cumsum(adjusted)
            r_val = float(np.max(cumdev) - np.min(cumdev))
            s_val = float(np.std(chunk))
            if s_val > 0:
                rs_chunk.append(r_val / s_val)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
        else:
            rs_values.append(np.nan)

    valid = ~np.isnan(rs_values)
    if valid.sum() < 4:
        return {"score": 0.0, "hurst": 0.5, "regime": "unknown"}

    log_lags = np.log(lags[valid])
    log_rs = np.log(np.array(rs_values)[valid])

    # 最小二乗法で傾き(=Hurst)を推定
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    h = float(np.clip(slope, 0.0, 1.0))

    # 直近20日のリターン方向と組み合わせてスコア化
    recent_ret = float(np.sum(log_returns[-20:])) if len(log_returns) >= 20 else 0.0
    recent_dir = np.sign(recent_ret)

    if h > 0.55:
        regime = "trending"
        # トレンド継続を予測 → 直近方向を増幅
        score = float(recent_dir * min(1.0, (h - 0.5) * 4))
    elif h < 0.45:
        regime = "mean_reverting"
        # 平均回帰を予測 → 直近方向の逆を予測
        score = float(-recent_dir * min(1.0, (0.5 - h) * 4))
    else:
        regime = "random_walk"
        score = 0.0

    return {"score": score, "hurst": h, "regime": regime}


# ==============================================================================
# 3. Cross-Sectional Momentum — 相対順位
# ==============================================================================


def cross_sectional_momentum(
    prices_dict: dict[str, np.ndarray], lookback: int = 63
) -> dict[str, float]:
    """銘柄群の相対モメンタム順位を算出する。

    全銘柄のlookback期間リターンを順位化し、
    上位/下位は買い/売り候補とする。

    Args:
        prices_dict: {ticker: 価格配列}
        lookback: モメンタム計算期間（日）

    Returns:
        {ticker: score} の辞書 (-1.0~+1.0)
    """
    returns = {}
    for ticker, prices in prices_dict.items():
        if len(prices) < lookback + 1:
            continue
        # 直近1ヶ月を除外したlookback期間リターン (12-1モメンタム)
        skip = min(21, lookback // 4)
        end_idx = -skip if skip > 0 else None
        start_idx = -lookback - skip
        try:
            ret = float(prices[end_idx] / prices[start_idx] - 1) if end_idx else \
                  float(prices[-1] / prices[start_idx] - 1)
        except (IndexError, ZeroDivisionError):
            continue
        returns[ticker] = ret

    if len(returns) < 3:
        return {t: 0.0 for t in prices_dict}

    # 順位を [-1, 1] に正規化
    sorted_tickers = sorted(returns.items(), key=lambda x: x[1])
    n = len(sorted_tickers)
    scores = {}
    for rank, (ticker, _) in enumerate(sorted_tickers):
        # 線形マッピング: 最下位=-1, 最上位=+1
        scores[ticker] = float(2.0 * rank / (n - 1) - 1.0) if n > 1 else 0.0

    # 含まれない銘柄は0
    for t in prices_dict:
        if t not in scores:
            scores[t] = 0.0

    return scores


# ==============================================================================
# 4. Volatility Regime — GARCH風クラスタリング
# ==============================================================================


def volatility_regime(prices: np.ndarray) -> dict[str, float]:
    """ボラティリティ・レジームを検出し、リスクプレミアムを評価する。

    高ボラ局面 → リターン期待値を圧縮（防御的シグナル）
    低ボラ局面 → リターン期待値を増幅（積極的シグナル）

    Args:
        prices: 日次終値配列

    Returns:
        {"score": float, "current_vol": float, "vol_percentile": float,
         "regime": str}
    """
    n = len(prices)
    if n < 60:
        return {"score": 0.0, "current_vol": 0.0,
                "vol_percentile": 0.5, "regime": "unknown"}

    log_returns = np.diff(np.log(np.maximum(prices, 1e-9)))

    # ローリング21日ボラ
    rolling_vol = []
    for i in range(20, len(log_returns)):
        rolling_vol.append(float(np.std(log_returns[i - 20:i + 1])))
    rolling_vol = np.array(rolling_vol)

    if len(rolling_vol) < 30:
        return {"score": 0.0, "current_vol": 0.0,
                "vol_percentile": 0.5, "regime": "unknown"}

    current_vol = float(rolling_vol[-1])
    vol_percentile = float((rolling_vol < current_vol).mean())

    # 直近5日のリターン傾向
    recent_ret = float(np.sum(log_returns[-5:]))

    # 高ボラ + 下落 → 強い売り
    # 低ボラ + 上昇 → 強い買い
    # 高ボラ + 上昇 → 中立(リバウンドの可能性)
    # 低ボラ + 下落 → 弱い売り
    if vol_percentile > 0.80:
        regime = "high_vol"
        # 高ボラ時は方向を信用せず弱める
        score = float(np.tanh(recent_ret * 5)) * 0.3
    elif vol_percentile < 0.30:
        regime = "low_vol"
        # 低ボラ時は方向を増幅
        score = float(np.tanh(recent_ret * 30))
    else:
        regime = "normal"
        score = float(np.tanh(recent_ret * 15)) * 0.6

    return {
        "score": score,
        "current_vol": current_vol,
        "vol_percentile": vol_percentile,
        "regime": regime,
    }


# ==============================================================================
# 5. Mean Reversion — ロバストz-score
# ==============================================================================


def mean_reversion_score(prices: np.ndarray, window: int = 60) -> dict[str, float]:
    """MAD(median absolute deviation)ベースのロバストz-scoreを算出する。

    通常のz-scoreよりも外れ値に頑健。
    z >> 0 → 過熱(売り) / z << 0 → 売られ過ぎ(買い)

    Args:
        prices: 日次終値配列
        window: 中央値計算ウィンドウ

    Returns:
        {"score": float, "z_robust": float}
    """
    n = len(prices)
    if n < window:
        return {"score": 0.0, "z_robust": 0.0}

    recent = prices[-window:]
    median = float(np.median(recent))
    mad = float(np.median(np.abs(recent - median)))

    if mad < 1e-9:
        return {"score": 0.0, "z_robust": 0.0}

    # 1.4826 は正規分布での MAD → 標準偏差変換係数
    z = (float(prices[-1]) - median) / (1.4826 * mad)

    # 過熱なら売り(負)、売られすぎなら買い(正)
    # z=±2 で score=±0.76 程度
    score = float(-np.tanh(z / 2.5))

    return {"score": score, "z_robust": float(z)}


# ==============================================================================
# 6. ベイズ・スタッキング統合
# ==============================================================================


def load_predictor_weights() -> dict[str, float]:
    """過去の予測精度から各サブ予測器の事後ウェイトを読み込む。

    Returns:
        {predictor_name: weight} の辞書
    """
    if not ACCURACY_PATH.exists():
        return PRIOR_WEIGHTS.copy()

    try:
        with open(ACCURACY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return PRIOR_WEIGHTS.copy()

    weights = {}
    for name, prior in PRIOR_WEIGHTS.items():
        stats = data.get("predictor_stats", {}).get(name, {})
        accuracy = stats.get("accuracy", 0.5)
        n_samples = stats.get("evaluated", 0)
        # ベイズ更新: 精度50%基準で対称的に増減 / サンプル数で重み付け
        confidence_weight = min(1.0, n_samples / 50)
        adjustment = (accuracy - 0.5) * 2 * confidence_weight
        weights[name] = max(0.1, prior * (1 + adjustment))

    # 合計を1に正規化
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def predict_ticker(
    ticker: str,
    prices: np.ndarray,
    cs_score: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    """単一銘柄の統合予測を生成する。

    Args:
        ticker: ティッカー
        prices: 価格配列
        cs_score: クロスセクショナルスコア
        weights: 予測器ウェイト

    Returns:
        予測辞書
    """
    kt = kalman_trend(prices)
    hr = hurst_exponent(prices)
    vr = volatility_regime(prices)
    mr = mean_reversion_score(prices)

    sub_scores = {
        "kalman_trend":    kt["score"],
        "hurst_regime":    hr["score"],
        "cross_sectional": cs_score,
        "vol_regime":      vr["score"],
        "mean_reversion":  mr["score"],
    }

    # 重み付き合成スコア
    composite = sum(weights.get(k, 0) * v for k, v in sub_scores.items())
    composite = float(np.clip(composite, -1.0, 1.0))

    # シグナル判定
    abs_c = abs(composite)
    if abs_c >= STRONG_SIGNAL_THRESHOLD:
        signal = "STRONG_BUY" if composite > 0 else "STRONG_SELL"
        confidence = min(1.0, 0.7 + (abs_c - STRONG_SIGNAL_THRESHOLD) * 0.6)
    elif abs_c >= WEAK_SIGNAL_THRESHOLD:
        signal = "BUY" if composite > 0 else "SELL"
        confidence = 0.5 + (abs_c - WEAK_SIGNAL_THRESHOLD) * 0.8
    else:
        signal = "HOLD"
        confidence = 0.3 + abs_c * 0.5

    # 予測リターン（合成スコアを年率10%スケールにマッピング）
    expected_return_7d = composite * 0.025  # 年20%相当の最大値

    return {
        "ticker": ticker,
        "signal": signal,
        "composite_score": round(composite, 4),
        "confidence": round(confidence, 3),
        "expected_return_7d": round(expected_return_7d, 5),
        "sub_scores": {k: round(v, 4) for k, v in sub_scores.items()},
        "components": {
            "kalman": {
                "slope": round(kt["slope"], 5),
                "snr": round(kt["snr"], 3),
            },
            "hurst": {
                "value": round(hr["hurst"], 3),
                "regime": hr["regime"],
            },
            "volatility": {
                "current": round(vr["current_vol"], 5),
                "percentile": round(vr["vol_percentile"], 3),
                "regime": vr["regime"],
            },
            "mean_reversion": {
                "z_robust": round(mr["z_robust"], 3),
            },
        },
    }


def predict_all(
    prices_dict: dict[str, np.ndarray],
    weights: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """全銘柄の統合予測を生成する。

    Args:
        prices_dict: {ticker: 価格配列}
        weights: 予測器ウェイト（Noneなら自動読み込み）

    Returns:
        予測辞書のリスト（confidence降順）
    """
    if weights is None:
        weights = load_predictor_weights()

    cs_scores = cross_sectional_momentum(prices_dict)
    today = datetime.now().strftime("%Y-%m-%d")

    predictions = []
    for ticker, prices in prices_dict.items():
        if len(prices) < 60:
            continue
        try:
            pred = predict_ticker(ticker, prices, cs_scores.get(ticker, 0.0), weights)
            pred["id"] = f"adv-{today}-{ticker}"
            pred["date"] = today
            pred["price_at_prediction"] = round(float(prices[-1]), 2)
            pred["evaluated"] = False
            predictions.append(pred)
        except Exception as e:
            logger.warning("予測失敗 %s: %s", ticker, e)

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    n_strong = sum(1 for p in predictions if "STRONG" in p["signal"])
    n_buy = sum(1 for p in predictions if p["signal"] in ("BUY", "STRONG_BUY"))
    n_sell = sum(1 for p in predictions if p["signal"] in ("SELL", "STRONG_SELL"))
    logger.info(
        "🔮 高精度予測: %d銘柄 (BUY=%d, SELL=%d, STRONG=%d)",
        len(predictions), n_buy, n_sell, n_strong,
    )

    return predictions


# ==============================================================================
# 7. 自己評価エンジン
# ==============================================================================


def evaluate_past_predictions(
    past_predictions: list[dict[str, Any]],
    prices_dict: dict[str, np.ndarray],
    eval_days: int = 7,
) -> dict[str, Any]:
    """過去予測の方向精度を評価し、サブ予測器ごとの統計を更新する。

    Args:
        past_predictions: 過去の予測リスト
        prices_dict: 現在の価格データ
        eval_days: 評価対象とする経過日数の最低値

    Returns:
        {"predictor_stats": {name: {...}}, "evaluated_count": int}
    """
    today = datetime.now()
    predictor_results: dict[str, dict[str, int]] = {
        name: {"correct": 0, "evaluated": 0} for name in PRIOR_WEIGHTS
    }

    for pred in past_predictions:
        if pred.get("evaluated"):
            # 既に評価済の結果を統計へ反映
            for name, score in pred.get("sub_scores", {}).items():
                if name not in predictor_results:
                    continue
                actual_dir = pred.get("actual_direction", 0)
                if actual_dir == 0:
                    continue
                pred_dir = np.sign(score)
                if pred_dir == 0:
                    continue
                predictor_results[name]["evaluated"] += 1
                if pred_dir == actual_dir:
                    predictor_results[name]["correct"] += 1
            continue

        try:
            pred_date = datetime.strptime(pred["date"], "%Y-%m-%d")
        except (ValueError, KeyError, TypeError):
            continue

        days_elapsed = (today - pred_date).days
        if days_elapsed < eval_days + 2:
            continue

        ticker = pred.get("ticker", "")
        prices = prices_dict.get(ticker)
        pred_price = pred.get("price_at_prediction", 0)
        if prices is None or len(prices) < 2 or pred_price <= 0:
            continue

        actual_return = (float(prices[-1]) - pred_price) / pred_price
        actual_dir = float(np.sign(actual_return))

        pred["evaluated"] = True
        pred["actual_return"] = round(actual_return, 5)
        pred["actual_direction"] = actual_dir
        pred["evaluation_date"] = today.strftime("%Y-%m-%d")

        # 各サブ予測器の方向精度をカウント
        for name, score in pred.get("sub_scores", {}).items():
            if name not in predictor_results:
                continue
            pred_dir = np.sign(score)
            if pred_dir == 0:
                continue
            predictor_results[name]["evaluated"] += 1
            if pred_dir == actual_dir:
                predictor_results[name]["correct"] += 1

    stats = {}
    for name, r in predictor_results.items():
        ev = r["evaluated"]
        stats[name] = {
            "evaluated": ev,
            "correct": r["correct"],
            "accuracy": round(r["correct"] / ev, 4) if ev > 0 else 0.5,
        }

    return {"predictor_stats": stats}


def save_accuracy(stats: dict[str, Any]) -> None:
    """精度統計を永続化する。"""
    ACCURACY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(),
        "predictor_stats": stats.get("predictor_stats", {}),
    }
    with open(ACCURACY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 自己テスト: ランダムウォーク + トレンドで動作確認
    np.random.seed(42)
    n = 252
    base = np.cumsum(np.random.randn(n) * 0.01) + 0.0005 * np.arange(n)
    prices_test = {
        "TREND_UP":   100 * np.exp(base + 0.001 * np.arange(n)),
        "TREND_DOWN": 100 * np.exp(base - 0.0008 * np.arange(n)),
        "FLAT":       100 * np.exp(np.cumsum(np.random.randn(n) * 0.005)),
        "VOLATILE":   100 * np.exp(np.cumsum(np.random.randn(n) * 0.03)),
    }
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    preds = predict_all(prices_test)
    for p in preds:
        print(f"  {p['ticker']:10s} {p['signal']:12s} "
              f"score={p['composite_score']:+.3f} conf={p['confidence']:.2f}")
