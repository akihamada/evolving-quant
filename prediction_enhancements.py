#!/usr/bin/env python3
"""
prediction_enhancements.py — 予測力を世界一クラスに引き上げる強化モジュール

【6つの強化レイヤー】
  1. Isotonic Calibration       : 信頼度を真の確率に較正
  2. Online EMA Weight Update   : 即時適応の指数移動平均ウェイト
  3. Walk-Forward Sharpe        : "本当に儲かるか"を統計検証
  4. Stacking Meta-Learner      : 9ファクター上の非線形メタ学習
  5. Transfer Learning          : セクター類似銘柄でファクター共有
  6. Anomaly Detection          : 黒い白鳥イベント自動検知

依存: numpy のみ（外部MLライブラリ不要）
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("enhancements")

SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "data" / "calibration_table.json"
WALK_FORWARD_PATH = SCRIPT_DIR / "data" / "walk_forward_results.json"
META_LEARNER_PATH = SCRIPT_DIR / "data" / "meta_learner_weights.json"
ANOMALY_LOG_PATH = SCRIPT_DIR / "data" / "anomaly_log.json"

# ==============================================================================
# 1. Isotonic Regression Calibration
# ==============================================================================


def isotonic_regression(
    predictions: list[float],
    actuals: list[float],
    n_bins: int = 10,
) -> list[tuple[float, float]]:
    """単純な isotonic regression による信頼度較正。

    予測確信度を [0,1] のbinに分け、各binでの実勝率を計算し、
    monotonic 制約をかけてマッピングテーブルを生成。

    Args:
        predictions: 予測時の確信度リスト [0, 1]
        actuals: 実際の方向一致 (1 if correct else 0)
        n_bins: bin数

    Returns:
        [(bin_upper_bound, calibrated_prob), ...]
    """
    if not predictions or not actuals:
        return [(1.0, 0.5)]
    if len(predictions) != len(actuals):
        return [(1.0, 0.5)]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_means: list[tuple[float, float]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(p >= lo) and (p < hi if hi < 1 else p <= hi) for p in predictions]
        if not any(mask):
            continue
        actuals_in_bin = [a for a, m in zip(actuals, mask) if m]
        if not actuals_in_bin:
            continue
        avg_actual = float(np.mean(actuals_in_bin))
        bin_means.append((float(hi), avg_actual))

    # PAV (Pool Adjacent Violators) で monotonic に強制
    if len(bin_means) >= 2:
        ys = [v for _, v in bin_means]
        # 単純平均化アプローチ
        for _ in range(20):
            changed = False
            for i in range(len(ys) - 1):
                if ys[i] > ys[i + 1]:
                    avg = (ys[i] + ys[i + 1]) / 2
                    ys[i] = ys[i + 1] = avg
                    changed = True
            if not changed:
                break
        bin_means = [(b, ys[i]) for i, (b, _) in enumerate(bin_means)]

    return bin_means or [(1.0, 0.5)]


def calibrate_confidence(raw_confidence: float, calibration_table: list[tuple[float, float]]) -> float:
    """較正テーブルを使って raw confidence を真の確率に変換。"""
    if not calibration_table:
        return raw_confidence
    for upper, calibrated in calibration_table:
        if raw_confidence <= upper:
            return float(calibrated)
    return float(calibration_table[-1][1])


def build_calibration_from_history(history: dict[str, Any]) -> list[tuple[float, float]]:
    """master_prediction_history から較正テーブルを構築。

    Args:
        history: load_master_history() の出力

    Returns:
        較正テーブル
    """
    predictions = []
    actuals = []
    for p in history.get("predictions", []):
        if not p.get("evaluated"):
            continue
        conf = p.get("confidence", 0)
        composite = p.get("composite_score", 0)
        actual = p.get("actual_returns", {}).get("7d")
        if actual is None:
            continue
        pred_dir = float(np.sign(composite))
        actual_dir = float(np.sign(actual))
        if pred_dir == 0 or actual_dir == 0:
            continue
        is_correct = 1 if pred_dir == actual_dir else 0
        predictions.append(conf)
        actuals.append(is_correct)

    if len(predictions) < 20:
        logger.info("較正データ不足 (%d件) — スキップ", len(predictions))
        return []

    table = isotonic_regression(predictions, actuals)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "table": table,
            "n_samples": len(predictions),
            "updated_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    logger.info("✅ 較正テーブル更新: %d 件、%d bins", len(predictions), len(table))
    return table


def load_calibration_table() -> list[tuple[float, float]]:
    """較正テーブルを読み込む。"""
    if not CALIBRATION_PATH.exists():
        return []
    try:
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return [tuple(t) for t in data.get("table", [])]
    except (json.JSONDecodeError, OSError):
        return []


# ==============================================================================
# 2. Online EMA Weight Updates
# ==============================================================================


def ema_update_weights(
    prev_weights: dict[str, float],
    new_accuracies: dict[str, float],
    alpha: float = 0.1,
    prior_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """指数移動平均でウェイトを更新する。

    通常の Bayesian バッチ更新より敏感に直近変化を捕捉。

    Args:
        prev_weights: 前回のウェイト
        new_accuracies: {factor: accuracy} 直近精度
        alpha: 学習率 (0.1 = 約10日半減期)
        prior_weights: アンカー（重みが極端に振れるのを防ぐ）

    Returns:
        更新されたウェイト
    """
    if not prior_weights:
        prior_weights = prev_weights

    new_w = {}
    for name, prev_w in prev_weights.items():
        acc = new_accuracies.get(name, 0.5)
        prior = prior_weights.get(name, prev_w)
        # 精度50%基準で正負へ動かす
        target = prior * (1 + (acc - 0.5) * 2)
        target = max(0.02, min(0.40, target))
        # EMA: new = α * target + (1-α) * prev
        new_w[name] = float(alpha * target + (1 - alpha) * prev_w)

    # 正規化
    total = sum(new_w.values())
    if total > 0:
        new_w = {k: v / total for k, v in new_w.items()}
    return new_w


# ==============================================================================
# 3. Walk-Forward Sharpe Validation
# ==============================================================================


def walk_forward_sharpe(
    predictions_history: list[dict[str, Any]],
    prices_dict: dict[str, np.ndarray],
    holding_days: int = 30,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """予測の確信度に応じてポジションを取り、リターンを計算する。

    Args:
        predictions_history: 過去の予測リスト
        prices_dict: {ticker: 価格配列}
        holding_days: ホールド期間
        min_confidence: アクション最低確信度

    Returns:
        {"sharpe", "annual_return", "max_dd", "win_rate", "n_trades"}
    """
    trades = []
    for p in predictions_history:
        if not p.get("evaluated"):
            continue
        conf = p.get("confidence", 0)
        if conf < min_confidence:
            continue
        signal = p.get("signal", "HOLD")
        if signal == "HOLD":
            continue
        actual_30 = p.get("actual_returns", {}).get(f"{holding_days}d")
        if actual_30 is None:
            actual_30 = p.get("actual_returns", {}).get("7d")
        if actual_30 is None:
            continue
        # 売り側はリターンを反転
        sign = 1 if "BUY" in signal else -1
        trade_return = sign * actual_30
        trades.append({
            "date": p.get("date", ""),
            "ticker": p.get("ticker", ""),
            "signal": signal,
            "confidence": conf,
            "return": trade_return,
        })

    if len(trades) < 10:
        return {
            "n_trades": len(trades),
            "sharpe": 0.0,
            "annual_return": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "note": "サンプル不足",
        }

    returns = np.array([t["return"] for t in trades])
    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) + 1e-10
    # 年率換算 (1ホールディング ≒ 30日 → 12回/年)
    periods_per_year = 252 / holding_days
    annual_return = float(avg_ret * periods_per_year)
    sharpe = float((avg_ret / std_ret) * np.sqrt(periods_per_year))
    win_rate = float(np.sum(returns > 0) / len(returns))

    # Max DD（仮想累積資産曲線）
    cum_growth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_growth)
    dd_series = (cum_growth - peak) / peak
    max_dd = float(np.min(dd_series))

    result = {
        "n_trades": len(trades),
        "annual_return": round(annual_return, 4),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "avg_return_per_trade": round(avg_ret, 5),
        "executed_at": datetime.now().isoformat(),
    }

    WALK_FORWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WALK_FORWARD_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(
        "🎯 Walk-Forward: Sharpe=%.2f / 年率=%.1f%% / 勝率=%.0f%% / DD=%.1f%% / N=%d",
        sharpe, annual_return * 100, win_rate * 100, max_dd * 100, len(trades)
    )
    return result


# ==============================================================================
# 4. Stacking Meta-Learner (Logistic Regression on factor scores)
# ==============================================================================


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def train_stacking_meta(history: dict[str, Any], lr: float = 0.05, epochs: int = 200) -> dict[str, float]:
    """9ファクター score を入力として方向予測する logistic regression を訓練。

    Args:
        history: master_prediction_history
        lr: 学習率
        epochs: エポック数

    Returns:
        {"weights": {factor: w, ..., "bias": b}, "n_samples": int, "accuracy": float}
    """
    from master_predictor import FACTOR_NAMES

    X = []
    y = []
    for p in history.get("predictions", []):
        if not p.get("evaluated"):
            continue
        actual = p.get("actual_returns", {}).get("7d")
        if actual is None:
            continue
        actual_dir = 1.0 if actual > 0 else 0.0
        fs = p.get("factor_scores", {})
        row = []
        for name in FACTOR_NAMES:
            f_data = fs.get(name, {})
            score = f_data.get("score", 0) if isinstance(f_data, dict) else f_data
            row.append(float(score))
        X.append(row)
        y.append(actual_dir)

    if len(X) < 30:
        return {"weights": {}, "bias": 0.0, "n_samples": len(X),
                "accuracy": 0.5, "note": "サンプル不足"}

    X_arr = np.array(X)
    y_arr = np.array(y)
    n_features = X_arr.shape[1]

    # SGD logistic regression
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        z = X_arr @ w + b
        pred = 1.0 / (1.0 + np.exp(-z))
        err = pred - y_arr
        grad_w = X_arr.T @ err / len(X_arr)
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

    # 学習後の精度
    z = X_arr @ w + b
    pred_label = (z > 0).astype(float)
    accuracy = float(np.mean(pred_label == y_arr))

    weights = {FACTOR_NAMES[i]: float(round(w[i], 4)) for i in range(n_features)}
    weights["bias"] = float(round(b, 4))
    payload = {
        "weights": weights,
        "n_samples": len(X),
        "accuracy": round(accuracy, 4),
        "trained_at": datetime.now().isoformat(),
    }

    META_LEARNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_LEARNER_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "🤖 Meta-Learner 訓練: 精度=%.1f%% / N=%d", accuracy * 100, len(X)
    )
    return payload


def predict_with_meta(factor_scores: dict[str, float], meta: dict[str, Any]) -> float:
    """メタ学習器で factor_scores から方向確率を予測。

    Returns:
        確率 [0, 1] (1 = 強い買い)
    """
    weights = meta.get("weights", {})
    if not weights:
        return 0.5
    bias = weights.get("bias", 0.0)
    z = bias
    for name, score in factor_scores.items():
        z += weights.get(name, 0) * float(score)
    return float(_sigmoid(z))


def load_meta_learner() -> dict[str, Any]:
    """メタ学習器ウェイトを読み込む。"""
    if not META_LEARNER_PATH.exists():
        return {"weights": {}, "bias": 0.0, "n_samples": 0}
    try:
        with open(META_LEARNER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"weights": {}, "bias": 0.0, "n_samples": 0}


# ==============================================================================
# 5. Transfer Learning (Sector grouping)
# ==============================================================================

# セクターグループ定義
SECTOR_GROUPS = {
    "AI_Semi": ["NVDA", "TSM", "ARM", "MRVL", "AVGO", "AMD"],
    "Software": ["MSFT", "PLTR", "DDOG", "CRWD", "SNOW"],
    "Power": ["VRT", "ETN", "GLW"],
    "Nuclear": ["NNE", "OKLO", "VST"],
    "Pharma": ["LLY", "JNJ", "MRK"],
    "Photonics": ["COHR"],
    "Dividend_ETF": ["VYM", "1489.T"],
    "Gold": ["GLD", "1328.T"],
}


def get_sector_group(ticker: str) -> Optional[str]:
    """ティッカーからセクターグループを取得。"""
    for group, tickers in SECTOR_GROUPS.items():
        if ticker in tickers:
            return group
    return None


def transfer_learning_boost(
    ticker: str,
    history: dict[str, Any],
    own_factor_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """同セクター銘柄の精度を借りてサンプル数を増強する。

    Args:
        ticker: 対象銘柄
        history: 全予測履歴
        own_factor_stats: 対象銘柄の factor_stats

    Returns:
        セクター混合済みの factor_stats
    """
    group = get_sector_group(ticker)
    if not group:
        return own_factor_stats

    sector_tickers = SECTOR_GROUPS.get(group, [])
    sector_predictions = [
        p for p in history.get("predictions", [])
        if p.get("ticker", "") in sector_tickers and p.get("evaluated")
    ]

    if len(sector_predictions) < 10:
        return own_factor_stats

    from master_predictor import FACTOR_NAMES

    # セクター集計
    enhanced: dict[str, dict[str, Any]] = {}
    for name in FACTOR_NAMES:
        own_acc = own_factor_stats.get(name, {}).get("accuracy", 0.5)
        own_n = own_factor_stats.get(name, {}).get("samples", 0)

        sector_hits = 0
        sector_trials = 0
        for p in sector_predictions:
            fs = p.get("factor_scores", {})
            f_data = fs.get(name, {})
            score = f_data.get("score", 0) if isinstance(f_data, dict) else f_data
            actual = p.get("actual_returns", {}).get("7d")
            if actual is None or score == 0:
                continue
            sector_trials += 1
            if (score > 0 and actual > 0) or (score < 0 and actual < 0):
                sector_hits += 1

        sector_acc = sector_hits / sector_trials if sector_trials else 0.5
        # 自身データを優先しつつセクターで補強
        if own_n >= 20:
            blended_acc = own_acc * 0.7 + sector_acc * 0.3
        else:
            # 自身データが少ない場合はセクター比重UP
            blended_acc = own_acc * 0.4 + sector_acc * 0.6

        enhanced[name] = {
            "accuracy": round(blended_acc, 4),
            "own_accuracy": round(own_acc, 4),
            "sector_accuracy": round(sector_acc, 4),
            "samples": own_n + sector_trials,
            "decay_factor": own_factor_stats.get(name, {}).get("decay_factor", 1.0),
        }

    return enhanced


# ==============================================================================
# 6. Anomaly Detection (Black Swan)
# ==============================================================================


def detect_anomaly(
    macro_data: dict[str, Any],
    recent_returns: list[float],
) -> dict[str, Any]:
    """マクロ + リターン異常から黒い白鳥イベントを検知。

    Args:
        macro_data: {"vix": float, "market_return_30d": float}
        recent_returns: 直近5日のマーケットリターン

    Returns:
        {"is_anomaly": bool, "severity": float, "triggers": [...]}
    """
    triggers = []
    severity = 0.0

    vix = macro_data.get("vix")
    if vix is not None:
        if vix >= 40:
            triggers.append(f"VIX 極度高 ({vix:.0f})")
            severity += 0.4
        elif vix >= 30:
            triggers.append(f"VIX 高 ({vix:.0f})")
            severity += 0.2

    market_30d = macro_data.get("market_return_30d")
    if market_30d is not None and market_30d <= -0.10:
        triggers.append(f"30日 -{abs(market_30d)*100:.0f}% 急落")
        severity += 0.3

    # 直近5日で連続急落
    if recent_returns and len(recent_returns) >= 3:
        recent_arr = np.array(recent_returns[-5:])
        if np.sum(recent_arr) <= -0.07:
            triggers.append(f"直近5日 累計 {np.sum(recent_arr)*100:+.1f}%")
            severity += 0.2

    severity = min(1.0, severity)
    is_anomaly = severity >= 0.4

    if is_anomaly:
        result = {
            "is_anomaly": True,
            "severity": round(severity, 2),
            "triggers": triggers,
            "detected_at": datetime.now().isoformat(),
            "recommended_action": "crisis モードで Master Wisdom 再評価 / リスク資産縮小推奨",
        }
        # ログ追記
        ANOMALY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_data = {"events": []}
        if ANOMALY_LOG_PATH.exists():
            try:
                with open(ANOMALY_LOG_PATH, encoding="utf-8") as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        log_data.setdefault("events", []).append(result)
        log_data["events"] = log_data["events"][-100:]
        with open(ANOMALY_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.warning(
            "⚠️ 黒い白鳥検知: severity=%.1f / triggers=%s", severity, triggers
        )
    return {"is_anomaly": is_anomaly, "severity": round(severity, 2),
            "triggers": triggers}


def load_recent_anomalies(n: int = 5) -> list[dict[str, Any]]:
    """直近の異常イベントを読み込む。"""
    if not ANOMALY_LOG_PATH.exists():
        return []
    try:
        with open(ANOMALY_LOG_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("events", [])[-n:]
    except (json.JSONDecodeError, OSError):
        return []
