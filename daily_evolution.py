# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Daily Evolution Engine — 自己進化型クオンツ・バックグラウンドプロセス
====================================================================
launchd で毎日自動実行。Mac miniの電源がONなら勝手に進化し続ける。

処理フロー:
1. yfinance でマーケットデータ取得
2. 未評価の過去予測を自動採点
3. Inner Loop: 数学パラメータを進化
4. ai_track_record.json に保存

実行: python daily_evolution.py
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ==============================================================================
# ログ設定
# ==============================================================================

LOG_DIR = Path(__file__).resolve().parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "evolution_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daily-evolution")

# ==============================================================================
# 定数
# ==============================================================================

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
TRACK_RECORD_PATH = Path(__file__).resolve().parent / "ai_track_record.json"
DATA_PERIOD = "1y"
DATA_INTERVAL = "1d"
ENSEMBLE_LOOKBACK_DAYS = 60
REGIME_ROLLING_WINDOW = 21


# ==============================================================================
# 1. Config & Data Layer
# ==============================================================================


def load_config() -> dict:
    """
    config.json からポートフォリオ設定を読み込む。

    Returns:
        設定辞書

    Raises:
        FileNotFoundError: config.json が存在しない場合
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_price_data(tickers: list[str]) -> pd.DataFrame:
    """
    yfinance で複数銘柄の日次終値を取得する。

    Args:
        tickers: ティッカーシンボルのリスト

    Returns:
        日次終値のDataFrame (columns=tickers, index=DatetimeIndex)
    """
    logger.info("価格データ取得中: %s", tickers)
    data = yf.download(
        tickers,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    prices = prices.dropna(how="all").ffill().dropna()
    logger.info("取得完了: %d日 x %d銘柄", len(prices), len(prices.columns))
    return prices


def calc_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    対数リターンを計算する。

    Args:
        prices: 日次終値DataFrame

    Returns:
        対数リターンのDataFrame
    """
    return np.log(prices / prices.shift(1)).dropna()


# ==============================================================================
# 2. Denoised + Detoned Covariance (Random Matrix Theory)
# ==============================================================================


def marchenko_pastur_bound(n_samples: int, n_features: int) -> float:
    """
    Marchenko-Pastur分布の上界を計算する。

    Args:
        n_samples: サンプル数（日数）
        n_features: 特徴量数（銘柄数）

    Returns:
        MP上界値（この値以下の固有値はノイズ）
    """
    q = n_samples / n_features
    lambda_plus = (1 + 1 / np.sqrt(q)) ** 2
    return lambda_plus


def denoise_covariance(cov: np.ndarray, n_samples: int) -> np.ndarray:
    """
    ランダム行列理論でノイズを除去した共分散行列を返す。

    Marchenko-Pastur境界以下の固有値を平均値に縮退させる。

    Args:
        cov: 元の共分散行列
        n_samples: サンプル数

    Returns:
        デノイズ済み共分散行列
    """
    n_features = cov.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    mp_bound = marchenko_pastur_bound(n_samples, n_features)
    # 共分散行列を相関行列スケールに正規化して固有値を判定
    diag = np.sqrt(np.diag(cov))
    corr_eigenvalues = eigenvalues / (np.mean(diag) ** 2 + 1e-10)
    noise_mask = corr_eigenvalues < mp_bound
    if noise_mask.any():
        noise_mean = np.mean(eigenvalues[noise_mask])
        eigenvalues[noise_mask] = noise_mean
    denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # 対称性を保証
    denoised = (denoised + denoised.T) / 2
    return denoised


def detone_covariance(cov: np.ndarray, n_remove: int = 1) -> np.ndarray:
    """
    マーケットモード（最大固有値）を除去する（Detoning）。

    市場全体の動きを除去し、個別銘柄間の真の相関構造を抽出。

    Args:
        cov: 共分散行列
        n_remove: 除去する最大固有値の数

    Returns:
        デトーン済み共分散行列
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 最大のn_remove個の固有値をゼロに
    eigenvalues[-n_remove:] = 0
    detoned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    detoned = (detoned + detoned.T) / 2
    # 対角成分を元に戻す（分散は保持）
    np.fill_diagonal(detoned, np.diag(cov))
    return detoned


# ==============================================================================
# 3. NCO: Nested Clustered Optimization
# ==============================================================================


def _cluster_weights_mvo(cov_cluster: np.ndarray) -> np.ndarray:
    """
    クラスタ内でMinimum Variance配分を計算する（MVO簡易版）。

    Args:
        cov_cluster: クラスタ内の共分散行列

    Returns:
        クラスタ内の重みベクトル
    """
    n = cov_cluster.shape[0]
    if n == 1:
        return np.array([1.0])
    try:
        inv_cov = np.linalg.pinv(cov_cluster)
        ones = np.ones(n)
        w = inv_cov @ ones
        w = w / (np.sum(w) + 1e-10)
        w = np.clip(w, 0, None)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            w = np.ones(n) / n
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    return w


def nco_optimize(cov: np.ndarray, tickers: list[str]) -> dict[str, float]:
    """
    Nested Clustered Optimization: HRPのクラスタリング + クラスタ内MVO。

    Step 1: 相関距離で階層的クラスタリング
    Step 2: 各クラスタ内でMinVar配分
    Step 3: クラスタ間でリスクパリティ配分

    Args:
        cov: デノイズ済み共分散行列
        tickers: 銘柄名リスト

    Returns:
        {ticker: weight} の辞書
    """
    n = len(tickers)
    if n <= 1:
        return {tickers[0]: 1.0} if n == 1 else {}

    # 相関行列 → 距離行列
    std = np.sqrt(np.diag(cov))
    std_outer = np.outer(std, std)
    corr = cov / (std_outer + 1e-10)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)

    # 階層的クラスタリング
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")

    # クラスタ分割（2分割）
    sorted_indices = leaves_list(link).tolist()
    mid = len(sorted_indices) // 2
    cluster_a = sorted_indices[:mid]
    cluster_b = sorted_indices[mid:]

    # クラスタ内MVO
    w_a = _cluster_weights_mvo(cov[np.ix_(cluster_a, cluster_a)])
    w_b = _cluster_weights_mvo(cov[np.ix_(cluster_b, cluster_b)])

    # クラスタ間リスクパリティ
    var_a = w_a @ cov[np.ix_(cluster_a, cluster_a)] @ w_a
    var_b = w_b @ cov[np.ix_(cluster_b, cluster_b)] @ w_b
    total_inv_var = 1 / (var_a + 1e-10) + 1 / (var_b + 1e-10)
    alpha_a = (1 / (var_a + 1e-10)) / total_inv_var
    alpha_b = (1 / (var_b + 1e-10)) / total_inv_var

    # 統合
    weights = np.zeros(n)
    for i, idx in enumerate(cluster_a):
        weights[idx] = alpha_a * w_a[i]
    for i, idx in enumerate(cluster_b):
        weights[idx] = alpha_b * w_b[i]

    weights = weights / (np.sum(weights) + 1e-10)
    return {tickers[i]: float(weights[i]) for i in range(n)}


# ==============================================================================
# 4. Strategy Ensemble (NCO / Risk Parity / Max Div / Min Var)
# ==============================================================================


def risk_parity_weights(cov: np.ndarray, tickers: list[str]) -> dict[str, float]:
    """
    リスクパリティ配分: 各銘柄のリスク寄与を均等化。

    Args:
        cov: 共分散行列
        tickers: 銘柄名リスト

    Returns:
        {ticker: weight} の辞書
    """
    std = np.sqrt(np.diag(cov))
    inv_std = 1.0 / (std + 1e-10)
    w = inv_std / np.sum(inv_std)
    return {tickers[i]: float(w[i]) for i in range(len(tickers))}


def min_variance_weights(cov: np.ndarray, tickers: list[str]) -> dict[str, float]:
    """
    最小分散ポートフォリオ: ポートフォリオ全体の分散を最小化。

    Args:
        cov: 共分散行列
        tickers: 銘柄名リスト

    Returns:
        {ticker: weight} の辞書
    """
    n = len(tickers)
    try:
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n)
        w = inv_cov @ ones
        w = w / (np.sum(w) + 1e-10)
        w = np.clip(w, 0, None)
        w = w / (np.sum(w) + 1e-10)
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    return {tickers[i]: float(w[i]) for i in range(n)}


def max_diversification_weights(
    cov: np.ndarray, tickers: list[str]
) -> dict[str, float]:
    """
    最大分散ポートフォリオ: 分散比率(DR)を最大化。

    Choueifaty (2008) の手法。DR = Σ(w_i * σ_i) / σ_portfolio。

    Args:
        cov: 共分散行列
        tickers: 銘柄名リスト

    Returns:
        {ticker: weight} の辞書
    """
    std = np.sqrt(np.diag(cov))
    try:
        inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ std
        w = w / (np.sum(w) + 1e-10)
        w = np.clip(w, 0, None)
        w = w / (np.sum(w) + 1e-10)
    except np.linalg.LinAlgError:
        w = np.ones(len(tickers)) / len(tickers)
    return {tickers[i]: float(w[i]) for i in range(len(tickers))}


def compute_ensemble(
    cov: np.ndarray,
    returns: pd.DataFrame,
    tickers: list[str],
    evolution_state: dict,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[float]]:
    """
    4戦略のアンサンブル配分を計算する。

    重みは evolution_state の ensemble_weights を使用（Inner Loopで更新済み）。

    Args:
        cov: 共分散行列
        returns: 日次リターンDataFrame
        tickers: 銘柄名リスト
        evolution_state: 進化パラメータ

    Returns:
        (ensemble_allocation, individual_strategies, weights)
    """
    strategies = {
        "NCO": nco_optimize(cov, tickers),
        "RiskParity": risk_parity_weights(cov, tickers),
        "MinVariance": min_variance_weights(cov, tickers),
        "MaxDiversification": max_diversification_weights(cov, tickers),
    }

    ew = evolution_state.get("ensemble_weights", [0.25, 0.25, 0.25, 0.25])
    strategy_names = list(strategies.keys())

    ensemble = {}
    for ticker in tickers:
        val = 0.0
        for i, name in enumerate(strategy_names):
            val += ew[i] * strategies[name].get(ticker, 0.0)
        ensemble[ticker] = val

    # 正規化
    total = sum(ensemble.values())
    if total > 0:
        ensemble = {k: v / total for k, v in ensemble.items()}

    return ensemble, strategies, ew


# ==============================================================================
# 5. Regime Detection (KL Divergence)
# ==============================================================================


def detect_regime(
    returns: pd.DataFrame, threshold: float = 0.05
) -> tuple[str, float, pd.Series]:
    """
    KLダイバージェンス法でマーケットレジームを検出する。

    ローリングウィンドウのリターン分布と長期ベースライン分布の乖離を測定。

    Args:
        returns: 日次リターンDataFrame
        threshold: レジーム転換閾値

    Returns:
        (regime_name, kl_value, kl_series)
    """
    portfolio_returns = returns.mean(axis=1)

    # ベースライン分布（全期間）
    baseline_mean = portfolio_returns.mean()
    baseline_std = portfolio_returns.std() + 1e-10

    # ローリング分布（直近21日）
    rolling_mean = portfolio_returns.rolling(REGIME_ROLLING_WINDOW).mean()
    rolling_std = portfolio_returns.rolling(REGIME_ROLLING_WINDOW).std() + 1e-10

    # 簡易KLダイバージェンス（正規分布仮定）
    kl = (
        np.log(rolling_std / baseline_std)
        + (baseline_std**2 + (rolling_mean - baseline_mean) ** 2)
        / (2 * rolling_std**2)
        - 0.5
    )
    kl = kl.dropna()

    if len(kl) == 0:
        return "unknown", 0.0, pd.Series(dtype=float)

    latest_kl = float(kl.iloc[-1])

    if latest_kl < threshold:
        regime = "low_vol"
    elif latest_kl < threshold * 3:
        regime = "transition"
    else:
        regime = "crisis"

    return regime, latest_kl, kl


# ==============================================================================
# 6. Exponential Gradient (Online Learning)
# ==============================================================================


def exponential_gradient_weights(
    returns: pd.DataFrame, tickers: list[str], eta: float = 0.5
) -> dict[str, float]:
    """
    Cover (1991) のExponential Gradient法。

    過去のリターンに基づき、乗法的に重みを更新。
    理論的にどんな固定ポートフォリオとも競合可能（regret bound保証）。

    Args:
        returns: 日次リターンDataFrame
        tickers: 銘柄名リスト
        eta: 学習率

    Returns:
        {ticker: weight} の辞書
    """
    n = len(tickers)
    w = np.ones(n) / n  # 均等初期化

    for _, row in returns.iterrows():
        r = row[tickers].values.astype(float)
        r = np.nan_to_num(r, nan=0.0)
        w = w * np.exp(eta * r)
        total = np.sum(w)
        if total > 0:
            w = w / total
        else:
            w = np.ones(n) / n

    return {tickers[i]: float(w[i]) for i in range(n)}


# ==============================================================================
# 7. Black-Litterman with Self-Reflection Views
# ==============================================================================


def black_litterman(
    cov: np.ndarray,
    market_caps: np.ndarray,
    views: dict[str, float] | None,
    omega_diag: dict[str, float] | None,
    tickers: list[str],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> dict[str, float]:
    """
    Black-Litterman モデル。AIの予測精度を信頼度に変換してViewsとして融合。

    Args:
        cov: 共分散行列
        market_caps: 時価総額（均等配分の場合は均一）
        views: AIの予測リターン {ticker: expected_return}
        omega_diag: AIの予測誤差分散 {ticker: variance}（← Inner Loopで進化）
        tickers: 銘柄名リスト
        risk_aversion: リスク回避係数
        tau: スカラー係数

    Returns:
        {ticker: weight} の辞書
    """
    n = len(tickers)

    # 均衡リターン (π = δΣw_mkt)
    w_mkt = market_caps / (np.sum(market_caps) + 1e-10)
    pi = risk_aversion * cov @ w_mkt

    if views is None or len(views) == 0:
        # Viewsなし → 均衡配分を返す
        w = w_mkt
    else:
        # P行列（単位行列 = 各銘柄に個別View）
        P = np.eye(n)
        Q = np.array([views.get(t, float(pi[i])) for i, t in enumerate(tickers)])

        # Ω行列（AI予測誤差の分散）
        if omega_diag is not None:
            omega_vals = [
                omega_diag.get(t, tau * float(cov[i, i]))
                for i, t in enumerate(tickers)
            ]
        else:
            omega_vals = [tau * float(cov[i, i]) for i in range(n)]
        Omega = np.diag(omega_vals)

        # BL公式
        tau_cov = tau * cov
        try:
            inv_tau_cov = np.linalg.pinv(tau_cov)
            inv_omega = np.linalg.pinv(Omega)
            posterior_cov = np.linalg.pinv(inv_tau_cov + P.T @ inv_omega @ P)
            posterior_mean = posterior_cov @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)

            # 後方リターンから最適重みを計算
            inv_cov = np.linalg.pinv(cov)
            w = inv_cov @ posterior_mean
            w = w / (np.sum(np.abs(w)) + 1e-10)
            w = np.clip(w, 0, None)
            total = np.sum(w)
            if total > 0:
                w = w / total
            else:
                w = w_mkt
        except np.linalg.LinAlgError:
            w = w_mkt

    return {tickers[i]: float(w[i]) for i in range(n)}


# ==============================================================================
# 8. Self-Reflection Engine (Track Record Management)
# ==============================================================================


def load_track_record() -> dict:
    """
    ai_track_record.json を読み込む。存在しない場合は初期構造を返す。

    Returns:
        トラックレコード辞書
    """
    if TRACK_RECORD_PATH.exists():
        with open(TRACK_RECORD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "meta": {
            "created": datetime.now().strftime("%Y-%m-%d"),
            "version": 5,
            "total_predictions": 0,
            "evolution_state": {
                "ensemble_weights": [0.25, 0.25, 0.25, 0.25],
                "regime_threshold": 0.05,
                "bl_omega_diagonal": {},
            },
        },
        "records": [],
        "evaluations": [],
    }


def save_track_record(track: dict) -> None:
    """
    ai_track_record.json を保存する。

    Args:
        track: トラックレコード辞書
    """
    with open(TRACK_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(track, f, indent=2, ensure_ascii=False)
    logger.info("トラックレコード保存完了: %s", TRACK_RECORD_PATH)


def evaluate_pending_records(
    track: dict, prices: pd.DataFrame, returns: pd.DataFrame
) -> dict:
    """
    未評価の過去予測を自動採点する。

    各レコードの予測日から現在までの実際のリターンを計算し、
    予測誤差・方向性精度・因子分解を行う。

    Args:
        track: トラックレコード辞書
        prices: 日次終値DataFrame
        returns: 日次リターンDataFrame

    Returns:
        更新されたトラックレコード
    """
    evaluated_ids = {e["record_id"] for e in track.get("evaluations", [])}
    today = datetime.now().strftime("%Y-%m-%d")

    for record in track.get("records", []):
        rid = record.get("id", "")
        if rid in evaluated_ids:
            continue
        if record.get("evaluated", False):
            continue

        record_date = record.get("date", "")
        if not record_date:
            continue

        # 予測日からの実際のリターンを計算
        try:
            record_dt = pd.Timestamp(record_date)
        except (ValueError, TypeError):
            continue

        # 最低7日経過していないと評価しない
        if (pd.Timestamp(today) - record_dt).days < 7:
            continue

        ai_alloc = record.get("ai_allocation", {})
        if not ai_alloc:
            continue

        # 各銘柄の実際リターンを計算
        actual_returns = {}
        predicted_returns = {}
        available_tickers = [t for t in ai_alloc if t in prices.columns]

        for ticker in available_tickers:
            ticker_prices = prices[ticker]
            mask = ticker_prices.index >= record_dt
            if mask.sum() < 2:
                continue
            post_prices = ticker_prices[mask]
            actual_ret = float(
                (post_prices.iloc[-1] - post_prices.iloc[0]) / post_prices.iloc[0]
            )
            actual_returns[ticker] = actual_ret
            # AI配分重みから暗示される期待リターンを推定
            predicted_returns[ticker] = float(ai_alloc.get(ticker, 0.0)) * 0.1

        if not actual_returns:
            continue

        # 予測誤差の計算
        pred_vals = np.array([predicted_returns.get(t, 0) for t in available_tickers])
        actual_vals = np.array([actual_returns.get(t, 0) for t in available_tickers])
        rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))

        # 方向性精度
        direction_matches = np.sum(np.sign(pred_vals) == np.sign(actual_vals))
        direction_accuracy = float(direction_matches / len(available_tickers))

        # 因子分解
        market_return = float(np.mean(actual_vals))
        market_timing_error = float(np.mean(pred_vals) - market_return)
        sector_errors = pred_vals - actual_vals - market_timing_error
        sector_rotation_error = float(np.std(sector_errors))
        idiosyncratic_error = float(rmse - abs(market_timing_error) - sector_rotation_error)

        # 較正スコア（確信度 vs 的中率）
        confidence = record.get("confidence", 0.5)
        calibration_score = float(1.0 - abs(confidence - direction_accuracy))

        # バイアス分析テキスト生成
        bias_parts = []
        for ticker in available_tickers:
            diff = predicted_returns.get(ticker, 0) - actual_returns.get(ticker, 0)
            if abs(diff) > 0.01:
                direction = "過大評価" if diff > 0 else "過小評価"
                bias_parts.append(f"{ticker}: {direction} {abs(diff)*100:.1f}%")
        bias_text = " / ".join(bias_parts) if bias_parts else "バイアスなし"

        evaluation = {
            "record_id": rid,
            "eval_date": today,
            "actual_returns": actual_returns,
            "predicted_vs_actual": {
                "rmse": rmse,
                "direction_accuracy": direction_accuracy,
                "calibration_score": calibration_score,
            },
            "error_decomposition": {
                "market_timing": market_timing_error,
                "sector_rotation": sector_rotation_error,
                "idiosyncratic": idiosyncratic_error,
            },
            "bias_analysis": bias_text,
        }

        track["evaluations"].append(evaluation)
        record["evaluated"] = True
        logger.info("予測 %s を自動採点: RMSE=%.4f, 方向精度=%.1f%%", rid, rmse, direction_accuracy * 100)

    return track


def evolve_inner_loop(
    track: dict,
    returns: pd.DataFrame,
    cov: np.ndarray,
    tickers: list[str],
) -> dict:
    """
    Inner Loop: 過去の評価結果から数学パラメータを更新する。

    1. Ensemble重み = 各戦略の直近OOS Sharpeのsoftmax
    2. BL Ω = AI予測誤差の分散
    3. レジーム閾値 = 最適化

    Args:
        track: トラックレコード辞書
        returns: 日次リターンDataFrame
        cov: 共分散行列
        tickers: 銘柄名リスト

    Returns:
        更新されたトラックレコード
    """
    evaluations = track.get("evaluations", [])
    evolution_state = track["meta"]["evolution_state"]

    if len(evaluations) < 2:
        logger.info("評価データ不足（%d件）、Inner Loop スキップ", len(evaluations))
        return track

    # --- 1. Ensemble重みの進化 ---
    recent_returns = returns.tail(ENSEMBLE_LOOKBACK_DAYS)
    if len(recent_returns) > 10:
        strategies = {
            "NCO": nco_optimize(cov, tickers),
            "RiskParity": risk_parity_weights(cov, tickers),
            "MinVariance": min_variance_weights(cov, tickers),
            "MaxDiversification": max_diversification_weights(cov, tickers),
        }

        sharpes = []
        for name, alloc in strategies.items():
            w = np.array([alloc.get(t, 0) for t in tickers])
            portfolio_ret = (recent_returns[tickers].values @ w)
            if len(portfolio_ret) > 1 and np.std(portfolio_ret) > 0:
                sharpe = float(np.mean(portfolio_ret) / np.std(portfolio_ret) * np.sqrt(252))
            else:
                sharpe = 0.0
            sharpes.append(sharpe)

        # Softmax（temperature=0.5）
        sharpes_arr = np.array(sharpes)
        exp_sharpes = np.exp(sharpes_arr / 0.5)
        ensemble_weights = (exp_sharpes / np.sum(exp_sharpes)).tolist()
        evolution_state["ensemble_weights"] = ensemble_weights
        logger.info("Ensemble重み更新: %s", [f"{w:.3f}" for w in ensemble_weights])

    # --- 2. BL Ω（AI予測信頼度）の進化 ---
    bl_omega = {}
    for ticker in tickers:
        errors = []
        for ev in evaluations:
            actual = ev.get("actual_returns", {})
            record = next(
                (r for r in track["records"] if r.get("id") == ev.get("record_id")),
                None,
            )
            if record and ticker in actual:
                ai_alloc = record.get("ai_allocation", {})
                predicted = ai_alloc.get(ticker, 0) * 0.1
                errors.append(predicted - actual[ticker])
        if len(errors) >= 2:
            bl_omega[ticker] = float(np.var(errors))
    if bl_omega:
        evolution_state["bl_omega_diagonal"] = bl_omega
        logger.info("BL Ω更新: %d銘柄", len(bl_omega))

    # --- 3. レジーム閾値の進化 ---
    # 直近の予測精度が高い閾値を探索
    if len(evaluations) >= 3:
        best_threshold = evolution_state.get("regime_threshold", 0.05)
        best_score = -1.0
        for threshold in np.linspace(0.01, 0.15, 15):
            correct = 0
            total_eval = 0
            for ev in evaluations[-10:]:
                accuracy = ev.get("predicted_vs_actual", {}).get("direction_accuracy", 0)
                if accuracy > 0.5:
                    correct += 1
                total_eval += 1
            score = correct / (total_eval + 1e-10)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        evolution_state["regime_threshold"] = best_threshold
        logger.info("レジーム閾値更新: %.3f", best_threshold)

    track["meta"]["evolution_state"] = evolution_state
    return track


# ==============================================================================
# メイン処理
# ==============================================================================


def run_daily_evolution() -> dict[str, Any]:
    """
    日次進化の全処理を実行する。

    Returns:
        計算結果の辞書（Streamlitダッシュボードが参照）
    """
    logger.info("=" * 60)
    logger.info("🧬 Daily Evolution Engine 起動 [%s]", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    # 設定読み込み
    config = load_config()
    holdings = config.get("portfolio", {}).get("holdings", [])
    tickers = [h["ticker"] for h in holdings]
    logger.info("対象銘柄: %s", tickers)

    # データ取得
    prices = fetch_price_data(tickers)
    available_tickers = [t for t in tickers if t in prices.columns]
    returns = calc_log_returns(prices[available_tickers])

    # Denoised + Detoned 共分散行列
    cov_raw = returns[available_tickers].cov().values
    n_samples = len(returns)
    cov_denoised = denoise_covariance(cov_raw, n_samples)
    cov_detoned = detone_covariance(cov_denoised)

    # トラックレコード読み込み
    track = load_track_record()

    # 未評価予測の自動採点
    track = evaluate_pending_records(track, prices, returns)

    # Inner Loop: パラメータ進化
    track = evolve_inner_loop(track, returns, cov_detoned, available_tickers)

    evolution_state = track["meta"]["evolution_state"]

    # 4戦略アンサンブル
    ensemble_alloc, individual_strategies, ew = compute_ensemble(
        cov_detoned, returns, available_tickers, evolution_state
    )

    # レジーム検出
    threshold = evolution_state.get("regime_threshold", 0.05)
    regime, kl_value, kl_series = detect_regime(returns, threshold)
    logger.info("レジーム: %s (KL=%.4f)", regime, kl_value)

    # Exponential Gradient
    eg_alloc = exponential_gradient_weights(returns, available_tickers)

    # Black-Litterman（AI Viewsがあれば融合）
    last_ai_record = None
    for r in reversed(track.get("records", [])):
        if r.get("ai_allocation"):
            last_ai_record = r
            break

    market_caps = np.ones(len(available_tickers))  # 均等仮定
    if last_ai_record:
        views = {
            t: float(last_ai_record["ai_allocation"].get(t, 0)) * 0.1
            for t in available_tickers
        }
        omega_diag = evolution_state.get("bl_omega_diagonal")
        bl_alloc = black_litterman(
            cov_detoned, market_caps, views, omega_diag, available_tickers
        )
    else:
        bl_alloc = black_litterman(
            cov_detoned, market_caps, None, None, available_tickers
        )

    # === 高精度予測アンサンブル + 自己学習サイクル統合 ===
    advanced_signals: list = []
    learning_result: dict = {}
    try:
        from advanced_predictor import (
            predict_all, archive_predictions, run_learning_cycle,
        )
        prices_dict = {t: prices[t].dropna().values for t in available_tickers if t in prices.columns}

        # 1. 予測生成
        advanced_signals = predict_all(prices_dict)

        # 2. 履歴アーカイブ（後の評価のため永続化）
        archive_predictions(advanced_signals)

        # 3. 毎日の学習サイクル:
        #    過去予測の答え合わせ → サブ予測器ウェイトの Bayesian 更新
        #    → 学習ジャーナル追記
        learning_result = run_learning_cycle(prices_dict)
        logger.info(
            "🧠 学習サイクル完了: 新規評価 %d件 / 直近精度 %.0f%% / 累計 %d件",
            learning_result.get("n_evaluated", 0),
            learning_result.get("accuracy_recent", 0) * 100,
            learning_result.get("n_total_evaluated", 0),
        )
    except Exception as e:
        logger.warning("高精度予測スキップ: %s", e)

    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "tickers": available_tickers,
        "regime": regime,
        "kl_value": kl_value,
        "allocations": {
            "ensemble": ensemble_alloc,
            "bl": bl_alloc,
            "eg": eg_alloc,
            "strategies": individual_strategies,
        },
        "ensemble_weights": ew,
        "evolution_state": evolution_state,
        "advanced_signals": advanced_signals,
        "learning_summary": learning_result,
    }

    # トラックレコード保存
    save_track_record(track)

    # 結果をキャッシュファイルにも保存（Streamlitが読む用）
    results_path = Path(__file__).resolve().parent / "latest_evolution_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("結果保存: %s", results_path)

    logger.info("=" * 60)
    logger.info("🧬 Daily Evolution 完了")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_daily_evolution()
