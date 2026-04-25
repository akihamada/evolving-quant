#!/usr/bin/env python3
"""
portfolio_advisor_pro.py — プロ級ポートフォリオ・アドバイザリー強化群

【6機能】
  1. Black-Litterman with Master Views : Master Wisdom 信号を BL の view として統合
  2. Risk Parity Overlay              : ファクター別リスク寄与均等化
  3. Sector Rotation Layer            : セクター強弱の動的補正
  4. Adversarial Stress Test          : 過去最悪日でのポートフォリオ被害シミュレーション
  5. Tax-Aware Optimization           : NISA/特定口座を考慮した売買助言
  6. Realized P&L Feedback            : 実購入ログ vs 予測の事後検証

依存: numpy のみ
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("portfolio-pro")

SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_OPT_PATH = SCRIPT_DIR / "data" / "portfolio_optimization.json"
SECTOR_ROTATION_PATH = SCRIPT_DIR / "data" / "sector_rotation_history.json"
STRESS_TEST_PATH = SCRIPT_DIR / "data" / "stress_test_results.json"
PNL_FEEDBACK_PATH = SCRIPT_DIR / "data" / "pnl_feedback.json"

# ==============================================================================
# 1. Black-Litterman with Master Wisdom Views
# ==============================================================================


def black_litterman_with_master(
    cov: np.ndarray,
    market_caps: np.ndarray,
    master_signals: list[dict[str, Any]],
    tickers: list[str],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    confidence_scale: float = 0.05,
) -> dict[str, float]:
    """Master Wisdom の予測を View として Black-Litterman に統合する。

    Master の合成スコアを期待リターンに変換し、確信度を Ω 行列の対角に。

    Args:
        cov: 共分散行列 (n × n)
        market_caps: 各銘柄の時価総額（市場均衡用）
        master_signals: Master Wisdom の予測リスト
        tickers: 銘柄リスト
        risk_aversion: リスク回避係数
        tau: BL スカラー係数
        confidence_scale: 合成スコアをリターンに変換する係数

    Returns:
        {ticker: weight}
    """
    n = len(tickers)
    if n == 0:
        return {}

    # 市場均衡リターン
    w_mkt = market_caps / (np.sum(market_caps) + 1e-10)
    pi = risk_aversion * cov @ w_mkt

    # Master signal を view に変換
    # composite_score [-1, 1] → 期待年率リターン (composite × confidence_scale)
    signal_map = {s["ticker"]: s for s in master_signals}
    Q = np.zeros(n)
    omega_diag = np.zeros(n)
    for i, t in enumerate(tickers):
        if t in signal_map:
            sig = signal_map[t]
            comp = sig.get("composite_score", 0)
            conf = sig.get("confidence", 0.5)
            Q[i] = comp * confidence_scale
            # 確信度が低いほど Ω 大（uncertain view）
            omega_diag[i] = (1 - conf) * tau * float(cov[i, i]) * 5
        else:
            Q[i] = float(pi[i])
            omega_diag[i] = tau * float(cov[i, i])

    P = np.eye(n)
    Omega = np.diag(np.maximum(omega_diag, 1e-8))
    tau_cov = tau * cov

    # BL 公式
    ridge = 1e-8 * np.eye(n)
    try:
        inv_tau_cov = np.linalg.pinv(tau_cov + ridge)
        inv_omega = np.linalg.pinv(Omega + ridge)
        post_cov = np.linalg.pinv(inv_tau_cov + P.T @ inv_omega @ P + ridge)
        post_mean = post_cov @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
        if np.any(~np.isfinite(post_mean)):
            return dict(zip(tickers, w_mkt.tolist()))
        inv_cov = np.linalg.pinv(cov + ridge)
        w = inv_cov @ post_mean
        # クリップ + 正規化
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
# 2. Risk Parity Overlay
# ==============================================================================


def risk_parity_weights(cov: np.ndarray, tickers: list[str]) -> dict[str, float]:
    """各銘柄のリスク寄与を均等化する重み。

    全銘柄が portfolio risk に均等に貢献するよう調整。
    """
    n = len(tickers)
    if n == 0:
        return {}
    std = np.sqrt(np.diag(cov))
    inv_std = 1.0 / (std + 1e-10)
    w = inv_std / np.sum(inv_std)
    return {tickers[i]: float(w[i]) for i in range(n)}


def factor_risk_parity_blend(
    bl_weights: dict[str, float],
    rp_weights: dict[str, float],
    rp_blend: float = 0.3,
) -> dict[str, float]:
    """Black-Litterman と Risk Parity をブレンドする。

    Args:
        rp_blend: RP の寄与度 (0 = 純粋 BL, 1 = 純粋 RP)
    """
    blended = {}
    for t in bl_weights:
        blended[t] = (1 - rp_blend) * bl_weights[t] + rp_blend * rp_weights.get(t, 0)
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


# ==============================================================================
# 3. Sector Rotation Layer
# ==============================================================================


def detect_sector_rotation(
    prices_dict: dict[str, np.ndarray],
    sector_groups: dict[str, list[str]],
    lookback_30d: bool = True,
) -> dict[str, Any]:
    """セクター別リターンから現在のセクターローテーションを検出。

    Args:
        prices_dict: {ticker: 価格配列}
        sector_groups: セクターグループ定義
        lookback_30d: 直近30日比較

    Returns:
        {"hot_sectors": [...], "cold_sectors": [...], "tilts": {sector: bias}}
    """
    sector_returns = {}
    days = 30 if lookback_30d else 90

    for sector, tickers in sector_groups.items():
        rets = []
        for t in tickers:
            p = prices_dict.get(t)
            if p is None or len(p) < days + 1:
                continue
            r = (float(p[-1]) - float(p[-days])) / float(p[-days])
            rets.append(r)
        if rets:
            sector_returns[sector] = float(np.mean(rets))

    if not sector_returns:
        return {"hot_sectors": [], "cold_sectors": [], "tilts": {}}

    sorted_sectors = sorted(sector_returns.items(), key=lambda x: -x[1])
    market_avg = float(np.mean(list(sector_returns.values())))

    hot = [s for s, r in sorted_sectors if r > market_avg + 0.03][:3]
    cold = [s for s, r in sorted_sectors[::-1] if r < market_avg - 0.03][:3]

    # tilts: 平均からの乖離 (小さなボーナス/ペナルティとして使用)
    tilts = {s: r - market_avg for s, r in sector_returns.items()}

    result = {
        "hot_sectors": hot,
        "cold_sectors": cold,
        "tilts": {k: round(v, 4) for k, v in tilts.items()},
        "market_avg_return": round(market_avg, 4),
        "lookback_days": days,
        "executed_at": datetime.now().isoformat(),
    }

    SECTOR_ROTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SECTOR_ROTATION_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("📊 セクターローテーション: hot=%s / cold=%s", hot, cold)
    return result


# ==============================================================================
# 4. Adversarial Stress Test
# ==============================================================================


def adversarial_stress_test(
    weights: dict[str, float],
    prices_dict: dict[str, np.ndarray],
    historical_scenarios: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """過去の最悪日 + 仮想ストレスシナリオで損失をシミュレート。

    Args:
        weights: ポートフォリオ重み
        prices_dict: 価格履歴
        historical_scenarios: カスタムシナリオ（None ならデフォルト）

    Returns:
        各シナリオでの想定損失
    """
    if historical_scenarios is None:
        historical_scenarios = [
            {"name": "1日 -5% Tech売り", "shocks": {"NVDA": -0.10, "TSM": -0.08, "MSFT": -0.05, "PLTR": -0.10, "ARM": -0.10, "MRVL": -0.10, "DDOG": -0.10, "default": -0.03}},
            {"name": "金利急上昇 (Tech暴落)", "shocks": {"NVDA": -0.15, "TSM": -0.12, "PLTR": -0.20, "DDOG": -0.18, "VYM": +0.02, "GLD": +0.03, "default": -0.05}},
            {"name": "リセッション (全面安)", "shocks": {"default": -0.08, "GLD": +0.05, "VYM": -0.05, "1328.T": +0.04}},
            {"name": "AI バブル崩壊", "shocks": {"NVDA": -0.30, "PLTR": -0.40, "ARM": -0.25, "TSM": -0.20, "DDOG": -0.30, "MRVL": -0.25, "default": -0.05}},
            {"name": "VIX急騰 (パニック売り)", "shocks": {"default": -0.10, "GLD": +0.06}},
        ]

    results = []
    portfolio_value = 0.0
    # 簡易: 重みを持ち分とみなし1ドルあたり収益を計算
    for scenario in historical_scenarios:
        loss = 0.0
        for t, w in weights.items():
            shock = scenario["shocks"].get(t, scenario["shocks"].get("default", -0.05))
            loss += w * shock
        results.append({
            "scenario": scenario["name"],
            "portfolio_pnl_pct": round(loss * 100, 2),
            "rating": "🔴" if loss < -0.10 else "🟡" if loss < -0.05 else "🟢",
        })

    # 実際の最悪日（過去データから1日 worst-case を計算）
    worst_1d = None
    if prices_dict:
        all_returns = []
        for date_idx in range(-min(252, max(len(p) for p in prices_dict.values()) - 1), 0):
            day_loss = 0.0
            valid = False
            for t, w in weights.items():
                p = prices_dict.get(t)
                if p is None or abs(date_idx) + 1 >= len(p):
                    continue
                r = float((p[date_idx] - p[date_idx - 1]) / p[date_idx - 1])
                day_loss += w * r
                valid = True
            if valid:
                all_returns.append(day_loss)
        if all_returns:
            worst_1d = float(np.min(all_returns)) * 100
            results.append({
                "scenario": "過去1年最悪1日 (再現)",
                "portfolio_pnl_pct": round(worst_1d, 2),
                "rating": "🔴" if worst_1d < -10 else "🟡" if worst_1d < -5 else "🟢",
            })

    payload = {
        "scenarios": results,
        "worst_case_1d_historical": worst_1d,
        "executed_at": datetime.now().isoformat(),
    }
    STRESS_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRESS_TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "🛡️ ストレステスト: %d シナリオ / 最悪 %.1f%%",
        len(results),
        min(r["portfolio_pnl_pct"] for r in results)
    )
    return payload


# ==============================================================================
# 5. Tax-Aware Optimization
# ==============================================================================


def tax_aware_recommendation(
    holdings: dict[str, Any],
    actions: dict[str, str],
) -> list[dict[str, Any]]:
    """NISA/特定口座を考慮した税効率的な売買助言を生成。

    ロジック:
      - 利益 SELL → NISA優先（税金0%）
      - 損失 SELL → 特定口座優先（損益通算可）
      - 含み益小さい場合は持ち越し検討（税金待ち）

    Args:
        holdings: portfolio_holdings.json の内容
        actions: {ticker: BUY/HOLD/SELL}

    Returns:
        税効率推奨リスト
    """
    recs = []
    sell_tickers = [t for t, a in actions.items() if a == "SELL"]
    if not sell_tickers:
        return recs

    # 全保有ポジションを口座別に集計
    positions: dict[str, list[dict]] = {}
    for acct in ["tokutei", "nisa"]:
        for s in holdings.get("us_stocks", {}).get(acct, []):
            t = s.get("ticker", "")
            positions.setdefault(t, []).append({
                "account": acct,
                "shares": s.get("shares", 0),
                "cost": s.get("cost_basis_usd", 0),
                "price": s.get("current_price_usd", 0),
                "pnl": s.get("unrealized_pnl_usd", 0),
            })

    for t in sell_tickers:
        pos_list = positions.get(t, [])
        if not pos_list:
            continue

        # 利益・損失で口座優先順を判定
        for pos in pos_list:
            pnl = pos["pnl"]
            acct = pos["account"]
            shares = pos["shares"]
            if shares == 0:
                continue

            if pnl > 0 and acct == "nisa":
                priority = "🟢 最優先売却 (NISA利益、税金0)"
            elif pnl > 0 and acct == "tokutei":
                priority = "🟡 後回し検討 (税金20%発生、可能ならNISA売却優先)"
            elif pnl < 0 and acct == "tokutei":
                priority = "🟢 損益通算チャンス (損失で他の利益相殺)"
            elif pnl < 0 and acct == "nisa":
                priority = "🔴 NISA損失は通算不可 → 売却見送り推奨"
            else:
                priority = "中立"

            recs.append({
                "ticker": t,
                "account": acct,
                "shares": shares,
                "pnl_usd": round(pnl, 2),
                "tax_priority": priority,
            })

    return recs


# ==============================================================================
# 6. Realized P&L Feedback Loop
# ==============================================================================


def realized_pnl_feedback(
    purchase_log: dict[str, Any],
    master_history: dict[str, Any],
) -> dict[str, Any]:
    """実購入ログと予測履歴を照合し、実体験ベースの学習統計を生成。

    Args:
        purchase_log: data/purchase_log.json
        master_history: master_prediction_history

    Returns:
        {"matched_predictions": int, "win_rate", "avg_return"}
    """
    purchases = purchase_log.get("purchases", [])
    predictions = master_history.get("predictions", [])
    if not purchases or not predictions:
        return {"matched_predictions": 0, "note": "データ不足"}

    matched = []
    for purchase in purchases:
        if purchase.get("type") not in ("manual_buy", "tsumitate"):
            continue
        p_date = purchase.get("date", "")[:10]
        p_ticker = purchase.get("ticker", "")
        # 同日・同銘柄の予測を探す
        for pred in predictions:
            if pred.get("ticker") == p_ticker and pred.get("date") == p_date:
                actual = pred.get("actual_returns", {}).get("30d")
                if actual is not None:
                    matched.append({
                        "date": p_date,
                        "ticker": p_ticker,
                        "predicted_signal": pred.get("signal", "HOLD"),
                        "predicted_score": pred.get("composite_score", 0),
                        "actual_return_30d": actual,
                    })
                break

    if not matched:
        return {"matched_predictions": 0, "note": "実購入と予測の一致なし"}

    returns = [m["actual_return_30d"] for m in matched]
    avg_ret = float(np.mean(returns))
    win_rate = float(np.mean([r > 0 for r in returns]))

    # 予測 BUY → 実際UP の的中率
    buy_matches = [m for m in matched if "BUY" in m["predicted_signal"]]
    buy_correct = sum(1 for m in buy_matches if m["actual_return_30d"] > 0)
    buy_rate = buy_correct / len(buy_matches) if buy_matches else 0

    payload = {
        "matched_predictions": len(matched),
        "avg_realized_return_30d": round(avg_ret, 4),
        "win_rate": round(win_rate, 4),
        "buy_signal_accuracy": round(buy_rate, 4),
        "executed_at": datetime.now().isoformat(),
    }
    PNL_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PNL_FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "💰 実P&Lフィードバック: 一致 %d 件 / 勝率 %.0f%% / 平均 %.1f%%",
        len(matched), win_rate * 100, avg_ret * 100
    )
    return payload
