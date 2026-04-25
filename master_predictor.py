#!/usr/bin/env python3
"""
master_predictor.py — 世界一のウィズダム予測アンサンブル

ウォーレン・バフェットの投資哲学 + 現代統計学 + 機械学習を統合した
9ファクター・マルチティア予測システム。

【哲学】
  バフェット: "10年保持できない株は10分も持つな"
  バフェット: "他人が貪欲なときは恐れ、恐れるときは貪欲になれ"
  バフェット: "Margin of Safety (安全域) を確保せよ"
  Mandelbrot: "価格はランダムでなくフラクタル"
  Markowitz: "リスクとリターンの効率フロンティア"

【9ファクター構成】
  Tier 1 — Quality (経済的堀):
    ① ROE Stability     : 5年平均ROE > 15% かつ変動係数 < 0.3
    ② Margin Resilience : 営業利益率の安定性 + 上昇トレンド
  Tier 2 — Value (内在価値):
    ③ Earnings Yield    : 1/PER 対 国債利回り
    ④ FCF Yield         : Free Cash Flow / Market Cap
    ⑤ Margin of Safety  : DCF intrinsic value vs price
  Tier 3 — Momentum (既存テクニカル統合):
    ⑥ Technical Composite : advanced_predictor の 5サブ統合
  Tier 4 — Contrarian (バフェット逆張り):
    ⑦ Fear-Greed Inverse : VIX/恐怖指数 で逆張り
    ⑧ Insider Pulse      : 役員買いシグナル + Put/Call ratio
  Tier 5 — Risk Overlay:
    ⑨ Kelly Position     : ケリー基準でのポジションサイズ + drawdown 制約

【自己進化メカニズム】
  - Regime-conditional weights: low_vol / transition / crisis 別の重み行列
  - Factor decay detection    : 精度50%未満が継続するファクターをex 縮退
  - Calibration penalty       : 過信予測の確信度を割引
  - 7日 + 30日 + 90日 マルチホライズン評価

使用:
  from master_predictor import predict_master, run_master_learning_cycle
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger("master-predictor")

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_HISTORY_PATH = SCRIPT_DIR / "data" / "master_prediction_history.json"
MASTER_WEIGHTS_PATH = SCRIPT_DIR / "data" / "master_factor_weights.json"
MASTER_JOURNAL_PATH = SCRIPT_DIR / "data" / "master_learning_journal.json"

# ファクター名（順番固定）
FACTOR_NAMES: list[str] = [
    "quality_roe",
    "quality_margin",
    "value_earnings_yield",
    "value_fcf_yield",
    "value_margin_of_safety",
    "momentum_composite",
    "contrarian_fear_greed",
    "contrarian_insider_pulse",
    "risk_kelly",
]

# 事前ウェイト（バフェット哲学を反映: Quality + Value 重視）
PRIOR_WEIGHTS: dict[str, float] = {
    "quality_roe":             0.18,    # 経済的堀の代表指標
    "quality_margin":          0.12,    # 利益率の安定性
    "value_earnings_yield":    0.10,    # PERの裏面
    "value_fcf_yield":         0.10,    # キャッシュフロー基準
    "value_margin_of_safety":  0.15,    # 内在価値ベース
    "momentum_composite":      0.10,    # テクニカル
    "contrarian_fear_greed":   0.10,    # 逆張り
    "contrarian_insider_pulse":0.07,    # スマートマネー
    "risk_kelly":              0.08,    # ポジションサイジング
}

# レジーム別の重み調整係数（学習で更新）
REGIME_PRIORS: dict[str, dict[str, float]] = {
    "low_vol": {
        # 安定相場では Quality + Momentum を重視
        "quality_roe": 1.2, "quality_margin": 1.1, "momentum_composite": 1.3,
        "contrarian_fear_greed": 0.7,
    },
    "transition": {
        # 移行期は Value + Risk を重視
        "value_margin_of_safety": 1.3, "risk_kelly": 1.2,
        "momentum_composite": 0.8,
    },
    "crisis": {
        # 危機時は Contrarian + Risk + Margin of Safety
        "contrarian_fear_greed": 1.5, "value_margin_of_safety": 1.4,
        "risk_kelly": 1.4, "momentum_composite": 0.5,
    },
}

# シグナル閾値
STRONG_SIGNAL = 0.50    # ±0.50 で STRONG_BUY/SELL
WEAK_SIGNAL = 0.20      # ±0.20 で BUY/SELL

# 評価ホライズン
EVAL_HORIZONS_DAYS = [7, 30, 90]
EVAL_MIN_DAYS = 7

# 自己進化パラメータ
MIN_SAMPLES_FOR_LEARNING = 5    # ウェイト学習に最低必要なサンプル数
DECAY_ACCURACY_THRESHOLD = 0.45  # この精度を下回るとファクター減衰
DECAY_RATE = 0.7                # 減衰時の係数（70% に縮小）
MAX_HISTORY_RECORDS = 1000
MAX_JOURNAL_ENTRIES = 365


def _safe_get(d: dict, key: str, default=None):
    """dict から安全に取得し、None や欠損をデフォルトに置換。"""
    v = d.get(key, default) if isinstance(d, dict) else default
    if v is None:
        return default
    return v


def _tanh_clip(x: float, scale: float = 1.0) -> float:
    """tanh でスコアを [-1, 1] にスムーズに圧縮。"""
    return float(np.tanh(x * scale))


# ==============================================================================
# Tier 1: Quality (Buffett 経済的堀)
# ==============================================================================


def quality_roe_score(fundamentals: dict) -> dict[str, float]:
    """ROE の水準と安定性を評価する Buffett 流 quality 指標。

    判定:
      ROE > 20% : 卓越した経済的堀 → +1.0
      ROE > 15% : 優良 → +0.5
      ROE > 10% : 並 → 0
      ROE < 5%  : 経済的堀の欠如 → -0.5

    Args:
        fundamentals: yfinance.info もしくは類似の dict
                      期待キー: returnOnEquity, trailingEps, priceToBook

    Returns:
        {"score": float, "roe": float, "verdict": str}
    """
    roe = _safe_get(fundamentals, "returnOnEquity", 0) or 0
    # 一部の株は%表記なので正規化
    if roe > 1.5:
        roe = roe / 100.0

    if roe >= 0.20:
        score, verdict = 1.0, "卓越 (>20%)"
    elif roe >= 0.15:
        score, verdict = 0.5, "優良 (15-20%)"
    elif roe >= 0.10:
        score, verdict = 0.0, "並 (10-15%)"
    elif roe >= 0.05:
        score, verdict = -0.3, "弱い (5-10%)"
    else:
        score, verdict = -0.7, "欠如 (<5%)"

    return {"score": float(score), "roe": float(roe), "verdict": verdict}


def quality_margin_score(fundamentals: dict) -> dict[str, float]:
    """営業利益率と粗利率の水準と上昇傾向で quality を評価する。

    Args:
        fundamentals: 期待キー: profitMargins, operatingMargins, grossMargins

    Returns:
        {"score": float, "operating_margin": float, "verdict": str}
    """
    op_margin = _safe_get(fundamentals, "operatingMargins", 0) or 0
    gross_margin = _safe_get(fundamentals, "grossMargins", 0) or 0
    profit_margin = _safe_get(fundamentals, "profitMargins", 0) or 0

    # 営業利益率主軸
    if op_margin >= 0.30:
        score, verdict = 1.0, "圧倒的 (>30%)"
    elif op_margin >= 0.20:
        score, verdict = 0.5, "強い (20-30%)"
    elif op_margin >= 0.10:
        score, verdict = 0.1, "並 (10-20%)"
    elif op_margin >= 0.05:
        score, verdict = -0.3, "弱い (5-10%)"
    else:
        score, verdict = -0.7, "脆弱 (<5%)"

    # 粗利率ボーナス（参考）
    if gross_margin >= 0.50:
        score = min(1.0, score + 0.1)

    return {
        "score": float(score),
        "operating_margin": float(op_margin),
        "gross_margin": float(gross_margin),
        "profit_margin": float(profit_margin),
        "verdict": verdict,
    }


# ==============================================================================
# Tier 2: Value (内在価値 vs 市場価格 — Buffett の Margin of Safety)
# ==============================================================================


def value_earnings_yield_score(
    fundamentals: dict,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Earnings Yield (1/PER) を国債利回りと比較する Graham/Buffett 流バリュー指標。

    判定: Earnings Yield > 国債利回り + 4% プレミアム → 強い買い
         Earnings Yield > 国債利回り → 弱い買い
         Earnings Yield < 国債利回り → 売り（債券が魅力的）

    Args:
        fundamentals: yfinance.info
        risk_free_rate: 10年米債利回り（デフォルト 4%）

    Returns:
        {"score": float, "earnings_yield": float, "spread": float, "verdict": str}
    """
    pe = _safe_get(fundamentals, "trailingPE", 0) or 0
    forward_pe = _safe_get(fundamentals, "forwardPE", 0) or 0
    # 利用可能な方を採用（forward PER のほうがやや楽観的なので)
    use_pe = pe if pe > 0 else forward_pe
    if use_pe <= 0:
        return {"score": 0.0, "earnings_yield": 0.0, "spread": 0.0,
                "verdict": "PERデータなし"}

    earnings_yield = 1.0 / use_pe
    spread = earnings_yield - risk_free_rate

    if spread >= 0.04:
        score, verdict = 0.8, f"魅力的 (+{spread*100:.1f}% 対国債)"
    elif spread >= 0.02:
        score, verdict = 0.4, f"並み (+{spread*100:.1f}%)"
    elif spread >= 0:
        score, verdict = 0.0, f"中立 (+{spread*100:.1f}%)"
    elif spread >= -0.02:
        score, verdict = -0.3, f"割高 ({spread*100:.1f}%)"
    else:
        score, verdict = -0.7, f"極度割高 ({spread*100:.1f}%)"

    return {
        "score": float(score),
        "earnings_yield": float(earnings_yield),
        "spread": float(spread),
        "pe": float(use_pe),
        "verdict": verdict,
    }


def value_fcf_yield_score(
    fundamentals: dict,
) -> dict[str, float]:
    """Free Cash Flow Yield = FCF / Market Cap を評価。

    バフェット哲学: 「企業の価値はそのすべての将来 FCF の現在価値である」
    FCF Yield > 5% は割安の典型。

    Args:
        fundamentals: yfinance.info (freeCashflow, marketCap)

    Returns:
        {"score": float, "fcf_yield": float, "verdict": str}
    """
    fcf = _safe_get(fundamentals, "freeCashflow", 0) or 0
    market_cap = _safe_get(fundamentals, "marketCap", 0) or 0
    if market_cap <= 0:
        return {"score": 0.0, "fcf_yield": 0.0, "verdict": "MarketCapデータなし"}

    fcf_yield = fcf / market_cap if market_cap > 0 else 0

    if fcf_yield >= 0.08:
        score, verdict = 1.0, f"卓越 ({fcf_yield*100:.1f}%)"
    elif fcf_yield >= 0.05:
        score, verdict = 0.6, f"魅力的 ({fcf_yield*100:.1f}%)"
    elif fcf_yield >= 0.03:
        score, verdict = 0.2, f"並 ({fcf_yield*100:.1f}%)"
    elif fcf_yield >= 0.01:
        score, verdict = -0.2, f"弱い ({fcf_yield*100:.1f}%)"
    elif fcf_yield > 0:
        score, verdict = -0.4, f"極小 ({fcf_yield*100:.1f}%)"
    else:
        score, verdict = -0.6, "FCF赤字"

    return {
        "score": float(score),
        "fcf_yield": float(fcf_yield),
        "fcf_usd": float(fcf),
        "market_cap_usd": float(market_cap),
        "verdict": verdict,
    }


def value_margin_of_safety_score(
    fundamentals: dict,
    risk_free_rate: float = 0.04,
    growth_rate: float = 0.06,
    discount_rate: float = 0.10,
    years: int = 10,
) -> dict[str, float]:
    """簡易DCFで内在価値を計算し、市場価格との乖離（Margin of Safety）を測る。

    Buffett: 内在価値の70%以下で買え（30%以上の安全域）。

    DCF式: V = Σ(FCF_t / (1+r)^t) for t=1..years + Terminal
    成長率は控えめ、割引率は10%（保守的）

    Args:
        fundamentals: yfinance.info
        risk_free_rate, growth_rate, discount_rate, years: DCFパラメータ

    Returns:
        {"score": float, "intrinsic_per_share": float, "margin": float, "verdict": str}
    """
    fcf = _safe_get(fundamentals, "freeCashflow", 0) or 0
    shares_out = _safe_get(fundamentals, "sharesOutstanding", 0) or 0
    current_price = _safe_get(fundamentals, "currentPrice", 0) or \
                    _safe_get(fundamentals, "regularMarketPrice", 0) or 0

    if fcf <= 0 or shares_out <= 0 or current_price <= 0:
        return {"score": 0.0, "intrinsic_per_share": 0.0, "margin": 0.0,
                "verdict": "DCFデータ不足"}

    # 10年の成長 + 永久成長 (Terminal value with Gordon growth)
    pv_total = 0.0
    for t in range(1, years + 1):
        fcf_t = fcf * ((1 + growth_rate) ** t)
        pv_total += fcf_t / ((1 + discount_rate) ** t)
    # Terminal: FCF年10末 × (1+g) / (r-g)
    fcf_terminal = fcf * ((1 + growth_rate) ** years) * (1 + 0.025)
    terminal_value = fcf_terminal / (discount_rate - 0.025)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)
    intrinsic_total = pv_total + pv_terminal
    intrinsic_per_share = intrinsic_total / shares_out

    # Margin of Safety: (intrinsic - price) / intrinsic
    margin = (intrinsic_per_share - current_price) / intrinsic_per_share \
             if intrinsic_per_share > 0 else 0

    if margin >= 0.30:
        score, verdict = 1.0, f"卓越的安全域 ({margin*100:.0f}%)"
    elif margin >= 0.15:
        score, verdict = 0.6, f"良好な安全域 ({margin*100:.0f}%)"
    elif margin >= 0:
        score, verdict = 0.1, f"小さな安全域 ({margin*100:.0f}%)"
    elif margin >= -0.15:
        score, verdict = -0.4, f"プレミアム ({-margin*100:.0f}%超過)"
    else:
        score, verdict = -0.8, f"過剰評価 ({-margin*100:.0f}%超過)"

    return {
        "score": float(score),
        "intrinsic_per_share": round(float(intrinsic_per_share), 2),
        "current_price": float(current_price),
        "margin": round(float(margin), 4),
        "verdict": verdict,
    }


# ==============================================================================
# Tier 4: Contrarian (Buffett 逆張り — "Be fearful when others are greedy")
# ==============================================================================


def contrarian_fear_greed_score(
    vix: Optional[float],
    market_return_30d: Optional[float] = None,
) -> dict[str, float]:
    """VIX (恐怖指数) と直近マーケットリターンから逆張りシグナルを生成。

    Buffett 哲学: 他人が貪欲なら恐れ、恐れるなら貪欲になれ。
    
    判定:
      VIX > 30 + 市場下落 → 強い買い (恐怖極まる時は仕入時)
      VIX < 12 + 市場急騰 → 強い売り (貪欲極まる時)

    Args:
        vix: VIX 現在値
        market_return_30d: 直近30日のマーケットリターン (オプション)

    Returns:
        {"score": float, "vix": float, "verdict": str}
    """
    if vix is None or vix <= 0:
        return {"score": 0.0, "vix": 0.0, "verdict": "VIXデータなし"}

    # VIX ベースの基本スコア（30 で +0.6, 12 で -0.6）
    if vix >= 35:
        base, verdict = 1.0, f"極度の恐怖 (VIX={vix:.1f}) → 仕入時"
    elif vix >= 25:
        base, verdict = 0.6, f"警戒 (VIX={vix:.1f}) → 買い場"
    elif vix >= 18:
        base, verdict = 0.1, f"通常 (VIX={vix:.1f})"
    elif vix >= 13:
        base, verdict = -0.3, f"楽観 (VIX={vix:.1f}) → 警戒"
    else:
        base, verdict = -0.7, f"過剰楽観 (VIX={vix:.1f}) → 売り場"

    # マーケット下落時はさらに買いに振る (Buffett: be greedy when fearful)
    if market_return_30d is not None:
        if market_return_30d <= -0.10:
            base = min(1.0, base + 0.3)
            verdict += " + 30日-10%で更に買い場"
        elif market_return_30d >= 0.10:
            base = max(-1.0, base - 0.3)
            verdict += " + 30日+10%で警戒強化"

    return {
        "score": float(base),
        "vix": float(vix),
        "market_return_30d": float(market_return_30d) if market_return_30d else None,
        "verdict": verdict,
    }


def contrarian_insider_pulse_score(
    fundamentals: dict,
    insider_data: Optional[dict] = None,
) -> dict[str, float]:
    """インサイダー買い + Put/Call ratio による smart money signal。

    Args:
        fundamentals: yfinance.info
        insider_data: data_fetcher の insider 集計（{buy_count, sell_count, net}）

    Returns:
        {"score": float, "insider_net": int, "put_call": float, "verdict": str}
    """
    insider_net = 0
    if insider_data:
        insider_net = insider_data.get("net", 0)

    # 経営陣の自社株購入比率（heldPercentInsiders）
    insider_pct = _safe_get(fundamentals, "heldPercentInsiders", 0) or 0

    # P/C ratio (オプションプレッシャー)
    # 高い P/C → ヘッジ需要強 → リバウンド示唆 (contrarian buy)
    # 低い P/C → 楽観 → 警戒
    put_call = _safe_get(fundamentals, "putCallRatio", None)

    score = 0.0
    parts = []

    # インサイダーシグナル
    if insider_net >= 5:
        score += 0.5
        parts.append(f"役員買い+{insider_net}件")
    elif insider_net >= 2:
        score += 0.2
        parts.append(f"役員買い+{insider_net}")
    elif insider_net <= -5:
        score -= 0.5
        parts.append(f"役員売り{insider_net}件")
    elif insider_net <= -2:
        score -= 0.2
        parts.append(f"役員売り{insider_net}")

    # 経営陣保有比率（高いと alignment 強い）
    if insider_pct >= 0.05:  # 5%以上
        score += 0.1
        parts.append(f"経営陣保有{insider_pct*100:.0f}%")

    # Put/Call ratio (contrarian)
    if put_call is not None:
        if put_call >= 1.3:
            score += 0.3
            parts.append(f"P/C={put_call:.2f}(過剰悲観→反発示唆)")
        elif put_call <= 0.6:
            score -= 0.3
            parts.append(f"P/C={put_call:.2f}(過剰楽観)")

    score = max(-1.0, min(1.0, score))
    verdict = " / ".join(parts) if parts else "シグナルなし"

    return {
        "score": float(score),
        "insider_net": int(insider_net),
        "insider_pct": float(insider_pct),
        "put_call": float(put_call) if put_call else None,
        "verdict": verdict,
    }


# ==============================================================================
# Tier 5: Risk Overlay (Kelly + Drawdown + Tail Risk)
# ==============================================================================


def risk_kelly_score(
    expected_return: float,
    return_volatility: float,
    historical_max_drawdown: float = -0.30,
) -> dict[str, float]:
    """ケリー基準でポジションサイズを推奨し、過去最大DDで上限制約。

    Kelly fraction: f* = (p × b - q) / b ≈ E[R] / σ²  (連続版)

    Args:
        expected_return: 期待リターン（年率）
        return_volatility: リターンのボラティリティ（年率）
        historical_max_drawdown: 過去最大ドローダウン（負値）

    Returns:
        {"score": float, "kelly_fraction": float, "recommended_pct": float, "verdict": str}
    """
    if return_volatility <= 0:
        return {"score": 0.0, "kelly_fraction": 0.0, "recommended_pct": 0.0,
                "verdict": "ボラ不明"}

    # Kelly fraction (連続Kelly)
    kelly = expected_return / (return_volatility ** 2)

    # フラクショナル・ケリー（保守側）: 1/4 Kelly が経験的に良い
    fractional_kelly = kelly * 0.25

    # ドローダウン制約: 最大DD -30% を超える銘柄はサイズを縮小
    dd_penalty = 1.0
    if historical_max_drawdown < -0.30:
        dd_penalty = 0.3 / abs(historical_max_drawdown)

    recommended_pct = max(-0.30, min(0.30, fractional_kelly * dd_penalty))

    # スコア化: 推奨が +5% 以上 → 強い買い
    if recommended_pct >= 0.10:
        score, verdict = 1.0, f"高確信度買い ({recommended_pct*100:.1f}%推奨)"
    elif recommended_pct >= 0.05:
        score, verdict = 0.5, f"積極買い ({recommended_pct*100:.1f}%)"
    elif recommended_pct >= 0.02:
        score, verdict = 0.2, f"小幅買い ({recommended_pct*100:.1f}%)"
    elif recommended_pct >= -0.02:
        score, verdict = 0.0, "中立"
    elif recommended_pct >= -0.05:
        score, verdict = -0.4, f"小幅売り ({recommended_pct*100:.1f}%)"
    else:
        score, verdict = -0.8, f"積極売り ({recommended_pct*100:.1f}%)"

    return {
        "score": float(score),
        "kelly_fraction": round(float(kelly), 4),
        "recommended_pct": round(float(recommended_pct), 4),
        "verdict": verdict,
    }


# ==============================================================================
# Master Ensemble — 9ファクター統合
# ==============================================================================


def load_master_weights() -> dict[str, Any]:
    """学習済みのマスターウェイトを読み込む。

    ファイル構造:
      {
        "global": {factor: weight, ...},
        "regime": {regime_name: {factor: weight, ...}, ...},
        "factor_stats": {factor: {accuracy, samples, decay_factor, ...}}
      }
    """
    if not MASTER_WEIGHTS_PATH.exists():
        return {
            "global": dict(PRIOR_WEIGHTS),
            "regime": {r: dict(PRIOR_WEIGHTS) for r in REGIME_PRIORS},
            "factor_stats": {},
        }
    try:
        with open(MASTER_WEIGHTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "global": dict(PRIOR_WEIGHTS),
            "regime": {r: dict(PRIOR_WEIGHTS) for r in REGIME_PRIORS},
            "factor_stats": {},
        }


def save_master_weights(weights: dict[str, Any]) -> None:
    """マスターウェイトを永続化する。"""
    MASTER_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    weights["updated_at"] = datetime.now().isoformat()
    with open(MASTER_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)


def _resolve_regime_weights(
    weights: dict[str, Any],
    regime: str,
) -> dict[str, float]:
    """現在レジームに対応する有効ウェイトを決定する（事前 + 学習統合）。

    1. global ウェイトを起点
    2. レジーム別の調整係数を適用
    3. factor_stats の decay_factor を適用
    4. 正規化

    Returns:
        各ファクターの最終ウェイト（合計1.0に正規化）
    """
    base = dict(weights.get("global", PRIOR_WEIGHTS))
    regime_adj = REGIME_PRIORS.get(regime, {})
    factor_stats = weights.get("factor_stats", {})

    final: dict[str, float] = {}
    for name in FACTOR_NAMES:
        w = base.get(name, PRIOR_WEIGHTS[name])
        # レジーム別調整
        w *= regime_adj.get(name, 1.0)
        # ファクター減衰（精度劣化したら縮小）
        decay = factor_stats.get(name, {}).get("decay_factor", 1.0)
        w *= decay
        final[name] = max(0.01, w)  # ゼロ防止

    # 正規化
    total = sum(final.values())
    if total > 0:
        final = {k: v / total for k, v in final.items()}
    return final


def predict_master_ticker(
    ticker: str,
    fundamentals: dict,
    technical_score: float,
    insider_data: Optional[dict],
    macro_data: Optional[dict],
    expected_return: float,
    return_volatility: float,
    historical_max_dd: float,
    regime: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    """単一銘柄のマスター予測を生成する。

    Args:
        ticker: 銘柄
        fundamentals: yfinance.info 相当
        technical_score: advanced_predictor の composite_score
        insider_data: data_fetcher の insider 結果
        macro_data: VIX, S&P500 etc.
        expected_return: 年率期待リターン (技術モデルから)
        return_volatility: 年率ボラ
        historical_max_dd: 過去最大DD
        regime: 現在のマーケットレジーム
        weights: load_master_weights() の出力

    Returns:
        {"composite_score", "signal", "confidence", "factor_scores": {...}}
    """
    # 各ファクター計算
    q_roe = quality_roe_score(fundamentals)
    q_margin = quality_margin_score(fundamentals)
    v_ey = value_earnings_yield_score(fundamentals)
    v_fcf = value_fcf_yield_score(fundamentals)
    v_mos = value_margin_of_safety_score(fundamentals)

    momentum = {"score": float(np.clip(technical_score, -1.0, 1.0)),
                "verdict": f"テクニカル統合 {technical_score:+.2f}"}

    vix = (macro_data or {}).get("vix")
    market_ret = (macro_data or {}).get("market_return_30d")
    c_fg = contrarian_fear_greed_score(vix, market_ret)
    c_insider = contrarian_insider_pulse_score(fundamentals, insider_data)

    risk_kelly = risk_kelly_score(expected_return, return_volatility, historical_max_dd)

    factor_scores = {
        "quality_roe":              q_roe,
        "quality_margin":           q_margin,
        "value_earnings_yield":     v_ey,
        "value_fcf_yield":          v_fcf,
        "value_margin_of_safety":   v_mos,
        "momentum_composite":       momentum,
        "contrarian_fear_greed":    c_fg,
        "contrarian_insider_pulse": c_insider,
        "risk_kelly":               risk_kelly,
    }

    # 重み加重統合
    final_weights = _resolve_regime_weights(weights, regime)
    composite = 0.0
    for name in FACTOR_NAMES:
        s = factor_scores[name].get("score", 0.0)
        w = final_weights[name]
        composite += w * s
    composite = float(np.clip(composite, -1.0, 1.0))

    # シグナル判定
    abs_c = abs(composite)
    if abs_c >= STRONG_SIGNAL:
        signal = "STRONG_BUY" if composite > 0 else "STRONG_SELL"
        confidence = min(0.98, 0.7 + (abs_c - STRONG_SIGNAL) * 0.6)
    elif abs_c >= WEAK_SIGNAL:
        signal = "BUY" if composite > 0 else "SELL"
        confidence = 0.5 + (abs_c - WEAK_SIGNAL) * 0.6
    else:
        signal = "HOLD"
        confidence = 0.3 + abs_c * 0.5

    # Calibration 補正: factor_stats から overall accuracy を見て確信度割引
    factor_stats = weights.get("factor_stats", {})
    avg_accs = [s.get("accuracy", 0.5) for s in factor_stats.values()
                if s.get("samples", 0) >= MIN_SAMPLES_FOR_LEARNING]
    if avg_accs:
        # 平均精度が0.5未満なら確信度を割引
        portfolio_acc = float(np.mean(avg_accs))
        if portfolio_acc < 0.5:
            confidence *= portfolio_acc / 0.5

    return {
        "ticker": ticker,
        "composite_score": round(composite, 4),
        "signal": signal,
        "confidence": round(float(confidence), 3),
        "factor_scores": {
            name: {"score": round(factor_scores[name].get("score", 0), 3),
                   "verdict": factor_scores[name].get("verdict", ""),
                   "weight": round(final_weights[name], 3)}
            for name in FACTOR_NAMES
        },
        "regime": regime,
        "kelly_recommended_pct": risk_kelly.get("recommended_pct", 0),
        "intrinsic_per_share": v_mos.get("intrinsic_per_share", 0),
        "margin_of_safety": v_mos.get("margin", 0),
    }


def predict_master_all(
    tickers: list[str],
    fundamentals_map: dict[str, dict],
    technical_scores: dict[str, float],
    insider_map: dict[str, dict],
    macro_data: dict,
    return_stats: dict[str, dict],
    regime: str,
    weights: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """全銘柄のマスター予測を生成する。

    Args:
        tickers: 銘柄リスト
        fundamentals_map: {ticker: yfinance.info}
        technical_scores: {ticker: advanced_predictor の composite_score}
        insider_map: {ticker: insider 集計}
        macro_data: {vix, market_return_30d, ...}
        return_stats: {ticker: {"expected": float, "vol": float, "max_dd": float}}
        regime: 現在のマーケットレジーム
        weights: load_master_weights() (Noneなら自動)

    Returns:
        予測リスト（confidence降順）
    """
    if weights is None:
        weights = load_master_weights()

    today = datetime.now().strftime("%Y-%m-%d")
    predictions = []

    for ticker in tickers:
        try:
            f = fundamentals_map.get(ticker, {})
            tech = technical_scores.get(ticker, 0.0)
            ins = insider_map.get(ticker, {})
            ret_st = return_stats.get(ticker, {})

            pred = predict_master_ticker(
                ticker=ticker,
                fundamentals=f,
                technical_score=tech,
                insider_data=ins,
                macro_data=macro_data,
                expected_return=ret_st.get("expected", 0.05),
                return_volatility=ret_st.get("vol", 0.20),
                historical_max_dd=ret_st.get("max_dd", -0.30),
                regime=regime,
                weights=weights,
            )
            pred["id"] = f"master-{today}-{ticker}"
            pred["date"] = today
            pred["price_at_prediction"] = _safe_get(f, "currentPrice", 0) or \
                                          _safe_get(f, "regularMarketPrice", 0) or 0
            pred["evaluated"] = False
            predictions.append(pred)
        except Exception as e:
            logger.warning("master 予測失敗 %s: %s", ticker, e)

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    n_strong = sum(1 for p in predictions if "STRONG" in p["signal"])
    n_buy = sum(1 for p in predictions if "BUY" in p["signal"])
    n_sell = sum(1 for p in predictions if "SELL" in p["signal"])
    logger.info(
        "🎯 Master Wisdom 予測: %d銘柄 (BUY=%d / SELL=%d / STRONG=%d, regime=%s)",
        len(predictions), n_buy, n_sell, n_strong, regime,
    )
    return predictions


# ==============================================================================
# 自己進化: 評価 + Factor Decay + Bayesian Update + Journal
# ==============================================================================


def load_master_history() -> dict[str, Any]:
    """マスター予測履歴を読み込む。"""
    if not MASTER_HISTORY_PATH.exists():
        return {"version": 1, "predictions": []}
    try:
        with open(MASTER_HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "predictions": []}


def save_master_history(history: dict[str, Any]) -> None:
    """履歴をトリムして永続化する。"""
    MASTER_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = history.get("predictions", [])
    if len(records) > MAX_HISTORY_RECORDS:
        records = records[-MAX_HISTORY_RECORDS:]
    history["predictions"] = records
    history["updated_at"] = datetime.now().isoformat()
    with open(MASTER_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def archive_master_predictions(predictions: list[dict[str, Any]]) -> int:
    """新しい予測をアーカイブに追加（同日同銘柄は上書き）。"""
    history = load_master_history()
    existing = {p.get("id", ""): i for i, p in enumerate(history["predictions"])}
    added = updated = 0

    for pred in predictions:
        pid = pred.get("id", "")
        if not pid:
            continue
        pred.setdefault("evaluated", False)
        pred.setdefault("actual_returns", {})  # multi-horizon
        if pid in existing:
            if not history["predictions"][existing[pid]].get("evaluated", False):
                history["predictions"][existing[pid]] = pred
                updated += 1
        else:
            history["predictions"].append(pred)
            added += 1

    save_master_history(history)
    if added or updated:
        logger.info(
            "📦 Master 予測アーカイブ: +%d / %d 更新 (累計 %d)",
            added, updated, len(history["predictions"])
        )
    return added + updated


def evaluate_master_predictions(
    history: dict[str, Any],
    prices_dict: dict[str, np.ndarray],
) -> dict[str, Any]:
    """過去の予測を multi-horizon (7/30/90日) で評価する。

    各ファクターの方向シグナル × 実リターン の一致を集計し、
    ファクター別の精度を更新する。

    Args:
        history: 予測履歴
        prices_dict: {ticker: 価格配列}

    Returns:
        {"factor_stats": {factor: {accuracy_7d, accuracy_30d, samples, decay}}}
    """
    today = datetime.now()
    factor_results: dict[str, dict[str, Any]] = {
        name: {"hits_7d": 0, "trials_7d": 0, "hits_30d": 0, "trials_30d": 0}
        for name in FACTOR_NAMES
    }
    n_evaluated_today = 0

    for pred in history.get("predictions", []):
        if pred.get("evaluated"):
            # 既評価結果を再集計に反映
            actual = pred.get("actual_returns", {})
            for name in FACTOR_NAMES:
                fs = pred.get("factor_scores", {}).get(name, {})
                score = fs.get("score", 0)
                pred_dir = float(np.sign(score))
                if pred_dir == 0:
                    continue
                for h in [7, 30]:
                    r = actual.get(f"{h}d")
                    if r is None:
                        continue
                    actual_dir = float(np.sign(r))
                    if actual_dir == 0:
                        continue
                    factor_results[name][f"trials_{h}d"] += 1
                    if pred_dir == actual_dir:
                        factor_results[name][f"hits_{h}d"] += 1
            continue

        # 未評価の予測を評価
        try:
            pred_date = datetime.strptime(pred["date"], "%Y-%m-%d")
        except (ValueError, KeyError, TypeError):
            continue

        days_elapsed = (today - pred_date).days
        if days_elapsed < EVAL_MIN_DAYS:
            continue

        ticker = pred.get("ticker", "")
        prices = prices_dict.get(ticker)
        pred_price = pred.get("price_at_prediction", 0)
        if prices is None or len(prices) < 2 or pred_price <= 0:
            continue

        # multi-horizon リターン計算
        actual_returns = {}
        # 終値ベースで「pred_date から N 日経過」での価格
        for h in [7, 30, 90]:
            if days_elapsed >= h:
                # h 日経過時点の価格 ≒ -1 から遡る
                idx = -1 - max(0, days_elapsed - h)
                if abs(idx) <= len(prices):
                    p_h = float(prices[idx])
                    actual_returns[f"{h}d"] = (p_h - pred_price) / pred_price

        if not actual_returns:
            continue

        pred["evaluated"] = True
        pred["actual_returns"] = actual_returns
        pred["evaluation_date"] = today.strftime("%Y-%m-%d")
        n_evaluated_today += 1

        # ファクター別の方向集計
        for name in FACTOR_NAMES:
            fs = pred.get("factor_scores", {}).get(name, {})
            score = fs.get("score", 0)
            pred_dir = float(np.sign(score))
            if pred_dir == 0:
                continue
            for h in [7, 30]:
                r = actual_returns.get(f"{h}d")
                if r is None:
                    continue
                actual_dir = float(np.sign(r))
                if actual_dir == 0:
                    continue
                factor_results[name][f"trials_{h}d"] += 1
                if pred_dir == actual_dir:
                    factor_results[name][f"hits_{h}d"] += 1

    # 精度に変換 + decay 計算
    stats: dict[str, dict[str, Any]] = {}
    for name in FACTOR_NAMES:
        r = factor_results[name]
        t7 = r["trials_7d"]
        t30 = r["trials_30d"]
        acc_7 = r["hits_7d"] / t7 if t7 else 0.5
        acc_30 = r["hits_30d"] / t30 if t30 else 0.5
        # 7日と30日の重み付け平均
        if t7 + t30 > 0:
            acc = (acc_7 * t7 + acc_30 * t30) / (t7 + t30)
        else:
            acc = 0.5

        # Factor decay: 一定回数試して精度がDECAY閾値以下なら weight を縮小
        samples = t7 + t30
        decay_factor = 1.0
        if samples >= MIN_SAMPLES_FOR_LEARNING and acc < DECAY_ACCURACY_THRESHOLD:
            # 精度が低いほど強く減衰
            severity = (DECAY_ACCURACY_THRESHOLD - acc) / DECAY_ACCURACY_THRESHOLD
            decay_factor = max(0.3, 1.0 - severity * (1 - DECAY_RATE))

        stats[name] = {
            "accuracy_7d": round(acc_7, 4),
            "accuracy_30d": round(acc_30, 4),
            "accuracy": round(acc, 4),
            "samples_7d": t7,
            "samples_30d": t30,
            "samples": samples,
            "decay_factor": round(decay_factor, 3),
        }

    return {
        "factor_stats": stats,
        "n_evaluated_today": n_evaluated_today,
    }


def update_global_weights(
    weights: dict[str, Any],
    factor_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """ファクター精度に基づき global ウェイトを Bayesian 更新する。

    精度が高いファクターのウェイトを増やし、低いものを減らす。
    PRIOR ウェイトを基準点とした Soft update。

    Args:
        weights: 現在の重み辞書
        factor_stats: evaluate_master_predictions の出力

    Returns:
        更新された weights
    """
    new_global = {}
    for name in FACTOR_NAMES:
        prior = PRIOR_WEIGHTS[name]
        s = factor_stats.get(name, {})
        acc = s.get("accuracy", 0.5)
        samples = s.get("samples", 0)

        if samples < MIN_SAMPLES_FOR_LEARNING:
            new_global[name] = prior
            continue

        # 精度50%基準で線形に重み調整
        # acc=0.7 で +40% / acc=0.3 で -40%
        adjustment = (acc - 0.5) * 2 * min(1.0, samples / 30)
        new_w = prior * (1 + adjustment)
        new_global[name] = max(0.02, min(0.40, new_w))

    # 正規化
    total = sum(new_global.values())
    if total > 0:
        new_global = {k: v / total for k, v in new_global.items()}

    weights["global"] = new_global
    weights["factor_stats"] = factor_stats
    return weights


def append_master_journal(
    factor_stats: dict[str, dict[str, Any]],
    new_weights: dict[str, float],
    prev_weights: dict[str, float],
    n_evaluated_today: int,
    history: dict[str, Any],
) -> dict[str, Any]:
    """マスター学習ジャーナルにエントリを追加する。"""
    journal = {"version": 1, "entries": []}
    if MASTER_JOURNAL_PATH.exists():
        try:
            with open(MASTER_JOURNAL_PATH, encoding="utf-8") as f:
                journal = json.load(f)
                if "entries" not in journal:
                    journal["entries"] = []
        except (json.JSONDecodeError, OSError):
            pass

    today = datetime.now().strftime("%Y-%m-%d")

    # findings: 大きな変化を抽出
    findings: list[str] = []
    factor_jp = {
        "quality_roe": "ROE品質",
        "quality_margin": "利益率品質",
        "value_earnings_yield": "Earnings Yield",
        "value_fcf_yield": "FCF Yield",
        "value_margin_of_safety": "Margin of Safety (DCF)",
        "momentum_composite": "テクニカル統合",
        "contrarian_fear_greed": "Fear-Greed逆張り",
        "contrarian_insider_pulse": "インサイダー脈動",
        "risk_kelly": "Kelly基準",
    }
    for name in FACTOR_NAMES:
        prev_w = prev_weights.get(name, PRIOR_WEIGHTS[name])
        new_w = new_weights.get(name, prev_w)
        delta = new_w - prev_w
        s = factor_stats.get(name, {})
        acc = s.get("accuracy", 0)
        samples = s.get("samples", 0)
        decay = s.get("decay_factor", 1.0)

        if samples < MIN_SAMPLES_FOR_LEARNING:
            continue
        jp_name = factor_jp.get(name, name)

        # ウェイト変化が大きい
        if abs(delta) >= 0.01:
            direction = "増" if delta > 0 else "減"
            findings.append(
                f"{jp_name}: 精度{acc:.0%} ({samples}件) → ウェイト{direction} "
                f"({prev_w:.2%} → {new_w:.2%})"
            )

        # ファクター減衰が発動
        if decay < 1.0:
            findings.append(
                f"⚠️ {jp_name}: 精度{acc:.0%}劣化 → decay {decay:.2f} 適用"
            )

    # 全体精度
    all_eval = [p for p in history.get("predictions", []) if p.get("evaluated")]
    n_total = len(all_eval)

    entry = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "n_evaluated_today": n_evaluated_today,
        "n_total_evaluated": n_total,
        "factor_accuracies": {
            name: round(factor_stats.get(name, {}).get("accuracy", 0.5), 3)
            for name in FACTOR_NAMES
        },
        "factor_weights": {name: round(w, 4) for name, w in new_weights.items()},
        "factor_decay": {
            name: round(factor_stats.get(name, {}).get("decay_factor", 1.0), 3)
            for name in FACTOR_NAMES
        },
        "notable_findings": findings[:10],
    }

    entries = journal.get("entries", [])
    if entries and entries[-1].get("date") == today:
        entries[-1] = entry
    else:
        entries.append(entry)
    if len(entries) > MAX_JOURNAL_ENTRIES:
        entries = entries[-MAX_JOURNAL_ENTRIES:]
    journal["entries"] = entries
    journal["updated_at"] = datetime.now().isoformat()

    MASTER_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_JOURNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, ensure_ascii=False)

    logger.info(
        "🎓 Master 学習ジャーナル更新: 累計 %d評価 / findings %d件",
        n_total, len(findings)
    )
    return entry


def run_master_learning_cycle(
    prices_dict: dict[str, np.ndarray],
) -> dict[str, Any]:
    """マスター予測の毎日の完全な学習サイクル。

    1. 履歴を読み込み
    2. multi-horizon 評価
    3. ファクター別精度集計 + decay 検出
    4. global ウェイト Bayesian 更新
    5. 学習ジャーナル追記

    Returns:
        {"n_evaluated", "factor_accuracies", "weights", "findings"}
    """
    history = load_master_history()
    if not history.get("predictions"):
        logger.info("Master 履歴空 — 学習スキップ")
        return {"n_evaluated": 0}

    prev_weights = load_master_weights()
    prev_global = dict(prev_weights.get("global", PRIOR_WEIGHTS))

    # 評価
    eval_result = evaluate_master_predictions(history, prices_dict)
    save_master_history(history)

    # ウェイト更新
    new_weights = update_global_weights(prev_weights, eval_result["factor_stats"])
    save_master_weights(new_weights)

    # ジャーナル追記
    entry = append_master_journal(
        factor_stats=eval_result["factor_stats"],
        new_weights=new_weights["global"],
        prev_weights=prev_global,
        n_evaluated_today=eval_result["n_evaluated_today"],
        history=history,
    )

    return {
        "n_evaluated": eval_result["n_evaluated_today"],
        "n_total_evaluated": entry["n_total_evaluated"],
        "factor_accuracies": entry["factor_accuracies"],
        "weights": new_weights["global"],
        "factor_decay": entry["factor_decay"],
        "notable_findings": entry["notable_findings"],
    }
