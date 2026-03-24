# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Evolving Quant Dashboard — 自己進化型ポートフォリオ最適化ダッシュボード
======================================================================
Streamlit + Plotly によるリッチな対話型ダッシュボード。
daily_evolution.py が生成した進化済みパラメータを読み込み、
可視化・メタプロンプト生成・AI推奨記録を行う。

起動: streamlit run evolving_quant_dashboard.py
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# daily_evolution のロジックをインポート
from daily_evolution import (
    load_config,
    fetch_price_data,
    calc_log_returns,
    denoise_covariance,
    detone_covariance,
    nco_optimize,
    compute_ensemble,
    detect_regime,
    exponential_gradient_weights,
    black_litterman,
    load_track_record,
    save_track_record,
    run_daily_evolution,
)

# ==============================================================================
# Streamlit ページ設定
# ==============================================================================

st.set_page_config(
    page_title="🧬 Evolving Quant Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .stMetric { background: linear-gradient(135deg, #e8ecf1 0%, #dfe6ed 100%);
                border-radius: 10px; padding: 15px; border: 1px solid #c5cdd6; color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8ecf1; border-radius: 8px 8px 0 0;
        border: 1px solid #c5cdd6; padding: 10px 20px; color: #1a1a2e;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a7dbd 0%, #6c5ce7 100%); color: #fff;
    }
    div[data-testid="stCodeBlock"] { border: 2px solid #e94560; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# データ読み込み
# ==============================================================================

RESULTS_PATH = Path(__file__).resolve().parent / "latest_evolution_results.json"
TRACK_RECORD_PATH = Path(__file__).resolve().parent / "ai_track_record.json"


@st.cache_data(ttl=600)
def load_latest_results() -> dict | None:
    """最新の進化結果JSONを読み込む。"""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def get_price_data(tickers_tuple: tuple) -> pd.DataFrame:
    """
    価格データを取得する（Streamlitキャッシュ対応）。

    Args:
        tickers_tuple: 銘柄のタプル（hashable化）

    Returns:
        日次終値DataFrame
    """
    return fetch_price_data(list(tickers_tuple))


# ==============================================================================
# サイドバー
# ==============================================================================


def render_sidebar() -> dict:
    """
    サイドバーUIを描画し、ユーザー入力を返す。

    Returns:
        サイドバーの入力値辞書
    """
    st.sidebar.title("🧬 Evolving Quant")
    st.sidebar.markdown("---")

    # 進化エンジン実行ボタン
    if st.sidebar.button("🔄 今すぐ進化実行", use_container_width=True, type="primary"):
        with st.spinner("🧬 Daily Evolution Engine 実行中..."):
            results = run_daily_evolution()
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")

    # ポートフォリオ情報
    config = load_config()
    holdings = config.get("portfolio", {}).get("holdings", [])
    st.sidebar.subheader("📊 保有銘柄")
    for h in holdings:
        st.sidebar.text(f"  {h['ticker']:8s} {h['shares']:>4d}株  [{h['sector']}]")

    # トラックレコード統計
    track = load_track_record()
    n_records = len(track.get("records", []))
    n_evals = len(track.get("evaluations", []))
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 学習状態")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("予測記録", f"{n_records}件")
    col2.metric("採点済み", f"{n_evals}件")

    # Ensemble 重み表示
    ew = track["meta"]["evolution_state"].get(
        "ensemble_weights", [0.25, 0.25, 0.25, 0.25]
    )
    st.sidebar.markdown("**Ensemble 重み（進化済み）:**")
    strategy_names = ["NCO", "RiskParity", "MinVar", "MaxDiv"]
    for name, w in zip(strategy_names, ew):
        st.sidebar.progress(w, text=f"{name}: {w:.1%}")

    return {"config": config, "holdings": holdings, "track": track}


# ==============================================================================
# Tab 1: Portfolio Overview
# ==============================================================================


def render_overview_tab(results: dict, sidebar_data: dict) -> None:
    """配分比較の概要タブを描画する。"""
    st.header("📊 Portfolio Allocation Overview")

    allocations = results.get("allocations", {})
    tickers = results.get("tickers", [])
    regime = results.get("regime", "unknown")
    kl_value = results.get("kl_value", 0)

    # レジーム表示
    regime_emoji = {"low_vol": "🟢", "transition": "🟡", "crisis": "🔴"}.get(
        regime, "⚪"
    )
    regime_label = {"low_vol": "Low Volatility", "transition": "Transition", "crisis": "Crisis"}.get(
        regime, "Unknown"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Market Regime", f"{regime_emoji} {regime_label}")
    col2.metric("KL Divergence", f"{kl_value:.4f}")
    col3.metric("Evolution Generation", f"Gen {len(sidebar_data['track'].get('evaluations', []))}")

    st.markdown("---")

    # 配分比較チャート
    ensemble = allocations.get("ensemble", {})
    bl = allocations.get("bl", {})
    eg = allocations.get("eg", {})

    # 現在の均等配分（ベースライン）
    equal_w = 1.0 / len(tickers) if tickers else 0

    df_alloc = pd.DataFrame(
        {
            "Ticker": tickers,
            "Equal Weight": [equal_w] * len(tickers),
            "Ensemble (進化済み)": [ensemble.get(t, 0) for t in tickers],
            "Black-Litterman": [bl.get(t, 0) for t in tickers],
            "Exp. Gradient": [eg.get(t, 0) for t in tickers],
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        # サンバーストチャート
        fig_sun = px.sunburst(
            pd.DataFrame(
                [
                    {"Strategy": "Ensemble", "Ticker": t, "Weight": ensemble.get(t, 0)}
                    for t in tickers
                ]
                + [
                    {"Strategy": "B-L", "Ticker": t, "Weight": bl.get(t, 0)}
                    for t in tickers
                ]
            ),
            path=["Strategy", "Ticker"],
            values="Weight",
            color="Weight",
            color_continuous_scale="Viridis",
            title="Strategy × Ticker Allocation",
        )
        fig_sun.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(t=50, l=10, r=10, b=10),
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    with col2:
        # グループ棒グラフ
        df_melt = df_alloc.melt(
            id_vars="Ticker", var_name="Strategy", value_name="Weight"
        )
        fig_bar = px.bar(
            df_melt,
            x="Ticker",
            y="Weight",
            color="Strategy",
            barmode="group",
            title="Allocation Comparison",
            color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
        )
        fig_bar.update_layout(
            template="plotly_white",
            height=450,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 個別戦略の詳細テーブル
    with st.expander("📋 個別戦略の配分詳細"):
        strategies = allocations.get("strategies", {})
        rows = []
        for t in tickers:
            row = {"Ticker": t}
            for s_name, s_alloc in strategies.items():
                row[s_name] = f"{s_alloc.get(t, 0):.1%}"
            row["Ensemble"] = f"{ensemble.get(t, 0):.1%}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================================================================
# Tab 2: Performance
# ==============================================================================


def render_performance_tab(results: dict, sidebar_data: dict) -> None:
    """パフォーマンス比較タブを描画する。"""
    st.header("📈 パフォーマンス比較")
    st.caption("ポートフォリオ戦略ごとの累積リターンとリスク指標を確認できます。")

    tickers = results.get("tickers", [])
    if not tickers:
        st.warning("銘柄データがありません。")
        return

    prices = get_price_data(tuple(tickers))
    returns = calc_log_returns(prices[[t for t in tickers if t in prices.columns]])

    # --- 期間セレクター ---
    period_options = {"1ヶ月": 21, "3ヶ月": 63, "6ヶ月": 126, "1年": 252, "2年": 504}
    selected_period = st.radio(
        "📅 表示期間",
        list(period_options.keys()),
        index=3,
        horizontal=True,
        key="perf_period",
    )
    n_days = period_options[selected_period]

    # 期間でスライス（データが足りない場合は全期間）
    if len(returns) > n_days:
        returns_period = returns.iloc[-n_days:]
    else:
        returns_period = returns

    # 累積リターン
    cumulative = (1 + returns_period).cumprod()

    # ポートフォリオリターン（各戦略）
    allocations = results.get("allocations", {})
    ensemble = allocations.get("ensemble", {})
    available = [t for t in tickers if t in returns_period.columns]

    fig = go.Figure()

    # 個別銘柄
    for t in available:
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[t],
                name=t,
                opacity=0.4,
                line=dict(width=1),
            )
        )

    # Ensembleポートフォリオ
    w_ensemble = np.array([ensemble.get(t, 0) for t in available])
    port_ret = (returns_period[available].values @ w_ensemble)
    port_cumul = pd.Series(np.cumprod(1 + port_ret), index=returns_period.index)
    fig.add_trace(
        go.Scatter(
            x=port_cumul.index,
            y=port_cumul.values,
            name="🧬 Ensemble Portfolio",
            line=dict(color="#e94560", width=3),
        )
    )

    # 均等配分ポートフォリオ
    w_equal = np.ones(len(available)) / len(available)
    equal_ret = (returns_period[available].values @ w_equal)
    equal_cumul = pd.Series(np.cumprod(1 + equal_ret), index=returns_period.index)
    fig.add_trace(
        go.Scatter(
            x=equal_cumul.index,
            y=equal_cumul.values,
            name="📐 Equal Weight",
            line=dict(color="#636EFA", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"累積リターン（{selected_period}）",
        xaxis_title="日付",
        yaxis_title="累積リターン（1.0 = 変化なし）",
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- リスク指標（期間ベース） ---
    st.subheader("📊 リスク指標")

    port_daily = pd.Series(port_ret, index=returns_period.index)
    ann_return = float(port_daily.mean() * 252)
    ann_vol = float(port_daily.std() * np.sqrt(252))
    sharpe = ann_return / (ann_vol + 1e-10)
    max_dd = float((port_cumul / port_cumul.cummax() - 1).min())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("年率リターン", f"{ann_return:.1%}")
    col2.metric("年率ボラティリティ", f"{ann_vol:.1%}")
    col3.metric("シャープレシオ", f"{sharpe:.2f}")
    col4.metric("最大ドローダウン", f"{max_dd:.1%}")

    # --- 各指標の解説 ---
    with st.expander("💡 各指標の意味（解説を開く）", expanded=False):
        st.markdown("""
| 指標 | 意味 | 目安 |
|---|---|---|
| **年率リターン** | 1年間でポートフォリオ全体がどれだけ増えるかの推定値。過去データから年率換算。 | **10%以上**で優秀 |
| **年率ボラティリティ** | 価格がどれだけ上下にブレるかの指標。高いほどリスク（＝値動き）が大きい。 | **15%以下**が安定的 |
| **シャープレシオ** | 「リスク1単位あたりどれだけリターンが得られるか」の効率指標。高いほど効率的。 | **1.0以上**が良好、**2.0以上**で優秀 |
| **最大ドローダウン** | 過去の最高値から最も落ちた割合。「最悪の場合、一時的にどれだけ下がるか」を示す。 | **-20%以内**が防御的 |
        """)


# ==============================================================================
# Tab 3: Risk Analysis
# ==============================================================================


def render_risk_tab(results: dict) -> None:
    """リスク分析タブを描画する。"""
    st.header("🔥 Risk Analysis")

    tickers = results.get("tickers", [])
    if not tickers:
        return

    prices = get_price_data(tuple(tickers))
    available = [t for t in tickers if t in prices.columns]
    returns = calc_log_returns(prices[available])

    col1, col2 = st.columns(2)

    with col1:
        # 相関ヒートマップ
        corr = returns.corr()
        fig_corr = px.imshow(
            corr,
            x=available,
            y=available,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix (Denoised)",
            aspect="auto",
        )
        fig_corr.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        # リスク寄与度
        ensemble = results.get("allocations", {}).get("ensemble", {})
        w = np.array([ensemble.get(t, 0) for t in available])
        cov = returns.cov().values
        port_var = w @ cov @ w
        risk_contrib = w * (cov @ w) / (port_var + 1e-10)

        fig_risk = go.Figure(
            go.Bar(
                x=available,
                y=risk_contrib,
                marker_color=["#e94560" if rc > 1 / len(available) else "#00CC96" for rc in risk_contrib],
                text=[f"{rc:.1%}" for rc in risk_contrib],
                textposition="outside",
            )
        )
        fig_risk.update_layout(
            template="plotly_white",
            title="Risk Contribution (Ensemble)",
            yaxis_tickformat=".0%",
            height=400,
            yaxis_title="Risk Contribution",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ローリングボラティリティ
    st.subheader("📉 Rolling Volatility (21-day)")
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    fig_vol = go.Figure()
    for t in available:
        fig_vol.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t, line=dict(width=1.5))
        )
    fig_vol.update_layout(
        template="plotly_white", height=350, yaxis_tickformat=".0%", yaxis_title="Annualized Vol"
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ==============================================================================
# Tab 4: AI Report Card
# ==============================================================================


def render_ai_report_tab(track: dict) -> None:
    """AI成績表タブを描画する。"""
    st.header("🎯 AI Report Card — 自己進化の成績表")

    evaluations = track.get("evaluations", [])

    if not evaluations:
        st.info("📭 まだ評価データがありません。AI推奨を記録し、7日以上経過すると自動採点されます。")
        return

    # 精度推移チャート
    eval_dates = [e.get("eval_date", "") for e in evaluations]
    rmses = [e.get("predicted_vs_actual", {}).get("rmse", 0) for e in evaluations]
    dir_accs = [e.get("predicted_vs_actual", {}).get("direction_accuracy", 0) for e in evaluations]
    cal_scores = [e.get("predicted_vs_actual", {}).get("calibration_score", 0) for e in evaluations]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Prediction RMSE", "Direction Accuracy & Calibration"])

    fig.add_trace(
        go.Scatter(x=eval_dates, y=rmses, name="RMSE", line=dict(color="#e94560", width=2), fill="tozeroy"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=eval_dates, y=dir_accs, name="Direction Accuracy", line=dict(color="#00CC96", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=eval_dates, y=cal_scores, name="Calibration Score", line=dict(color="#AB63FA", width=2)),
        row=2, col=1,
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 誤差因子分解
    st.subheader("🔍 Error Decomposition (因子分解)")
    decomp_data = []
    for e in evaluations:
        decomp = e.get("error_decomposition", {})
        decomp_data.append({
            "Date": e.get("eval_date", ""),
            "Market Timing": decomp.get("market_timing", 0),
            "Sector Rotation": decomp.get("sector_rotation", 0),
            "Idiosyncratic": decomp.get("idiosyncratic", 0),
        })
    if decomp_data:
        df_decomp = pd.DataFrame(decomp_data)
        fig_decomp = px.bar(
            df_decomp.melt(id_vars="Date", var_name="Factor", value_name="Error"),
            x="Date", y="Error", color="Factor", barmode="stack",
            color_discrete_sequence=["#EF553B", "#FFA15A", "#636EFA"],
            title="Prediction Error Decomposition",
        )
        fig_decomp.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_decomp, use_container_width=True)

    # バイアス分析
    st.subheader("🧠 Bias Analysis")
    for e in evaluations[-5:]:
        bias = e.get("bias_analysis", "データなし")
        st.text(f"  [{e.get('eval_date', '')}] {bias}")


# ==============================================================================
# Tab 5: Meta-Prompt Generator
# ==============================================================================


def generate_meta_prompt(results: dict, track: dict) -> str:
    """
    Claude Opus用のメタプロンプトを生成する。

    全データを統合した自己進化型プロンプト。
    Few-Shot例、Devil's Advocate、Chain-of-Thought指示を含む。

    Args:
        results: 最新の進化結果
        track: トラックレコード

    Returns:
        プロンプト文字列
    """
    tickers = results.get("tickers", [])
    allocations = results.get("allocations", {})
    regime = results.get("regime", "unknown")
    kl_value = results.get("kl_value", 0)
    ensemble = allocations.get("ensemble", {})
    bl = allocations.get("bl", {})
    strategies = allocations.get("strategies", {})
    ew = results.get("ensemble_weights", [])
    evaluations = track.get("evaluations", [])

    regime_jp = {"low_vol": "低ボラティリティ（安定上昇局面）", "transition": "移行期（ボラ上昇中）", "crisis": "危機（高ボラ・急落局面）"}.get(regime, "不明")

    # --- SECTION A ---
    section_a = "═══ SECTION A: 数学的ベースライン（Denoised NCO + 4戦略アンサンブル） ═══\n\n"
    section_a += f"■ 現在のマーケットレジーム: {regime_jp} (KL divergence = {kl_value:.4f})\n\n"
    section_a += "■ Ensemble最適配分（NCO/RiskParity/MinVar/MaxDiv の重み付け統合）:\n"
    for t in tickers:
        section_a += f"  - {t}: {ensemble.get(t, 0):.1%}\n"
    section_a += f"\n  Ensemble重み: NCO={ew[0]:.1%}, RP={ew[1]:.1%}, MV={ew[2]:.1%}, MD={ew[3]:.1%}\n"
    section_a += "\n■ Black-Litterman配分（AI信頼度で加重した均衡リターン融合）:\n"
    for t in tickers:
        section_a += f"  - {t}: {bl.get(t, 0):.1%}\n"

    # --- SECTION B ---
    section_b = "\n═══ SECTION B: 自己反省データ（成績表 - Self-Reflection Report Card） ═══\n\n"
    if evaluations:
        latest_evals = evaluations[-5:]
        avg_rmse = np.mean([e.get("predicted_vs_actual", {}).get("rmse", 0) for e in latest_evals])
        avg_dir = np.mean([e.get("predicted_vs_actual", {}).get("direction_accuracy", 0) for e in latest_evals])
        avg_cal = np.mean([e.get("predicted_vs_actual", {}).get("calibration_score", 0) for e in latest_evals])

        section_b += f"■ 直近{len(latest_evals)}回の予測精度サマリー:\n"
        section_b += f"  - 平均RMSE: {avg_rmse:.4f}\n"
        section_b += f"  - 方向性的中率: {avg_dir:.1%}\n"
        section_b += f"  - 較正スコア: {avg_cal:.2f} （1.0=完全較正、<0.5=過信バイアス）\n\n"

        section_b += "■ 誤差因子分解（平均）:\n"
        avg_timing = np.mean([e.get("error_decomposition", {}).get("market_timing", 0) for e in latest_evals])
        avg_sector = np.mean([e.get("error_decomposition", {}).get("sector_rotation", 0) for e in latest_evals])
        avg_idio = np.mean([e.get("error_decomposition", {}).get("idiosyncratic", 0) for e in latest_evals])
        section_b += f"  - Market Timing Error: {avg_timing:+.4f} {'（市場の方向を読み誤る傾向）' if abs(avg_timing) > 0.01 else ''}\n"
        section_b += f"  - Sector Rotation Error: {avg_sector:.4f} {'（セクター配分判断に改善余地）' if avg_sector > 0.02 else ''}\n"
        section_b += f"  - Idiosyncratic Error: {avg_idio:.4f}\n\n"

        section_b += "■ バイアス分析:\n"
        for e in latest_evals:
            section_b += f"  [{e.get('eval_date', '')}] {e.get('bias_analysis', 'N/A')}\n"
    else:
        section_b += "（まだ評価データがありません。初回記録後、7日以上で自動採点開始。）\n"

    # --- SECTION C ---
    section_c = "\n═══ SECTION C: Few-Shot Examples（過去の予測-結果ペア） ═══\n\n"
    records = track.get("records", [])
    matched_examples = []
    for ev in evaluations[-5:]:
        record = next((r for r in records if r.get("id") == ev.get("record_id")), None)
        if record:
            matched_examples.append((record, ev))

    if matched_examples:
        for i, (rec, ev) in enumerate(matched_examples, 1):
            section_c += f"--- Example {i} ---\n"
            section_c += f"  予測日: {rec.get('date', '')}\n"
            section_c += f"  AI配分: {json.dumps(rec.get('ai_allocation', {}), indent=None)}\n"
            section_c += f"  確信度: {rec.get('confidence', 'N/A')}\n"
            section_c += f"  理由: {rec.get('ai_reasoning', 'N/A')}\n"
            section_c += f"  実際のリターン: {json.dumps(ev.get('actual_returns', {}), indent=None)}\n"
            section_c += f"  RMSE: {ev.get('predicted_vs_actual', {}).get('rmse', 'N/A')}\n"
            section_c += f"  教訓: {ev.get('bias_analysis', 'N/A')}\n\n"
    else:
        section_c += "（まだFew-Shot例がありません。予測を記録してください。）\n"

    # --- SECTION D ---
    section_d = """
═══ SECTION D: Devil's Advocate（反証思考） ═══

以下の手順で推論してください:
1. まず、あなたの直感的な最初のリバランス提案を述べてください。
2. 次に、その提案に対する【最も強力な反論を3つ】挙げてください。
   - 反論1: マクロ経済リスクの観点から
   - 反論2: テクニカル/モメンタムの観点から
   - 反論3: あなた自身の過去のバイアスパターンの観点から
3. 各反論に対して再反論し、最終的な結論を導いてください。
"""

    # --- SECTION E ---
    section_e = f"""
═══ SECTION E: 推論指示（Structured Output） ═══

あなたはプロのクオンツ・アナリストです。以下の形式で回答してください:

【マクロ環境分析】（3行以内）
現在のマクロ経済環境の要点。

【自己バイアス分析】（3行以内）
SECTION B の成績表を踏まえ、あなたの過去の予測バイアスを分析。

【リバランス提案】
各銘柄の配分を小数点で提示（合計1.0）。
{chr(10).join(f'  {t}: [配分]' for t in tickers)}
  理由: [簡潔に]

【確信度】
0.0〜1.0 の数値。較正スコアを意識し、過信しないこと。

【リスクシナリオ】
- Bull case (確率 %): ...
- Base case (確率 %): ...
- Bear case (確率 %): ...
"""

    # 統合
    today = datetime.now().strftime("%Y-%m-%d")
    header = f"""╔══════════════════════════════════════════════════════════════╗
║  🧬 EVOLVING QUANT META-PROMPT  —  Generated: {today}      ║
║  Self-Reflection Generation: {len(evaluations):3d}                              ║
╚══════════════════════════════════════════════════════════════╝

あなたは自己学習型クオンツAIです。以下のデータには、あなた自身の「過去の予測」と
「実際の市場の動き」の比較（成績表）が含まれています。
この成績表を真摯に受け止め、自分のバイアスを修正した上で推論してください。

"""

    return header + section_a + section_b + section_c + section_d + section_e


def render_prompt_tab(results: dict, track: dict) -> None:
    """メタプロンプト生成タブを描画する。"""
    st.header("🧬 Meta-Prompt Generator")
    st.markdown(
        "以下のプロンプトをコピーして **Claude Opus** に渡してください。"
        "AIの過去の予測精度を含むフィードバックループが組み込まれています。"
    )

    prompt = generate_meta_prompt(results, track)
    st.code(prompt, language="text")

    st.download_button(
        "📋 プロンプトをダウンロード",
        prompt,
        file_name=f"meta_prompt_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )


# ==============================================================================
# Tab 6: AI推奨記録
# ==============================================================================


def render_record_tab(track: dict) -> None:
    """AI推奨の記録タブを描画する。"""
    st.header("📝 AI推奨を記録")
    st.markdown(
        "Claude Opusから受け取った推奨配分をここに入力してください。"
        "次回の進化サイクルで自動採点されます。"
    )

    config = load_config()
    holdings = config.get("portfolio", {}).get("holdings", [])
    tickers = [h["ticker"] for h in holdings]

    with st.form("record_form"):
        st.subheader("配分入力")
        alloc = {}
        cols = st.columns(min(len(tickers), 3))
        for i, ticker in enumerate(tickers):
            with cols[i % len(cols)]:
                alloc[ticker] = st.number_input(
                    f"{ticker} 配分",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / len(tickers), 2),
                    step=0.01,
                    format="%.2f",
                )

        confidence = st.slider("確信度", 0.0, 1.0, 0.5, 0.05)
        reasoning = st.text_area("AIの推奨理由（要約）", height=100)

        submitted = st.form_submit_button("💾 記録する", type="primary")

        if submitted:
            # 合計チェック
            total = sum(alloc.values())
            if abs(total - 1.0) > 0.05:
                st.error(f"配分の合計が {total:.2f} です。1.0に近づけてください。")
            else:
                # 正規化
                alloc = {k: v / total for k, v in alloc.items()}
                new_record = {
                    "id": str(uuid.uuid4())[:8],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "ai_allocation": alloc,
                    "ai_reasoning": reasoning,
                    "confidence": confidence,
                    "evaluated": False,
                    "evaluation": None,
                }
                track["records"].append(new_record)
                track["meta"]["total_predictions"] = len(track["records"])
                save_track_record(track)
                st.success("✅ 記録しました！次回の進化サイクルで自動採点されます。")
                st.balloons()

    # 過去の記録表示
    if track.get("records"):
        st.markdown("---")
        st.subheader("📜 過去の記録")
        for r in reversed(track["records"][-10:]):
            status = "✅ 採点済み" if r.get("evaluated") else "⏳ 評価待ち"
            with st.expander(f"[{r.get('date', '')}] {status} — 確信度 {r.get('confidence', 'N/A')}"):
                st.json(r.get("ai_allocation", {}))
                st.text(f"理由: {r.get('ai_reasoning', 'N/A')}")


# ==============================================================================
# メインアプリ
# ==============================================================================


def main() -> None:
    """Streamlitアプリのメインエントリーポイント。"""
    st.title("🧬 Evolving Quant Dashboard")
    st.caption("Self-Reflective Portfolio Optimization — 使うほどに賢くなるAIクオンツ")

    sidebar_data = render_sidebar()
    track = sidebar_data["track"]

    # 最新結果の読み込み
    results = load_latest_results()

    if results is None:
        st.warning("⚠️ まだ進化結果がありません。サイドバーの「🔄 今すぐ進化実行」を押してください。")
        # 初回実行でも最低限のUIを表示
        render_record_tab(track)
        return

    # タブ構成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Overview", "📈 Performance", "🔥 Risk", "🎯 AI Report Card", "🧬 Meta-Prompt", "📝 Record"]
    )

    with tab1:
        render_overview_tab(results, sidebar_data)
    with tab2:
        render_performance_tab(results, sidebar_data)
    with tab3:
        render_risk_tab(results)
    with tab4:
        render_ai_report_tab(track)
    with tab5:
        render_prompt_tab(results, track)
    with tab6:
        render_record_tab(track)


if __name__ == "__main__":
    main()
