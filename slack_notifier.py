#!/usr/bin/env python3
"""
slack_notifier.py — 週次クオンツレポートをSlack DMで送信

毎週土曜に実行され、ポートフォリオのアクション・レポート +
高精度予測アンサンブルの結果を Slack DM で送信する。

使用:
    python slack_notifier.py            # 実送信
    python slack_notifier.py --dry-run  # プレビューのみ
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("slack-notifier")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "latest_evolution_results.json"
TRACK_RECORD_PATH = SCRIPT_DIR / "ai_track_record.json"
HOLDINGS_PATH = SCRIPT_DIR / "data" / "portfolio_holdings.json"

AKI_USER_ID = "U7DSASM1A"
JST = timezone(timedelta(hours=9))
DASHBOARD_URL = "https://akihamada.github.io/evolving-quant/"


def load_json(path: Path) -> dict[str, Any]:
    """JSONファイルを安全に読み込む。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("JSON読み込み失敗: %s (%s)", path.name, e)
        return {}


# ==============================================================================
# 高精度予測 Advanced AI セクション
# ==============================================================================


SIGNAL_EMOJI = {
    "STRONG_BUY":  "🟢🟢",
    "BUY":         "🟢",
    "HOLD":        "🟡",
    "SELL":        "🔴",
    "STRONG_SELL": "🔴🔴",
}


def _format_sub_scores(sub: dict[str, float]) -> str:
    """5つのサブ予測器スコアを1行テキスト化する。

    Args:
        sub: {"kalman_trend": float, ...}

    Returns:
        "Kalman +0.80 / Hurst +0.60 / C-Sect +0.90 / Vol +0.50 / M-Rev -0.20"
    """
    labels = [
        ("kalman_trend",    "Kalman"),
        ("hurst_regime",    "Hurst"),
        ("cross_sectional", "C-Sect"),
        ("vol_regime",      "Vol"),
        ("mean_reversion",  "M-Rev"),
    ]
    parts = []
    for key, label in labels:
        v = sub.get(key, 0)
        parts.append(f"{label} {v:+.2f}")
    return " / ".join(parts)


def build_advanced_section(advanced: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """高精度予測の Slack Block Kit セクションを構築する。

    高確信度 (>=0.6) かつ HOLD 以外のシグナルのみ表示。

    Args:
        advanced: results["advanced_signals"]

    Returns:
        Block Kit ブロックのリスト（該当なければ空）
    """
    if not advanced:
        return []

    # 高確信度・非HOLDのみ抽出、確信度降順
    visible = [
        s for s in advanced
        if s.get("signal") != "HOLD" and s.get("confidence", 0) >= 0.6
    ]
    visible = sorted(visible, key=lambda x: x.get("confidence", 0), reverse=True)[:8]

    if not visible:
        return []

    blocks: list[dict[str, Any]] = []
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*🔮 高精度予測アンサンブル（5モデル統合）*\n"
                    f"_Kalman Trend / Hurst Regime / Cross-Sectional / Vol Regime / Mean Reversion_"
        },
    })

    lines = []
    for s in visible:
        ticker = s.get("ticker", "?")
        signal = s.get("signal", "HOLD")
        emoji = SIGNAL_EMOJI.get(signal, "⚪")
        conf = s.get("confidence", 0)
        score = s.get("composite_score", 0)
        sub_line = _format_sub_scores(s.get("sub_scores", {}))
        exp_ret = s.get("expected_return_7d", 0)
        ret_pct = exp_ret * 100

        lines.append(
            f"  {emoji} *{ticker}* — `{signal}` 合成 *{score:+.3f}* / 確信度 *{conf:.0%}* / "
            f"7日期待 {ret_pct:+.2f}%\n"
            f"     _{sub_line}_"
        )

    # Slack section text は 3000 文字上限
    text = "\n".join(lines)
    if len(text) > 2900:
        text = text[:2900] + "\n…（省略）"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": text},
    })

    return blocks


# ==============================================================================
# タイミング予測 / リバランス / 積立 / 学習ジャーナル セクション
# ==============================================================================


LEARNING_JOURNAL_PATH = SCRIPT_DIR / "data" / "learning_journal.json"


def build_timing_section(results: dict[str, Any]) -> list[dict[str, Any]]:
    """タイミング予測シグナル (price_predictor連携) のセクションを構築する。

    results["timing_signals"] または results["prediction_accuracy"] が
    存在する場合のみレンダリング。

    Args:
        results: latest_evolution_results.json

    Returns:
        Block Kit ブロックのリスト（該当なければ空）
    """
    signals = results.get("timing_signals", []) or []
    accuracy = results.get("prediction_accuracy", {}) or {}

    active = [
        s for s in signals
        if s.get("signal") and s["signal"] != "HOLD"
        and s.get("confidence", 0) >= 0.6
    ]
    if not active and not accuracy:
        return []

    blocks: list[dict[str, Any]] = []
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn",
                 "text": "*🎯 タイミング予測シグナル（price_predictor）*"},
    })

    if active:
        active_sorted = sorted(active, key=lambda x: x.get("confidence", 0), reverse=True)[:6]
        lines = []
        for s in active_sorted:
            sig = s.get("signal", "HOLD")
            emoji = "🔴" if sig == "SELL_HIGH" else "🟢" if sig == "BUY_LOW" else "🟡"
            triggers = ", ".join(s.get("triggers", [])[:3])
            lines.append(
                f"  {emoji} *{s.get('ticker', '?')}* `{sig}` "
                f"確信度 {s.get('confidence', 0):.0%}\n"
                f"     _{triggers}_"
            )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })

    if accuracy.get("total_predictions", 0) > 0:
        overall = accuracy.get("overall_accuracy", 0)
        buy_acc = accuracy.get("buy_accuracy", 0)
        sell_acc = accuracy.get("sell_accuracy", 0)
        trend = accuracy.get("trend", "stable")
        trend_emoji = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(trend, "➡️")
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"📊 予測精度: 全体 {overall:.0%} / "
                        f"BUY {buy_acc:.0%} / SELL {sell_acc:.0%} / "
                        f"トレンド {trend_emoji} {trend}"
            }],
        })

    return blocks


def build_rebalance_section(
    results: dict[str, Any],
    latest_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """リバランス提案 + マルチTFスコア + AIオピニオンのセクションを構築する。"""
    tf_scores = results.get("multi_tf_scores", {}) or {}
    proposals = results.get("rebalance_proposals", []) or []
    opinion = latest_record.get("rebalance_opinion", {}) or {}

    if not tf_scores and not proposals and not opinion:
        return []

    blocks: list[dict[str, Any]] = []

    if tf_scores:
        sorted_tf = sorted(
            tf_scores.items(),
            key=lambda x: x[1].get("composite", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        bulls = [k for k, v in sorted_tf
                 if isinstance(v, dict) and v.get("composite", 0) > 0.5][:3]
        bears = [k for k, v in sorted_tf[::-1]
                 if isinstance(v, dict) and v.get("composite", 0) < -0.5][:3]
        tf_lines = []
        if bulls:
            tf_lines.append(f"  🟢 強気トレンド: {', '.join(bulls)}")
        if bears:
            tf_lines.append(f"  🔴 弱気トレンド: {', '.join(bears)}")
        if tf_lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": "*📈 マルチTF モメンタム*\n" + "\n".join(tf_lines)},
            })

    if proposals:
        rb_lines = []
        for p in proposals[:5]:
            ptype = p.get("type", "UNKNOWN")
            urgency = p.get("urgency", "low")
            reasoning = (p.get("reasoning", "") or "")[:80]
            emoji = "🔄" if "ROTATE" in ptype else "✂️" if "TRIM" in ptype else "⚪"
            if urgency in ("immediate", "今すぐ", "high"):
                emoji = "🚨 " + emoji
            rb_lines.append(f"  {emoji} *[{ptype}]* {reasoning} _({urgency})_")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": "*⚖️ リバランスアドバイザー提案*\n" + "\n".join(rb_lines)},
        })

    if opinion:
        agree = opinion.get("agree_with_proposals", True)
        agree_text = "✅ *賛同*" if agree else "❌ *反対 / 独自案*"
        reason = (opinion.get("override_reason", "") or "")[:100]
        swap_text = ""
        swaps = opinion.get("additional_swaps", []) or []
        if swaps:
            swap_lines = [f"    → {s.get('sell', '?')} → {s.get('buy', '?')}: "
                          f"{(s.get('reason', '') or '')[:40]}"
                          for s in swaps[:3]]
            swap_text = "\n" + "\n".join(swap_lines)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"*🧠 AIリバランスオピニオン*\n  {agree_text} — {reason}{swap_text}"},
        })

    return blocks


def build_tsumitate_section(latest_record: dict[str, Any]) -> list[dict[str, Any]]:
    """積立設定レビューのセクションを構築する。"""
    tsumitate = latest_record.get("tsumitate_advice", {}) or {}
    changes = tsumitate.get("changes", []) or []
    if not changes:
        return []

    lines = [f"  → {c[:80]}" for c in changes[:3]]
    reasoning = (tsumitate.get("reasoning", "") or "")[:120]
    text = "*📊 積立設定レビュー*\n" + "\n".join(lines)
    if reasoning:
        text += f"\n  _{reasoning}_"

    return [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]


def build_learning_section() -> list[dict[str, Any]]:
    """学習ジャーナルから直近の学びのセクションを構築する。"""
    if not LEARNING_JOURNAL_PATH.exists():
        return []
    try:
        with open(LEARNING_JOURNAL_PATH, encoding="utf-8") as f:
            journal = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    learnings = journal.get("learnings", []) or []
    stats = journal.get("cumulative_stats", {}) or {}
    if not learnings:
        return []

    latest = learnings[-1]
    lessons = latest.get("lessons", []) or []
    dominant = latest.get("dominant_strategy", "")
    biases = latest.get("persistent_biases", {}) or {}

    if not lessons and not biases:
        return []

    blocks: list[dict[str, Any]] = []
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn",
                 "text": f"*🧬 学習ジャーナル #{stats.get('evolution_count', '?')}*"},
    })

    body_lines = []
    for l in lessons[:3]:
        body_lines.append(f"  💡 {l[:120]}")
    for t, b in list(biases.items())[:3]:
        direction = b.get("direction", "?")
        avg_err = b.get("avg_error_pct", 0)
        body_lines.append(f"  ⚠️ {t}: {direction} ({avg_err:+.1f}%)")
    if dominant:
        body_lines.append(f"  👑 最有効戦略: {dominant}")

    if body_lines:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(body_lines)},
        })

    overall_acc = stats.get("avg_direction_accuracy", 0)
    total_eval = stats.get("total_evaluations", 0)
    if total_eval > 0:
        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"累計 {total_eval}評価 / 平均方向精度 {overall_acc:.0%}"
            }],
        })

    return blocks


# ==============================================================================
# Slack メッセージ構築
# ==============================================================================


def build_slack_blocks(
    results: dict[str, Any],
    track: dict[str, Any],
    holdings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Block Kit 形式の週次レポートメッセージを構築する。

    Args:
        results: latest_evolution_results.json
        track: ai_track_record.json
        holdings: portfolio_holdings.json

    Returns:
        Block Kit ブロックのリスト
    """
    now = datetime.now(tz=JST)
    date_str = now.strftime("%Y-%m-%d (%a)")

    # --- Results ---
    regime_raw = results.get("regime", "unknown")
    regime_map = {
        "transition": "移行期（ボラ上昇中）",
        "low_vol":    "低ボラティリティ（安定）",
        "crisis":     "危機（高ボラ）",
        "bull":       "強気相場",
        "bear":       "弱気相場",
    }
    regime_label = regime_map.get(regime_raw, regime_raw)
    kl_value = results.get("kl_value", 0)
    advanced = results.get("advanced_signals", [])

    # --- Track record ---
    records = track.get("records", [])
    latest = records[-1] if records else {}
    confidence = latest.get("confidence", 0)
    actions = latest.get("actions", {})
    strategy = latest.get("ai_reasoning", "")
    risk_scenarios = latest.get("risk_scenarios", {})
    new_picks = latest.get("new_picks", [])
    action_reasons = latest.get("action_reasons", {})

    # --- Holdings ---
    us_summary = holdings.get("us_stocks", {}).get("summary", {})
    total_value = us_summary.get("total_market_value", 0)
    total_pnl = us_summary.get("total_unrealized_pnl", 0)
    cost_basis = total_value - total_pnl if total_value > total_pnl else 1
    pnl_pct = (total_pnl / cost_basis * 100) if cost_basis > 0 else 0
    num_stocks = us_summary.get("total_holdings", 0)

    # --- Actions categorize ---
    bl_alloc = results.get("allocations", {}).get("bl", {})
    buys = [t for t, a in actions.items() if a == "BUY"]
    sells = [t for t, a in actions.items() if a == "SELL"]
    holds = [t for t, a in actions.items() if a == "HOLD"]
    buys_sorted = sorted(buys, key=lambda t: bl_alloc.get(t, 0), reverse=True)

    MONTHLY_BUDGET = 200_000
    total_w = sum(bl_alloc.get(t, 0) for t in buys_sorted) or 1.0
    buy_amounts = {
        t: int(bl_alloc.get(t, 0) / total_w * MONTHLY_BUDGET) for t in buys_sorted
    }

    blocks: list[dict[str, Any]] = []

    # Header
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text",
                 "text": f"🧬 週次クオンツレポート  {date_str}",
                 "emoji": True},
    })

    # Portfolio overview
    pnl_sign = "+" if total_pnl >= 0 else ""
    blocks.append({
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*💰 総額*\n${total_value:,.0f} ({num_stocks}銘柄)"},
            {"type": "mrkdwn", "text": f"*📈 含み損益*\n{pnl_sign}${total_pnl:,.0f} ({pnl_sign}{pnl_pct:.1f}%)"},
            {"type": "mrkdwn", "text": f"*🎯 レジーム*\n{regime_label} (KL={kl_value:.3f})"},
            {"type": "mrkdwn", "text": f"*🧠 AI確信度*\n{confidence:.0%}"},
        ],
    })

    blocks.append({"type": "divider"})

    # Strategy summary
    if strategy:
        s_trunc = strategy[:280] + "…" if len(strategy) > 280 else strategy
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📋 戦略サマリー*\n{s_trunc}"},
        })

    # === Advanced AI 予測（5モデル統合） ===
    adv_blocks = build_advanced_section(advanced)
    if adv_blocks:
        blocks.extend(adv_blocks)
        blocks.append({"type": "divider"})

    # === タイミング予測シグナル（price_predictor） ===
    timing_blocks = build_timing_section(results)
    if timing_blocks:
        blocks.extend(timing_blocks)
        blocks.append({"type": "divider"})

    # === リバランス提案 + AIオピニオン ===
    rb_blocks = build_rebalance_section(results, latest)
    if rb_blocks:
        blocks.extend(rb_blocks)
        blocks.append({"type": "divider"})

    # BUY 優先順位
    if buys_sorted:
        lines = [f"*🟢 BUY — 優先順（月予算 ¥{MONTHLY_BUDGET:,}）*"]
        for rank, t in enumerate(buys_sorted, 1):
            bl_pct = bl_alloc.get(t, 0) * 100
            amount = buy_amounts[t]
            reason = action_reasons.get(t, "")[:50]
            lines.append(
                f"  {rank}. *{t}* ¥{amount:,} (BL {bl_pct:.1f}%)\n"
                f"      _{reason}_"
            )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })

    # SELL + HOLD 圧縮
    others = []
    if sells:
        others.append(f"🔴 *SELL* ({len(sells)}): {', '.join(sells)}")
    if holds:
        others.append(f"🟡 *HOLD* ({len(holds)}): {', '.join(holds)}")
    if others:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(others)},
        })

    blocks.append({"type": "divider"})

    # New picks
    if new_picks:
        picks_text = "\n".join(
            f"  • *{p.get('ticker', '?')}* ({p.get('sector', '')}) — {p.get('reason', '')[:60]}"
            for p in new_picks[:5]
        )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*🆕 新規推奨銘柄*\n{picks_text}"},
        })

    # Risk scenarios
    risk_lines = []
    for key, emoji in [("bull", "🐂"), ("base", "📊"), ("bear", "🐻")]:
        s = risk_scenarios.get(key, {})
        prob = s.get("probability", 0)
        desc = s.get("description", "")[:60]
        if prob:
            risk_lines.append(f"  {emoji} *{key.upper()}* ({prob*100:.0f}%): {desc}")
    if risk_lines:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": "*⚖️ リスクシナリオ*\n" + "\n".join(risk_lines)},
        })

    # === 積立設定レビュー ===
    tsumi_blocks = build_tsumitate_section(latest)
    if tsumi_blocks:
        blocks.extend(tsumi_blocks)

    # === 学習ジャーナル（累積の学び） ===
    learn_blocks = build_learning_section()
    if learn_blocks:
        blocks.append({"type": "divider"})
        blocks.extend(learn_blocks)

    blocks.append({"type": "divider"})

    # Dashboard link
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn",
                 "text": f"🔗 <{DASHBOARD_URL}|ダッシュボードで詳細確認>"},
    })

    # Footer
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn",
             "text": f"🧬 Evolving Quant · 自動生成 · "
                     f"{now.strftime('%Y-%m-%d %H:%M')} JST"},
        ],
    })

    return blocks


# ==============================================================================
# 送信エンジン + エントリポイント
# ==============================================================================


def build_fallback_text(results: dict[str, Any], track: dict[str, Any]) -> str:
    """Block Kit 非対応環境向けのプレーンテキストを構築する。"""
    records = track.get("records", [])
    latest = records[-1] if records else {}
    confidence = latest.get("confidence", 0)
    actions = latest.get("actions", {})
    buys = [t for t, a in actions.items() if a == "BUY"]
    sells = [t for t, a in actions.items() if a == "SELL"]

    adv = results.get("advanced_signals", [])
    strong = [s["ticker"] for s in adv if "STRONG" in s.get("signal", "")]

    parts = [f"🧬 週次クオンツレポート | 確信度{confidence:.0%}"]
    if buys:
        parts.append(f"BUY: {', '.join(buys[:5])}")
    if sells:
        parts.append(f"SELL: {', '.join(sells)}")
    if strong:
        parts.append(f"🔮 STRONG: {', '.join(strong[:3])}")
    parts.append(DASHBOARD_URL)
    return " | ".join(parts)


def send_slack_dm(
    blocks: list[dict[str, Any]],
    fallback_text: str,
    dry_run: bool = False,
) -> bool:
    """Slack DM に Block Kit メッセージを送信する。

    Args:
        blocks: Block Kit ブロック
        fallback_text: フォールバックテキスト
        dry_run: True なら送信せずプレビューのみ

    Returns:
        成功なら True
    """
    if dry_run:
        logger.info("=== DRY RUN — 送信しません ===")
        logger.info("送信先: %s", AKI_USER_ID)
        logger.info("ブロック数: %d", len(blocks))
        logger.info("フォールバック: %s", fallback_text)
        print(json.dumps(blocks, indent=2, ensure_ascii=False))
        return True

    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.error("SLACK_BOT_TOKEN 未設定")
        return False

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        logger.error("slack_sdk 未インストール: pip install slack_sdk")
        return False

    client = WebClient(token=token)
    try:
        resp = client.chat_postMessage(
            channel=AKI_USER_ID,
            text=fallback_text,
            blocks=blocks,
            mrkdwn=True,
        )
        logger.info("✅ Slack DM 送信完了: ts=%s", resp["ts"])
        return True
    except SlackApiError as e:
        logger.error("❌ Slack送信エラー: %s", e.response["error"])
        return False
    except Exception as e:
        logger.error("❌ 予期しないエラー: %s", e)
        return False


def main() -> None:
    """メインエントリポイント。"""
    dry_run = "--dry-run" in sys.argv
    logger.info("📤 Slack DM通知開始%s", " (dry-run)" if dry_run else "")

    results = load_json(RESULTS_PATH)
    track = load_json(TRACK_RECORD_PATH)
    holdings = load_json(HOLDINGS_PATH)

    if not results:
        logger.error("latest_evolution_results.json 未検出 — スキップ")
        return

    blocks = build_slack_blocks(results, track, holdings)
    fallback = build_fallback_text(results, track)

    ok = send_slack_dm(blocks, fallback, dry_run=dry_run)
    if ok:
        logger.info("🎉 Slack通知完了")
    else:
        logger.error("❌ Slack通知失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
