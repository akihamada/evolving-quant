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

    # === Advanced AI 予測（新規） ===
    adv_blocks = build_advanced_section(advanced)
    if adv_blocks:
        blocks.extend(adv_blocks)
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
