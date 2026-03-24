# -*- coding: utf-8 -*-
"""
Discord Notifier — Embed形式通知モジュール
=========================================
Gemini のpydantic構造化出力を美しいDiscord Embed形式に変換し、
Webhookで送信する。

Discordの制限:
  - Embed Description: 4096文字
  - Embed Fields: 最大25個
  - 全体リクエスト: 6000文字
  - Embeds/リクエスト: 最大10個
超過時はチャンク分割で複数リクエストに送信。
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import requests

from ai_analyzer import PortfolioAnalysis

logger = logging.getLogger("portfolio-bot.notify")

# Discord 制限定数
MAX_EMBED_DESC = 4096
MAX_EMBED_FIELDS = 25
MAX_FIELD_VALUE = 1024
MAX_EMBEDS_PER_REQUEST = 10

# カラーコード
COLOR_BULLISH = 0x00D166   # 緑
COLOR_BEARISH = 0xFD0061   # 赤
COLOR_NEUTRAL = 0x7289DA   # 灰青
COLOR_INFO = 0x5865F2      # Discord Blurple
COLOR_WARNING = 0xFEE75C   # 黄色


# ==============================================================================
# ヘルパー
# ==============================================================================


def _truncate(text: str, max_len: int) -> str:
    """
    テキストを指定長で切り詰める。

    Args:
        text: 対象テキスト
        max_len: 最大文字数

    Returns:
        切り詰め済みテキスト
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sentiment_color(sentiment: str) -> int:
    """
    センチメント文字列に対応するカラーコードを返す。

    Args:
        sentiment: "強気", "弱気", "中立" など

    Returns:
        Discord カラーコード (int)
    """
    if "強気" in sentiment or "bullish" in sentiment.lower():
        return COLOR_BULLISH
    elif "弱気" in sentiment or "bearish" in sentiment.lower():
        return COLOR_BEARISH
    return COLOR_NEUTRAL


def _verdict_emoji(verdict: str) -> str:
    """
    判定文字列に対応する絵文字を返す。

    Args:
        verdict: "強気", "割安", "弱気", "割高" など

    Returns:
        対応する絵文字
    """
    mapping = {
        "強気": "🟢", "bullish": "🟢",
        "弱気": "🔴", "bearish": "🔴",
        "中立": "🟡", "neutral": "🟡",
        "割安": "💰", "割高": "💸", "適正": "📊",
    }
    for key, emoji in mapping.items():
        if key in verdict.lower():
            return emoji
    return "⚪"


# ==============================================================================
# Embed ビルダー
# ==============================================================================


def build_embeds(analysis: PortfolioAnalysis) -> list[dict[str, Any]]:
    """
    PortfolioAnalysis からDiscord Embed群を構築する。

    Discordの制限を考慮し、論理的にEmbed群を分割する。

    Args:
        analysis: AI分析結果

    Returns:
        Embed辞書のリスト
    """
    embeds: list[dict[str, Any]] = []
    now = datetime.now().strftime("%Y/%m/%d %H:%M")

    # --- Embed 1: マクロ概況 + センチメント ---
    sentiment = analysis.english_news_sentiment
    macro_embed = {
        "title": "📊 ポートフォリオ分析レポート",
        "description": _truncate(analysis.macro_summary, MAX_EMBED_DESC),
        "color": _sentiment_color(sentiment.overall_sentiment),
        "fields": [
            {
                "name": "📰 ニュースセンチメント",
                "value": (
                    f"{_verdict_emoji(sentiment.overall_sentiment)} "
                    f"**{sentiment.overall_sentiment}** "
                    f"(スコア: {sentiment.score}/100)\n"
                    f"{_truncate(sentiment.summary, 200)}"
                ),
                "inline": False,
            },
            {
                "name": "🔑 キーテーマ",
                "value": " | ".join(sentiment.key_themes) if sentiment.key_themes else "N/A",
                "inline": True,
            },
            {
                "name": "🧠 スマートマネー動向",
                "value": _truncate(analysis.smart_money_signal, MAX_FIELD_VALUE),
                "inline": False,
            },
        ],
        "footer": {"text": f"生成: {now} | Gemini 2.5 Pro"},
    }
    embeds.append(macro_embed)

    # --- Embed 2: リスク警告 ---
    if analysis.risk_alerts:
        alerts_text = "\n".join(f"⚠️ {alert}" for alert in analysis.risk_alerts)
        risk_embed = {
            "title": "🚨 リスク警告",
            "description": _truncate(alerts_text, MAX_EMBED_DESC),
            "color": COLOR_WARNING,
        }
        embeds.append(risk_embed)

    # --- Embed 3+: 銘柄評価 (Fieldで最大25個制限があるので分割) ---
    stock_fields: list[dict] = []
    for ev in analysis.stock_evaluations:
        dip_flag = " | 🎯 **押し目買い**" if ev.is_buy_dip else ""
        earnings_flag = f"\n📅 {ev.earnings_note}" if ev.earnings_note else ""

        field = {
            "name": f"{ev.ticker} ({ev.company_name})",
            "value": _truncate(
                f"{_verdict_emoji(ev.technical_verdict)} テクニカル: {ev.technical_verdict} | "
                f"{_verdict_emoji(ev.fundamental_verdict)} ファンダ: {ev.fundamental_verdict}"
                f"{dip_flag}{earnings_flag}\n"
                f"💬 {ev.one_line_comment}",
                MAX_FIELD_VALUE,
            ),
            "inline": False,
        }
        stock_fields.append(field)

    # Field上限25個ごとにEmbedを分割
    for i in range(0, len(stock_fields), MAX_EMBED_FIELDS):
        chunk = stock_fields[i: i + MAX_EMBED_FIELDS]
        page_num = i // MAX_EMBED_FIELDS + 1
        stock_embed = {
            "title": f"📈 銘柄評価" + (f" ({page_num})" if len(stock_fields) > MAX_EMBED_FIELDS else ""),
            "color": COLOR_INFO,
            "fields": chunk,
        }
        embeds.append(stock_embed)

    # --- Embed Last: アクションプラン ---
    if analysis.action_plan:
        action_lines = []
        for act in analysis.action_plan:
            shares_text = f" ({act.shares}株)" if act.shares else ""
            emoji = {"買い増し": "🟢", "一部売却": "🟡", "全売却": "🔴", "保持": "⚪"}.get(act.action, "📌")
            action_lines.append(
                f"{emoji} **{act.ticker}** — {act.action}{shares_text}\n"
                f"  ↳ {act.rationale}"
            )

        action_text = "\n\n".join(action_lines)
        action_embed = {
            "title": "🎯 アクションプラン",
            "description": _truncate(action_text, MAX_EMBED_DESC),
            "color": COLOR_INFO,
            "footer": {"text": "※ 投資判断は自己責任でお願いします"},
        }
        embeds.append(action_embed)

    return embeds


# ==============================================================================
# Webhook 送信
# ==============================================================================


def send_to_discord(
    embeds: list[dict[str, Any]],
    webhook_url: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    """
    Discord Webhook にEmbed群を送信する。

    10個/リクエスト制限を超える場合はチャンク分割して逐次送信。
    レート制限 (429) 時は指数バックオフでリトライ。

    Args:
        embeds: Embed辞書のリスト
        webhook_url: Webhook URL (Noneの場合は環境変数から取得)
        max_retries: 最大リトライ回数

    Returns:
        全送信成功なら True
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        logger.error("DISCORD_WEBHOOK_URL が未設定です")
        return False

    all_success = True

    # チャンク分割 (10 embeds/request)
    for i in range(0, len(embeds), MAX_EMBEDS_PER_REQUEST):
        chunk = embeds[i: i + MAX_EMBEDS_PER_REQUEST]
        payload = {"embeds": chunk}

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=30)

                if resp.status_code == 204:
                    logger.info("  Webhook送信成功 (embeds %d-%d)", i + 1, i + len(chunk))
                    break
                elif resp.status_code == 429:
                    retry_after = resp.json().get("retry_after", 2 ** (attempt + 1))
                    logger.warning("  レート制限 (429)。%.1f秒後にリトライ", retry_after)
                    time.sleep(float(retry_after))
                else:
                    logger.error("  Webhook送信エラー: %d %s", resp.status_code, resp.text[:200])
                    all_success = False
                    break

            except requests.RequestException as e:
                logger.error("  Webhookリクエスト失敗: %s", e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    all_success = False

        # チャンク間に少し待機 (レート制限回避)
        if i + MAX_EMBEDS_PER_REQUEST < len(embeds):
            time.sleep(1)

    return all_success


def notify(analysis: PortfolioAnalysis) -> bool:
    """
    分析結果をDiscord Embedとして送信する統合関数。

    Args:
        analysis: AI分析結果

    Returns:
        送信成功なら True
    """
    logger.info("Discord通知を構築中...")
    embeds = build_embeds(analysis)
    logger.info("  Embed数: %d", len(embeds))
    return send_to_discord(embeds)
