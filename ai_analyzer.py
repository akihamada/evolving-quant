# -*- coding: utf-8 -*-
"""
AI Analyzer — ルールベース分析 + Gemini 2.5 Pro 構造化出力
=========================================================
GEMINI_API_KEY が設定されていれば Gemini で分析。
未設定の場合はルールベースのロジックでデータ駆動分析を行う。
"""

import json
import logging
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("portfolio-bot.ai")


# ==============================================================================
# Pydantic 出力スキーマ
# ==============================================================================


class StockEvaluation(BaseModel):
    """個別銘柄の評価。"""

    ticker: str = Field(description="ティッカーシンボル")
    company_name: str = Field(description="企業名")
    technical_verdict: str = Field(description="テクニカル判定: 強気/中立/弱気")
    fundamental_verdict: str = Field(description="ファンダメンタル判定: 割安/適正/割高")
    is_buy_dip: bool = Field(description="押し目買いチャンスかどうか")
    earnings_note: str = Field(description="決算に関する注意事項")
    one_line_comment: str = Field(description="一言コメント（50文字以内）")


class ActionItem(BaseModel):
    """具体的な売買アクション。"""

    ticker: str = Field(description="ティッカーシンボル")
    action: str = Field(description="アクション: 買い増し/一部売却/保持/全売却")
    shares: Optional[int] = Field(default=None, description="推奨株数（目安）")
    rationale: str = Field(description="論理的根拠（100文字以内）")


class NewsSentiment(BaseModel):
    """英語ニュースのセンチメント分析。"""

    overall_sentiment: str = Field(description="全体判定: 強気/中立/弱気")
    score: int = Field(description="センチメントスコア 0-100")
    key_themes: list[str] = Field(description="主要テーマ（英語キーワード3つ以内）")
    summary: str = Field(description="ニュース文脈の要約（日本語100文字以内）")


class PortfolioAnalysis(BaseModel):
    """ポートフォリオ分析の完全な結果スキーマ。"""

    macro_summary: str = Field(description="マクロ環境を踏まえた全体相場観（300文字以内）")
    risk_alerts: list[str] = Field(description="リスク警告のリスト（箇条書き）")
    english_news_sentiment: NewsSentiment = Field(description="英語ニュースセンチメント")
    smart_money_signal: str = Field(
        description="インサイダー取引とP/Cレシオから読む大口投資家の動向（200文字以内）"
    )
    stock_evaluations: list[StockEvaluation] = Field(description="各銘柄の評価リスト")
    action_plan: list[ActionItem] = Field(description="具体的な売買アクションプラン")


# ==============================================================================
# システムプロンプト (Gemini用)
# ==============================================================================

SYSTEM_PROMPT = """あなたは機関投資家（ヘッジファンド）レベルのポートフォリオアナリストです。
提供されたファクトデータのみに基づいて分析してください。

【絶対遵守ルール】
1. 推測・捏造（ハルシネーション）は厳禁。データにない情報を付け加えないでください。
2. 取得失敗でNoneの指標がある場合は「データ不足」と明示し、推測で代替しないでください。
3. 楽観バイアスを排除し、リスクファクターを重視してください。
4. 日本語で出力してください（英語ニュースのキーワードは英語のまま）。
5. 各フィールドの文字数制限を厳守してください。
6. 押し目買い判定は RSI < 30 かつ SMA200乖離率 < -10% を目安にしてください。
7. 相関係数 0.8以上の銘柄ペアはリスク警告で明示してください。
8. P/Cレシオ > 1.0 は弱気シグナル、< 0.7 は強気シグナルとして扱ってください。"""


# ==============================================================================
# ルールベース分析 (Gemini不要)
# ==============================================================================


def _rule_based_analysis(market_data: dict[str, Any]) -> PortfolioAnalysis:
    """
    ルールベースでポートフォリオ分析を行う。

    GEMINI_API_KEY 不要。データの数値のみに基づき、
    固定ルールで判定する。AIのような文脈理解はないが
    客観的な指標判定は正確に行える。

    Args:
        market_data: data_fetcher.collect_all_data() の出力

    Returns:
        PortfolioAnalysis インスタンス
    """
    logger.info("ルールベース分析を実行中 (Gemini APIなし)...")

    macro = market_data.get("macro", {})
    stocks = market_data.get("stocks", [])
    correlation = market_data.get("correlation", {})

    # --- マクロサマリー ---
    macro_parts = []
    for key, data in macro.items():
        cur = data.get("current")
        chg = data.get("month_change_pct")
        label = {"JPY=X": "ドル円", "^TNX": "米10年債利回り", "^VIX": "VIX"}.get(key, key)
        if cur is not None:
            direction = "上昇" if chg and chg > 0 else "下落" if chg and chg < 0 else "横ばい"
            macro_parts.append(f"{label}: {cur:.2f} (1M {direction} {chg:+.1f}%)" if chg else f"{label}: {cur:.2f}")

    vix_data = macro.get("^VIX", {})
    vix_val = vix_data.get("current")
    if vix_val and vix_val > 25:
        macro_parts.append("⚠️ VIXが25超 → リスクオフ環境")
    elif vix_val and vix_val < 15:
        macro_parts.append("✅ VIX低水準 → 安定的な環境")

    macro_summary = "【マクロ環境】" + " / ".join(macro_parts) if macro_parts else "マクロデータ取得不足"

    # --- リスク警告 ---
    risk_alerts: list[str] = []

    # 相関リスク
    if isinstance(correlation, dict):
        for pair, corr_val in correlation.items():
            if isinstance(corr_val, (int, float)) and corr_val > 0.8:
                risk_alerts.append(f"高相関: {pair} (相関{corr_val:.2f}) → 分散効果が弱い")

    # セクター偏り
    sectors: dict[str, int] = {}
    for s in stocks:
        sec = s.get("sector", "Unknown")
        sectors[sec] = sectors.get(sec, 0) + 1
    for sec, count in sectors.items():
        if count >= 2 and len(stocks) > 0:
            ratio = count / len(stocks) * 100
            if ratio >= 40:
                risk_alerts.append(f"セクター集中: {sec} が {ratio:.0f}% ({count}/{len(stocks)}銘柄)")

    # VIXリスク
    if vix_val and vix_val > 30:
        risk_alerts.append(f"VIX={vix_val:.1f} → 恐怖指数が極端に高い。防御的ポジショニングを推奨")

    if not risk_alerts:
        risk_alerts.append("現時点で重大なリスク警告なし")

    # --- ニュースセンチメント ---
    all_headlines = []
    for s in stocks:
        all_headlines.extend(s.get("news_headlines", []))

    bullish_words = ["surge", "rally", "beat", "growth", "upgrade", "strong", "record", "gain"]
    bearish_words = ["fall", "drop", "miss", "cut", "downgrade", "weak", "risk", "decline", "crash"]

    bull_count = sum(1 for h in all_headlines for w in bullish_words if w.lower() in h.lower())
    bear_count = sum(1 for h in all_headlines for w in bearish_words if w.lower() in h.lower())

    if bull_count > bear_count + 2:
        news_sentiment = "強気"
        news_score = min(80, 50 + (bull_count - bear_count) * 5)
    elif bear_count > bull_count + 2:
        news_sentiment = "弱気"
        news_score = max(20, 50 - (bear_count - bull_count) * 5)
    else:
        news_sentiment = "中立"
        news_score = 50

    key_themes = list(set(h.split()[0] for h in all_headlines[:6] if h.split()))[:3]

    english_news_sentiment = NewsSentiment(
        overall_sentiment=news_sentiment,
        score=news_score,
        key_themes=key_themes if key_themes else ["N/A"],
        summary=f"ヘッドライン{len(all_headlines)}件分析: 強気ワード{bull_count}件 vs 弱気ワード{bear_count}件"
    )

    # --- スマートマネー ---
    total_insider_buy = sum(s.get("insider", {}).get("buy_count", 0) for s in stocks)
    total_insider_sell = sum(s.get("insider", {}).get("sell_count", 0) for s in stocks)
    pc_ratios = [s.get("options", {}).get("put_call_ratio") for s in stocks
                 if s.get("options", {}).get("put_call_ratio") is not None]
    avg_pc = sum(pc_ratios) / len(pc_ratios) if pc_ratios else None

    smart_parts = [f"インサイダー: Buy {total_insider_buy}件 / Sell {total_insider_sell}件"]
    if avg_pc is not None:
        if avg_pc > 1.0:
            smart_parts.append(f"P/Cレシオ平均 {avg_pc:.2f} → 弱気シグナル（ヘッジ増加）")
        elif avg_pc < 0.7:
            smart_parts.append(f"P/Cレシオ平均 {avg_pc:.2f} → 強気シグナル（コール優勢）")
        else:
            smart_parts.append(f"P/Cレシオ平均 {avg_pc:.2f} → 中立")
    else:
        smart_parts.append("P/Cレシオ: データなし")

    smart_money_signal = " / ".join(smart_parts)

    # --- 銘柄評価 ---
    stock_evaluations: list[StockEvaluation] = []
    action_plan: list[ActionItem] = []

    for s in stocks:
        ticker = s.get("ticker", "???")
        name = s.get("company_name", ticker)
        rsi = s.get("rsi_14")
        sma50 = s.get("sma_50_dev_pct")
        sma200 = s.get("sma_200_dev_pct")
        pe = s.get("trailing_pe")
        fwd_pe = s.get("forward_pe")
        month_ret = s.get("month_return_pct")
        pc_ratio = s.get("options", {}).get("put_call_ratio")
        earnings = s.get("earnings", {})

        # テクニカル判定
        if rsi is not None:
            if rsi < 30:
                tech = "弱気"
            elif rsi > 70:
                tech = "強気"
            elif sma50 is not None and sma50 > 5:
                tech = "強気"
            elif sma50 is not None and sma50 < -5:
                tech = "弱気"
            else:
                tech = "中立"
        else:
            tech = "中立"

        # ファンダ判定
        if pe is not None:
            if pe < 15:
                funda = "割安"
            elif pe > 40:
                funda = "割高"
            else:
                funda = "適正"
        elif fwd_pe is not None:
            if fwd_pe < 15:
                funda = "割安"
            elif fwd_pe > 40:
                funda = "割高"
            else:
                funda = "適正"
        else:
            funda = "適正"

        # 押し目買い
        is_dip = rsi is not None and rsi < 30 and sma200 is not None and sma200 < -10

        # 決算ノート
        if earnings.get("imminent"):
            e_note = f"⚠️ 決算予定: {earnings.get('next_date', '不明')} (残{earnings.get('days_until', '?')}日)"
        else:
            e_note = ""

        # コメント生成
        parts = []
        if month_ret is not None:
            parts.append(f"1M {month_ret:+.1f}%")
        if rsi is not None:
            parts.append(f"RSI {rsi:.0f}")
        if pe is not None:
            parts.append(f"PER {pe:.1f}")
        comment = " / ".join(parts) if parts else "データ不足"

        stock_evaluations.append(StockEvaluation(
            ticker=ticker,
            company_name=name,
            technical_verdict=tech,
            fundamental_verdict=funda,
            is_buy_dip=is_dip,
            earnings_note=e_note,
            one_line_comment=comment[:50],
        ))

        # アクション提案
        if is_dip:
            action_plan.append(ActionItem(
                ticker=ticker, action="買い増し", shares=None,
                rationale=f"RSI={rsi:.0f}+SMA200乖離{sma200:.1f}%の押し目サイン"
            ))
        elif rsi is not None and rsi > 75 and funda == "割高":
            action_plan.append(ActionItem(
                ticker=ticker, action="一部売却", shares=None,
                rationale=f"RSI={rsi:.0f}+PER={pe:.0f}の過熱サイン"
            ))
        elif pc_ratio is not None and pc_ratio > 1.5:
            action_plan.append(ActionItem(
                ticker=ticker, action="保持", shares=None,
                rationale=f"P/Cレシオ{pc_ratio:.2f}と高め。ヘッジ需要増→様子見"
            ))
        else:
            action_plan.append(ActionItem(
                ticker=ticker, action="保持", shares=None,
                rationale="現時点で明確な売買サインなし"
            ))

    return PortfolioAnalysis(
        macro_summary=macro_summary[:300],
        risk_alerts=risk_alerts,
        english_news_sentiment=english_news_sentiment,
        smart_money_signal=smart_money_signal[:200],
        stock_evaluations=stock_evaluations,
        action_plan=action_plan,
    )


# ==============================================================================
# Gemini API 分析
# ==============================================================================


def _gemini_analysis(market_data: dict[str, Any]) -> Optional[PortfolioAnalysis]:
    """
    Gemini 2.5 Pro にマーケットデータを送信し、構造化出力で分析結果を取得する。

    Args:
        market_data: data_fetcher.collect_all_data() の出力

    Returns:
        PortfolioAnalysis インスタンス、失敗時は None
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai がインストールされていません → ルールベースにフォールバック")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        data_text = json.dumps(market_data, ensure_ascii=False, indent=2, default=str)

        if len(data_text) > 100000:
            logger.warning("入力データが長すぎるため切り詰め（100KB上限）")
            data_text = data_text[:100000] + "\n... (以下省略)"

        user_prompt = f"""以下のポートフォリオのマーケットデータを分析し、
機関投資家レベルのポートフォリオ診断を行ってください。

【マーケットデータ】
{data_text}

上記データのみに基づいて、各フィールドを正確に埋めてください。"""

        logger.info("Gemini 2.5 Pro に分析リクエスト送信中...")
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=PortfolioAnalysis,
                temperature=0.2,
            ),
        )

        if not response or not response.text:
            logger.error("Gemini API: 空の応答が返されました")
            return None

        raw_json = json.loads(response.text)
        result = PortfolioAnalysis.model_validate(raw_json)

        logger.info("AI分析完了: %d銘柄評価, %dアクション",
                     len(result.stock_evaluations), len(result.action_plan))
        return result

    except json.JSONDecodeError as e:
        logger.error("Gemini応答のJSONパース失敗: %s", e)
        return None
    except Exception as e:
        logger.error("Gemini API エラー: %s", e)
        return None


# ==============================================================================
# 統合エントリポイント
# ==============================================================================


def analyze_portfolio(market_data: dict[str, Any]) -> Optional[PortfolioAnalysis]:
    """
    ポートフォリオを分析するメイン関数。

    GEMINI_API_KEY が設定されていれば Gemini で分析。
    未設定 or Gemini失敗時はルールベース分析にフォールバック。

    Args:
        market_data: data_fetcher.collect_all_data() の出力

    Returns:
        PortfolioAnalysis インスタンス
    """
    # Gemini API を優先試行
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.info("Gemini API で分析を試行...")
        result = _gemini_analysis(market_data)
        if result:
            return result
        logger.warning("Gemini 分析失敗 → ルールベースにフォールバック")

    # ルールベース分析
    logger.info("ルールベース分析を使用（GEMINI_API_KEY 未設定）")
    return _rule_based_analysis(market_data)
