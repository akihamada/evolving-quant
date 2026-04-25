# -*- coding: utf-8 -*-
from __future__ import annotations
"""
auto_prompt_cycle.py — 自動プロンプト送信 + 応答取得 + 記録
==========================================================
daily_evolution.py 実行後に呼ばれ、以下を自動実行:
  1. メタプロンプトを生成（BUY/SELL/HOLD + 積立 + 新規銘柄）
  2. Antigravity (Claude Opus) に既存会話で送信
  3. 応答待機 (240秒)
  4. 応答ファイル (/tmp/quant_response.json) を読み取り
  5. ai_track_record.json に自動記録

起動: python auto_prompt_cycle.py
"""

import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from market_data_enricher import enrich_all, format_for_prompt

# ロギング
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("auto-prompt")

# ==============================================================================
# パス定数
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_SCRIPT = SCRIPT_DIR.parent / "discord-bot" / "send_to_antigravity_chat.sh"
RESPONSE_FILE = Path("/tmp/quant_response.json")
TRACK_RECORD_PATH = SCRIPT_DIR / "ai_track_record.json"
RESULTS_PATH = SCRIPT_DIR / "latest_evolution_results.json"
HOLDINGS_PATH = SCRIPT_DIR / "data" / "portfolio_holdings.json"

WAIT_SECONDS = 600  # 10分待機（Claudeの応答処理に十分な時間）
MAX_RECORDS = 90  # 約45日分（2回/日）を保持、古いレコードは自動削除


# ==============================================================================
# ユーティリティ
# ==============================================================================


def load_json(path: Path) -> Optional[dict]:
    """
    JSONファイルを安全に読み込む。

    Args:
        path: JSONファイルパス

    Returns:
        辞書データ、エラー時はNone
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("JSON読み込み失敗 (%s): %s", path, e)
        return None


# ==============================================================================
# プロンプト生成
# ==============================================================================


def generate_auto_prompt(results: dict, track: dict) -> str:
    """
    Claude Opus向けの完全自動メタプロンプトを生成する。

    含まれる分析指示:
      - 保有銘柄ごとの BUY/HOLD/SELL 判定
      - 積立設定の見直し提案
      - 新規銘柄の発掘提案
      - リバランス配分
      - Devil's Advocate

    Args:
        results: latest_evolution_results.json のデータ
        track: ai_track_record.json のデータ

    Returns:
        プロンプト文字列
    """
    tickers = results.get("tickers", [])
    allocations = results.get("allocations", {})
    regime = results.get("regime", "unknown")
    kl_value = results.get("kl_value", 0)
    ensemble = allocations.get("ensemble", {})
    bl = allocations.get("bl", {})
    ew = results.get("ensemble_weights", [0.25, 0.25, 0.25, 0.25])
    evaluations = track.get("evaluations", [])

    regime_jp = {
        "low_vol": "低ボラティリティ（安定上昇局面）",
        "transition": "移行期（ボラ上昇中）",
        "crisis": "危機（高ボラ・急落局面）",
    }.get(regime, "不明")

    today = datetime.now().strftime("%Y-%m-%d")

    # === HEADER ===
    prompt = f"""🧬 EVOLVING QUANT AUTO-PROMPT — {today}
Self-Reflection Generation: {len(evaluations)}

あなたは自己学習型クオンツAIであり、私の専属投資アドバイザーです。
以下のデータを全て分析し、具体的なアクション提案を行ってください。

═══ SECTION A: 数学的ベースライン ═══

マーケットレジーム: {regime_jp} (KL={kl_value:.4f})

Ensemble最適配分:
"""
    for t in tickers:
        prompt += f"  {t}: {ensemble.get(t, 0):.1%}\n"
    prompt += f"\nEnsemble重み: NCO={ew[0]:.1%}, RP={ew[1]:.1%}, MV={ew[2]:.1%}, MD={ew[3]:.1%}\n"
    prompt += "\nBlack-Litterman配分:\n"
    for t in tickers:
        prompt += f"  {t}: {bl.get(t, 0):.1%}\n"

    # === SECTION A2: 銘柄別マーケットデータ ===
    try:
        # 前回の推奨銘柄を取得
        recent_picks = []
        for r in track.get("records", [])[-3:]:
            for pick in r.get("new_picks", []):
                if isinstance(pick, dict) and pick.get("ticker"):
                    recent_picks.append(pick["ticker"])

        # portfolio_holdings.json から追加購入銘柄を検出
        holdings_data_for_tickers = load_json(HOLDINGS_PATH)
        if holdings_data_for_tickers:
            for section in ["tokutei", "nisa"]:
                for s in holdings_data_for_tickers.get("us_stocks", {}).get(section, []):
                    t = s.get("ticker")
                    if t and t not in tickers:
                        recent_picks.append(t)
            for s in holdings_data_for_tickers.get("japan_stocks", {}).get("nisa_growth", []):
                code = s.get("code", "")
                ticker_jp = f"{code}.T" if code and f"{code}.T" not in tickers else ""
                if ticker_jp and ticker_jp not in tickers:
                    recent_picks.append(ticker_jp)

        recent_picks = list(set(recent_picks))

        enriched = enrich_all(tickers, recent_picks)
        if enriched:
            prompt += "\n" + format_for_prompt(enriched)
        else:
            prompt += "\n（銘柄データ取得失敗）\n"
    except Exception as e:
        logger.warning("マーケットデータ取得スキップ: %s", e)
        prompt += "\n（銘柄データ取得スキップ）\n"

    # === SECTION B: 保有ポートフォリオ ===
    prompt += "\n═══ SECTION B: 現在の保有ポートフォリオ ═══\n\n"
    holdings_data = load_json(HOLDINGS_PATH)

    if holdings_data:
        prompt += "■ 米国株式:\n"
        all_us = holdings_data.get("us_stocks", {}).get("tokutei", []) + \
                 holdings_data.get("us_stocks", {}).get("nisa", [])
        for s in all_us:
            pnl = s.get("unrealized_pnl_usd", 0)
            sign = "+" if pnl >= 0 else ""
            prompt += (f"  {s['ticker']:6s} {s['shares']:>3d}株 "
                       f"取得${s.get('cost_basis_usd', 0):.0f} → "
                       f"現値${s.get('current_price_usd', 0):.0f} "
                       f"損益{sign}${pnl:.0f} [{s.get('sector', '')}]\n")

        summary = holdings_data.get("us_stocks", {}).get("summary", {})
        prompt += f"\n  合計: ${summary.get('grand_total_usd', 0):,.0f} (損益+${summary.get('grand_total_pnl_usd', 0):,.0f})\n"

        jp_stocks = holdings_data.get("japan_stocks", {}).get("nisa_growth", [])
        if jp_stocks:
            prompt += "\n■ 日本株式:\n"
            for s in jp_stocks:
                pnl = s.get("unrealized_pnl_jpy", 0)
                sign = "+" if pnl >= 0 else ""
                prompt += f"  [{s['code']}] {s['name']} {s['shares']}株 損益{sign}¥{pnl:,}\n"

        prompt += "\n■ 投資信託:\n"
        for cat in ["nisa_growth", "nisa_tsumitate"]:
            for f in holdings_data.get("mutual_funds", {}).get(cat, []):
                pnl = f.get("unrealized_pnl_jpy", 0)
                sign = "+" if pnl >= 0 else ""
                name = f["name"][:25]
                prompt += f"  {name}  損益{sign}¥{pnl:,}  [{f.get('category', '')}]\n"

        tsumitate = holdings_data.get("tsumitate_settings", {})
        prompt += f"\n■ 積立設定（月額¥{tsumitate.get('monthly_total_jpy', 0):,}）:\n"
        for p in tsumitate.get("plans", []):
            prompt += f"  ¥{p['monthly_jpy']:>6,}/月 {p['fund'][:30]} [{p['account']}]\n"

    # === SECTION C: 自己反省 ===
    prompt += "\n═══ SECTION C: 自己反省データ ═══\n\n"
    if evaluations:
        latest_evals = evaluations[-5:]
        avg_rmse = np.mean([e.get("predicted_vs_actual", {}).get("rmse", 0) for e in latest_evals])
        avg_dir = np.mean([e.get("predicted_vs_actual", {}).get("direction_accuracy", 0) for e in latest_evals])
        prompt += f"直近{len(latest_evals)}回: RMSE={avg_rmse:.4f}, 方向性的中率={avg_dir:.1%}\n"
        for e in latest_evals:
            prompt += f"  [{e.get('eval_date', '')}] {e.get('bias_analysis', 'N/A')}\n"
    else:
        prompt += "（初回: 評価データなし）\n"

    records = track.get("records", [])
    if records:
        prompt += "\n■ 直近の予測:\n"
        for r in records[-3:]:
            prompt += f"  [{r.get('date', '')}] 確信度={r.get('confidence', 'N/A')}\n"

    # === SECTION D: 推論指示 ===
    prompt += """
═══ SECTION D: 分析指示（6つ全て実行） ═══

【1. マクロ環境分析】3行で要点。

【2. 保有銘柄アクション判定】全銘柄に対して:
  🟢 BUY（買い増し）— 理由1行
  🟡 HOLD（保持）— 理由1行
  🔴 SELL（売却検討）— 理由1行

【3. 積立設定レビュー】
  - 各ファンドの配分は適切か？
  - 変更すべきファンドや割合は？
  - 追加すべきファンドは？

【4. 新規銘柄提案】3〜5銘柄
  ポートフォリオに不足するセクター・テーマから提案。
  ティッカー / セクター / 推奨理由 を各1行。

【5. Devil's Advocate】提案への反論3つ + 再反論。

【6. 価格予測（SECTION A2のデータを使用）】
  全保有銘柄+推奨銘柄に対して:
  - 1ヶ月後の予測レンジ（下限〜上限）
  - 3ヶ月後の予測レンジ（下限〜上限）
  - 予測の根拠（テクニカル/ファンダ/センチメントの組み合わせ）を1行で記述
  ※ SECTION A2のCAGR・RSI・アナリスト目標株価・危機耐性データを定量根拠に使うこと

"""

    # === SECTION E: JSON出力指示 ===
    ticker_template = ", ".join([f'"{t}": 0.XX' for t in tickers])
    prompt += f"""═══ 【最重要】出力指示 ═══

分析完了後、必ず以下のPythonコードを実行して /tmp/quant_response.json に保存:

```python
import json
response = {{
    "allocations": {{{ticker_template}}},
    "confidence": 0.XX,
    "reasoning": "推奨理由1-2行",
    "actions": {{"TICKER": "BUY/HOLD/SELL", ...全銘柄}},
    "action_reasons": {{"TICKER": "理由", ...全銘柄}},
    "tsumitate_advice": {{
        "keep_current": True,
        "changes": ["変更提案"],
        "reasoning": "分析"
    }},
    "new_picks": [
        {{"ticker": "XXXX", "sector": "セクター", "reason": "理由"}}
    ],
    "risk_scenarios": {{
        "bull": {{"probability": 0.XX, "description": "..."}},
        "base": {{"probability": 0.XX, "description": "..."}},
        "bear": {{"probability": 0.XX, "description": "..."}}
    }}
}}
with open("/tmp/quant_response.json", "w") as f:
    json.dump(response, f, ensure_ascii=False, indent=2)
print("✅ 保存完了: /tmp/quant_response.json")
```

配分合計=1.0。確信度0.0〜1.0。actionsは全保有銘柄。必ず実行。"""

    return prompt


# ==============================================================================
# Antigravity 送信
# ==============================================================================


def send_to_antigravity(message: str) -> bool:
    """
    既存のAntigravity会話にメッセージを送信する。

    プロンプトをファイルに保存し、--file モードでブリッジスクリプトに
    渡すことで、長い多行メッセージも確実に送信する。

    Args:
        message: 送信するメッセージ

    Returns:
        送信成功したかどうか
    """
    if not BRIDGE_SCRIPT.exists():
        logger.error("ブリッジスクリプトが見つかりません: %s", BRIDGE_SCRIPT)
        return False

    # プロンプトをファイルに保存（--file モードで送信）
    prompt_file = Path("/tmp/quant_latest_prompt.txt")
    prompt_file.write_text(message, encoding="utf-8")
    logger.info("プロンプトをファイルに保存: %s (%d bytes)", prompt_file, len(message.encode("utf-8")))

    try:
        result = subprocess.run(
            ["/bin/bash", str(BRIDGE_SCRIPT), "--file", str(prompt_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info("✅ Antigravity に送信成功")
            return True
        else:
            logger.error("送信失敗 (exit=%d): %s", result.returncode, result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("送信タイムアウト")
        return False
    except Exception as e:
        logger.error("送信エラー: %s", e)
        return False


# ==============================================================================
# 応答取得
# ==============================================================================


def wait_for_response(timeout: int = WAIT_SECONDS) -> Optional[dict]:
    """
    Claude の応答ファイルを待機・取得する。

    Args:
        timeout: 最大待機秒数

    Returns:
        応答辞書データ、またはNone
    """
    if RESPONSE_FILE.exists():
        RESPONSE_FILE.unlink()
        logger.info("古い応答ファイルを削除")

    logger.info("⏳ 応答待機中... (最大 %d秒)", timeout)
    start = time.time()

    while time.time() - start < timeout:
        if RESPONSE_FILE.exists():
            time.sleep(2)
            data = load_json(RESPONSE_FILE)
            if data and "allocations" in data:
                logger.info("✅ 応答取得成功 (%d秒後)", int(time.time() - start))
                return data
            else:
                logger.warning("応答ファイル不完全。再待機...")
        time.sleep(10)

    logger.warning("⏰ タイムアウト (%d秒)", timeout)
    return None


# ==============================================================================
# 自動記録
# ==============================================================================


def auto_record(response: dict, track: dict) -> None:
    """
    Claude の応答を ai_track_record.json に自動記録する。

    Args:
        response: パース済みの応答データ
        track: 現在のトラックレコード
    """
    allocations = response.get("allocations", {})
    confidence = response.get("confidence", 0.5)
    reasoning = response.get("reasoning", "")

    total = sum(allocations.values())
    if total > 0:
        allocations = {k: v / total for k, v in allocations.items()}

    new_record = {
        "id": str(uuid.uuid4())[:8],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ai_allocation": allocations,
        "ai_reasoning": reasoning,
        "confidence": confidence,
        "actions": response.get("actions", {}),
        "action_reasons": response.get("action_reasons", {}),
        "tsumitate_advice": response.get("tsumitate_advice", {}),
        "new_picks": response.get("new_picks", []),
        "risk_scenarios": response.get("risk_scenarios", {}),
        "auto_generated": True,
        "evaluated": False,
        "evaluation": None,
    }

    track["records"].append(new_record)

    # レコード上限チェック — 古いレコードを自動削除
    if len(track["records"]) > MAX_RECORDS:
        removed = len(track["records"]) - MAX_RECORDS
        track["records"] = track["records"][-MAX_RECORDS:]
        logger.info("🗑️ 古いレコード %d件を削除（上限%d件）", removed, MAX_RECORDS)

    track["meta"]["total_predictions"] = len(track["records"])

    with open(TRACK_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(track, f, ensure_ascii=False, indent=2)

    logger.info("✅ 自動記録完了 (id=%s, confidence=%.2f)", new_record["id"], confidence)


# ==============================================================================
# メイン
# ==============================================================================


def run_auto_cycle() -> bool:
    """
    自動プロンプトサイクルのメインエントリポイント。

    Returns:
        全工程が成功したかどうか
    """
    logger.info("=" * 60)
    logger.info("🧬 Auto Prompt Cycle 開始")
    logger.info("=" * 60)

    results = load_json(RESULTS_PATH)
    if not results:
        logger.error("latest_evolution_results.json が読み込めません")
        return False

    track = load_json(TRACK_RECORD_PATH)
    if not track:
        logger.error("ai_track_record.json が読み込めません")
        return False

    holdings = load_json(HOLDINGS_PATH) or {}

    # === モダン分析パス: Claude CLI (Max枠) → Gemini API → なし ===
    response = None

    # 1. Claude CLI (Claude Max 枠で追加課金なし)
    try:
        from claude_analyst import analyze_portfolio as claude_analyze
        logger.info("🤖 Claude CLI で分析中...")
        response = claude_analyze(results, track, holdings)
        if response:
            logger.info("✅ Claude CLI 分析完了")
    except ImportError:
        logger.info("claude_analyst なし → Gemini にフォールバック")
    except Exception as e:
        logger.warning("Claude CLI 失敗: %s → Gemini にフォールバック", e)

    # 2. Gemini API (.env の GEMINI_API_KEY 必須)
    if not response:
        try:
            from gemini_analyst import analyze_portfolio as gemini_analyze
            logger.info("🤖 Gemini 2.5 Flash で分析中...")
            response = gemini_analyze(results, track, holdings)
            if response:
                logger.info("✅ Gemini 分析完了")
        except ImportError:
            logger.warning("gemini_analyst モジュールなし")
        except Exception as e:
            logger.warning("Gemini 失敗: %s", e)

    if not response:
        logger.error("Claude/Gemini 両方失敗 — Antigravity レガシーは廃止済み")
        return False
    if not response:
        logger.warning("応答未取得。手動で Record タブから入力してください。")
        return False

    auto_record(response, track)

    # 一時ファイルクリーンアップ
    for tmp_file in [RESPONSE_FILE, Path("/tmp/quant_latest_prompt.txt")]:
        if tmp_file.exists():
            tmp_file.unlink()
            logger.info("🗑️ 一時ファイル削除: %s", tmp_file)

    logger.info("=" * 60)
    logger.info("🎉 Auto Prompt Cycle 完了")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = run_auto_cycle()
    sys.exit(0 if success else 1)
