#!/usr/bin/env python3
"""
claude_analyst.py — Claude Code CLI (Claude Max) によるポートフォリオ分析

追加API課金なしで Claude Sonnet/Opus の品質分析を得る。
`claude` CLI (Claude Code) を subprocess 経由で呼び出し、
構造化JSON応答を取得する。

前提条件:
  - `claude` CLI がインストール済み (https://claude.ai/code)
  - 実行ユーザーで `claude` に Anthropic アカウントログイン済み
  - Claude Max サブスクリプション有効

使用:
  from claude_analyst import analyze_portfolio
  response = analyze_portfolio(results, track, holdings)
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("claude-analyst")

SCRIPT_DIR = Path(__file__).resolve().parent

# Claude CLI 設定
DEFAULT_MODEL = "sonnet"         # "sonnet" / "opus" / "haiku"
DEFAULT_TIMEOUT = 180            # 秒 (3分)
MAX_RETRIES = 2                  # 失敗時のリトライ回数


def _is_cli_available() -> bool:
    """`claude` CLI が使用可能かチェックする。"""
    return shutil.which("claude") is not None


def _parse_json_robust(text: str) -> Optional[dict]:
    """LLM出力のJSONをロバストにパースする。

    gemini_analyst._parse_json_robust と同等ロジック。

    Args:
        text: CLI応答テキスト

    Returns:
        パース辞書 or None
    """
    # マークダウンコードブロック除去
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 文字列中の生改行をスペース化
    def _fix_newlines(s: str) -> str:
        result = []
        in_str = False
        escape = False
        for ch in s:
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == '\\':
                result.append(ch)
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
            result.append(' ' if (in_str and ch == '\n') else ch)
        return ''.join(result)

    text = _fix_newlines(text)
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON修復後もパース失敗: %s", e)
        return None


def build_analysis_prompt(
    results: dict,
    track: dict,
    holdings: dict,
) -> str:
    """Claude に送信する分析プロンプトを構築する。

    gemini_analyst.build_analysis_prompt と同等の内容を、
    Claude が得意とする構造化指示形式で組み立てる。

    Args:
        results: latest_evolution_results.json
        track: ai_track_record.json
        holdings: portfolio_holdings.json

    Returns:
        プロンプト文字列
    """
    from datetime import datetime

    tickers = results.get("tickers", [])
    allocs = results.get("allocations", {})
    ensemble = allocs.get("ensemble", {})
    bl = allocs.get("bl", {})
    ew = results.get("ensemble_weights", [0.25, 0.25, 0.25, 0.25])
    regime = results.get("regime", "unknown")
    kl_val = results.get("kl_value", 0)
    advanced = results.get("advanced_signals", [])

    regime_jp = {
        "low_vol":    "低ボラティリティ（安定上昇局面）",
        "transition": "移行期（ボラ上昇中）",
        "crisis":     "危機（高ボラ・急落局面）",
    }.get(regime, "不明")

    today = datetime.now().strftime("%Y-%m-%d")
    names = ["NCO", "RP", "MV", "MD"]

    lines = [
        f"# ポートフォリオ分析リクエスト ({today})",
        "",
        f"あなたは自己学習型クオンツAIポートフォリオアドバイザーです。",
        f"以下のデータを分析し、**厳密にJSON形式のみ**で回答してください。",
        f"JSON以外のテキスト（説明文・マークダウン・コードブロック）は一切含めないでください。",
        "",
        f"## マーケット環境",
        f"- レジーム: {regime_jp} (KL={kl_val:.4f})",
        f"- Ensemble重み: " + ", ".join(f"{n}={w:.1%}" for n, w in zip(names, ew)),
        "",
        f"## 数学的最適配分",
    ]
    for t in tickers:
        lines.append(
            f"- **{t}**: Ensemble={ensemble.get(t, 0):.1%} / BL={bl.get(t, 0):.1%}"
        )

    # Advanced AI signals
    if advanced:
        lines.append("")
        lines.append("## 高精度予測アンサンブル（5モデル統合）")
        strong = [s for s in advanced if "STRONG" in s.get("signal", "")]
        for s in strong[:5]:
            sub = s.get("sub_scores", {})
            sub_text = " / ".join(
                f"{k[:6]}={v:+.2f}" for k, v in sub.items()
            )
            lines.append(
                f"- **{s.get('ticker', '?')}**: `{s.get('signal', 'HOLD')}` "
                f"合成{s.get('composite_score', 0):+.3f} 確信度{s.get('confidence', 0):.0%}"
            )
            lines.append(f"  _{sub_text}_")

    # 保有状況
    lines.append("")
    lines.append("## 保有ポートフォリオ")
    for acct in ["tokutei", "nisa"]:
        for s in holdings.get("us_stocks", {}).get(acct, []):
            ticker = s.get("ticker", "")
            shares = s.get("shares", 0)
            cost = s.get("cost_basis_usd", 0)
            price = s.get("current_price_usd", 0)
            pnl = s.get("unrealized_pnl_usd", 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"- {ticker}: {shares}株 ${cost:.0f}→${price:.0f} ({sign}${pnl:.0f})"
            )

    # 自己反省
    evals = track.get("evaluations", [])
    if evals:
        lines.append("")
        lines.append("## 自己反省（直近予測精度）")
        for e in evals[-3:]:
            pva = e.get("predicted_vs_actual", {})
            lines.append(
                f"- [{e.get('eval_date', '')}] RMSE={pva.get('rmse', 0):.4f} "
                f"方向精度={pva.get('direction_accuracy', 0):.1%}"
            )

    # 出力スキーマ
    ticker_keys = ", ".join(f'"{t}": 0.XX' for t in tickers)
    action_keys = ", ".join(f'"{t}": "BUY|HOLD|SELL"' for t in tickers[:2]) + ", ..."
    reason_keys = ", ".join(f'"{t}": "理由"' for t in tickers[:2]) + ", ..."

    lines.extend([
        "",
        "## 出力JSON",
        "```",
        "{",
        f'  "allocations": {{{ticker_keys}}},',
        '  "confidence": 0.XX,',
        '  "reasoning": "全体推奨理由（1-2行）",',
        f'  "actions": {{{action_keys}}},',
        f'  "action_reasons": {{{reason_keys}}},',
        '  "tsumitate_advice": {',
        '    "keep_current": true,',
        '    "changes": ["変更提案"],',
        '    "reasoning": "積立分析1行"',
        '  },',
        '  "new_picks": [',
        '    {"ticker": "XXX", "sector": "セクター", "reason": "理由"}',
        '  ],',
        '  "risk_scenarios": {',
        '    "bull": {"probability": 0.3, "description": "強気シナリオ"},',
        '    "base": {"probability": 0.5, "description": "基本シナリオ"},',
        '    "bear": {"probability": 0.2, "description": "弱気シナリオ"}',
        '  },',
        '  "rebalance_opinion": {',
        '    "agree_with_proposals": true,',
        '    "override_reason": "理由",',
        '    "additional_swaps": []',
        '  }',
        "}",
        "```",
        "",
        "## 絶対ルール",
        "1. allocations の合計は1.0 (正規化)",
        "2. actions は全保有銘柄 (" + ", ".join(tickers) + ") に設定",
        "3. 文字列値内に改行禁止 (全て1行)",
        "4. 長期保持を基本方針、HOLDを優先、SELLは過熱サイン確定時のみ",
        "5. Advanced AI の STRONG シグナルは重み付けして判断",
        "6. 確信度はレジーム危機時に低く、安定時に高く動的に算出",
    ])

    return "\n".join(lines)


def call_claude_cli(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """`claude -p` を subprocess 経由で呼び出す。

    Claude Max サブスクリプションで認証済みの CLI を使用するため
    追加API課金なし。

    Args:
        prompt: プロンプト本文
        model: "sonnet" / "opus" / "haiku"
        timeout: 秒単位タイムアウト

    Returns:
        stdout応答テキスト、失敗時 None
    """
    if not _is_cli_available():
        logger.warning("`claude` CLI が見つかりません")
        return None

    try:
        start = time.time()
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            logger.error(
                "claude CLI 失敗 (code=%d): %s",
                result.returncode, result.stderr[:200]
            )
            return None

        logger.info("✅ Claude CLI応答取得 (%.1f秒, model=%s)", elapsed, model)
        return result.stdout

    except subprocess.TimeoutExpired:
        logger.error("claude CLI タイムアウト (%d秒)", timeout)
        return None
    except FileNotFoundError:
        logger.error("claude CLI 実行失敗: コマンド未検出")
        return None
    except Exception as e:
        logger.error("claude CLI 予期しないエラー: %s", e)
        return None


def analyze_portfolio(
    results: dict,
    track: dict,
    holdings: dict,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Claude CLI でポートフォリオ分析を実行する。

    Args:
        results: latest_evolution_results.json
        track: ai_track_record.json
        holdings: portfolio_holdings.json
        model: 使用モデル ("sonnet"推奨 / "opus"は高品質だが時間増)

    Returns:
        パース済みJSON応答、失敗時 None
    """
    if not _is_cli_available():
        logger.warning("claude CLI なし → スキップ")
        return None

    prompt = build_analysis_prompt(results, track, holdings)
    logger.info("📝 プロンプト生成完了 (%d文字)", len(prompt))

    for attempt in range(1, MAX_RETRIES + 2):
        logger.info("🤖 Claude CLI 呼び出し (試行 %d)...", attempt)
        response = call_claude_cli(prompt, model=model)
        if not response:
            if attempt <= MAX_RETRIES:
                time.sleep(5 * attempt)
                continue
            return None

        data = _parse_json_robust(response)
        if data is None:
            logger.warning("JSONパース失敗 (試行%d)", attempt)
            if attempt <= MAX_RETRIES:
                time.sleep(3)
                continue
            return None

        if "actions" not in data or "allocations" not in data:
            logger.warning("応答に必須フィールド欠落 (試行%d)", attempt)
            if attempt <= MAX_RETRIES:
                continue
            return None

        # allocations 合計を1.0に正規化
        allocs = data.get("allocations", {})
        total = sum(v for v in allocs.values() if isinstance(v, (int, float)))
        if total > 0:
            data["allocations"] = {k: v / total for k, v in allocs.items()}

        logger.info("✅ Claude分析完了 (確信度=%.2f)", data.get("confidence", 0))
        return data

    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # 自己テスト: CLI可用性確認
    if _is_cli_available():
        print("✅ claude CLI 利用可能")
    else:
        print("❌ claude CLI 未インストール")
