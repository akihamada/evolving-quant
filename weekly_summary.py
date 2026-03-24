#!/usr/bin/env python3
"""
weekly_summary.py — 週次サマリーをmacOS通知で送信

毎週土曜に実行され、今週の分析結果を要約して
macOS通知 + テキストファイルで報告する。

使用:
    python weekly_summary.py
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weekly-summary")

SCRIPT_DIR = Path(__file__).resolve().parent
TRACK_RECORD_PATH = SCRIPT_DIR / "ai_track_record.json"
SUMMARY_PATH = SCRIPT_DIR / "data" / "weekly_summary.txt"


def load_json(path: Path) -> dict:
    """JSONファイルを安全に読み込む。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_week_records(track: dict) -> list:
    """今週（過去7日間）のレコードを取得する。"""
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    records = track.get("records", [])
    return [r for r in records if r.get("date", "") >= cutoff]


def build_summary(week_records: list) -> str:
    """
    週次サマリーテキストを生成する。

    Args:
        week_records: 今週のレコードリスト

    Returns:
        サマリーテキスト
    """
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"📊 週次クオンツレポート ({today})", "=" * 40, ""]

    if not week_records:
        lines.append("今週の分析データはありません。")
        return "\n".join(lines)

    # 基本統計
    confidences = [r.get("confidence", 0) for r in week_records]
    avg_conf = sum(confidences) / len(confidences)
    lines.append(f"■ 分析回数: {len(week_records)}回")
    lines.append(f"■ 平均確信度: {avg_conf:.2f}")
    lines.append("")

    # 最新のアクション集計
    latest = week_records[-1]
    actions = latest.get("actions", {})
    buys = [t for t, a in actions.items() if a == "BUY"]
    sells = [t for t, a in actions.items() if a == "SELL"]
    holds = [t for t, a in actions.items() if a == "HOLD"]

    lines.append("■ 最新アクション判定:")
    if buys:
        lines.append(f"  🟢 BUY: {', '.join(buys)}")
    if sells:
        lines.append(f"  🔴 SELL: {', '.join(sells)}")
    lines.append(f"  🟡 HOLD: {len(holds)}銘柄")
    lines.append("")

    # 新規推奨
    picks = latest.get("new_picks", [])
    if picks:
        lines.append("■ 新規推奨銘柄:")
        for p in picks:
            if isinstance(p, dict):
                lines.append(f"  {p.get('ticker', '?')} — {p.get('reason', '')[:40]}")
    lines.append("")

    # リスクシナリオ
    risk = latest.get("risk_scenarios", {})
    if risk:
        lines.append("■ リスクシナリオ:")
        for scenario in ["bull", "base", "bear"]:
            s = risk.get(scenario, {})
            prob = s.get("probability", 0)
            desc = s.get("description", "")[:50]
            lines.append(f"  {scenario}: {prob:.0%} — {desc}")

    return "\n".join(lines)


def send_macos_notification(title: str, message: str) -> None:
    """
    macOS 通知を送信する。

    Args:
        title: 通知タイトル
        message: 通知メッセージ（短縮版）
    """
    script = f'''
    display notification "{message}" with title "{title}" sound name "Glass"
    '''
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True, timeout=10,
        )
        logger.info("✅ macOS通知を送信しました")
    except Exception as e:
        logger.warning("⚠️ 通知送信失敗: %s", e)


def main() -> None:
    """メインエントリポイント。"""
    track = load_json(TRACK_RECORD_PATH)
    week_records = get_week_records(track)

    summary = build_summary(week_records)

    # テキストファイルに保存
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    logger.info("📝 サマリー保存: %s", SUMMARY_PATH)

    # macOS通知
    if week_records:
        latest = week_records[-1]
        conf = latest.get("confidence", 0)
        actions = latest.get("actions", {})
        buys = [t for t, a in actions.items() if a == "BUY"]
        note_msg = f"確信度{conf:.0%} | BUY: {', '.join(buys[:4])}"
    else:
        note_msg = "今週の分析データなし"

    send_macos_notification("🧬 週次クオンツレポート", note_msg)

    # コンソール出力
    print(summary)


if __name__ == "__main__":
    main()
