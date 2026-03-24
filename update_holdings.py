#!/usr/bin/env python3
"""
update_holdings.py — 積立購入の自動反映

毎月の積立注文日（order_day）に portfolio_holdings.json を自動更新する。
投資信託の口数を概算追加し、next_order を翌月に更新。

使用:
    python update_holdings.py           # 通常実行
    python update_holdings.py --dry-run # 変更なしでプレビュー
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("update-holdings")

SCRIPT_DIR = Path(__file__).resolve().parent
HOLDINGS_PATH = SCRIPT_DIR / "data" / "portfolio_holdings.json"
PURCHASE_LOG_PATH = SCRIPT_DIR / "data" / "purchase_log.json"


def load_json(path: Path) -> Optional[dict]:
    """JSONファイルを安全に読み込む。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("JSON読み込み失敗: %s — %s", path, e)
        return None


def save_json(path: Path, data: dict) -> None:
    """JSONファイルに保存する。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_next_order_date(current_next: str, order_day: int) -> str:
    """
    次回注文日を算出する。

    Args:
        current_next: 現在のnext_order文字列 (YYYY-MM-DD)
        order_day: 毎月の注文日

    Returns:
        次回のnext_order (YYYY-MM-DD)
    """
    dt = datetime.strptime(current_next, "%Y-%m-%d")
    # 翌月の同日
    if dt.month == 12:
        next_dt = dt.replace(year=dt.year + 1, month=1, day=order_day)
    else:
        try:
            next_dt = dt.replace(month=dt.month + 1, day=order_day)
        except ValueError:
            # 月末調整（例: 31日がない月）
            last_day = (dt.replace(month=dt.month + 2, day=1) - timedelta(days=1)).day
            next_dt = dt.replace(month=dt.month + 1, day=min(order_day, last_day))
    return next_dt.strftime("%Y-%m-%d")


def find_fund_in_holdings(holdings: dict, fund_name: str) -> Optional[dict]:
    """
    投資信託の保有データを名前で検索する。

    Args:
        holdings: portfolio_holdings全体
        fund_name: ファンド名

    Returns:
        該当するファンドデータの辞書（参照）
    """
    for section_key in ["nisa_growth", "nisa_tsumitate"]:
        section = holdings.get("mutual_funds", {}).get(section_key, [])
        for fund in section:
            if fund.get("name") == fund_name:
                return fund
    return None


def apply_tsumitate(holdings: dict, dry_run: bool = False) -> list:
    """
    今日が積立注文日のファンドを更新する。

    Args:
        holdings: portfolio_holdings全体
        dry_run: Trueなら実際の更新を行わない

    Returns:
        適用された変更のリスト
    """
    today = datetime.now().strftime("%Y-%m-%d")
    today_day = datetime.now().day
    changes = []

    plans = holdings.get("tsumitate_settings", {}).get("plans", [])
    for plan in plans:
        order_day = plan.get("order_day", 0)
        next_order = plan.get("next_order", "")
        fund_name = plan.get("fund", "")
        monthly_jpy = plan.get("monthly_jpy", 0)

        # 今日がnext_orderの日付か確認
        if next_order != today:
            continue

        logger.info("📊 積立日検出: %s (¥%s)", fund_name[:20], f"{monthly_jpy:,}")

        # 該当ファンドを検索
        fund = find_fund_in_holdings(holdings, fund_name)
        if not fund:
            logger.warning("⚠️ ファンドが見つかりません: %s", fund_name)
            continue

        current_nav = fund.get("current_nav", 0)
        if current_nav <= 0:
            logger.warning("⚠️ NAVが0以下: %s", fund_name)
            continue

        # 概算口数 = 投資金額(円) ÷ 基準価額(円/万口) × 10000
        estimated_units = int(monthly_jpy / current_nav * 10000)
        old_units = fund.get("units", 0)
        new_units = old_units + estimated_units

        # 加重平均コスト更新
        old_cost = fund.get("cost_nav", current_nav)
        # 新しい加重平均 = (旧コスト×旧口数 + 現NAV×追加口数) / 新口数
        if new_units > 0:
            new_cost_nav = (old_cost * old_units + current_nav * estimated_units) / new_units
        else:
            new_cost_nav = current_nav

        change = {
            "date": today,
            "type": "tsumitate",
            "fund": fund_name,
            "amount_jpy": monthly_jpy,
            "estimated_units": estimated_units,
            "nav_at_purchase": current_nav,
            "units_before": old_units,
            "units_after": new_units,
            "cost_nav_before": old_cost,
            "cost_nav_after": round(new_cost_nav),
        }
        changes.append(change)

        if not dry_run:
            fund["units"] = new_units
            fund["cost_nav"] = round(new_cost_nav)
            # next_order を翌月に更新
            plan["next_order"] = get_next_order_date(next_order, order_day)
            logger.info(
                "✅ 更新: %s 口数 %d→%d (+%d), 次回=%s",
                fund_name[:20], old_units, new_units, estimated_units,
                plan["next_order"],
            )
        else:
            logger.info(
                "🔍 [DRY-RUN] %s 口数 %d→%d (+%d)",
                fund_name[:20], old_units, new_units, estimated_units,
            )

    return changes


def log_purchases(changes: list) -> None:
    """
    購入履歴を purchase_log.json に追記する。

    Args:
        changes: 適用された変更のリスト
    """
    if not changes:
        return

    log_data = load_json(PURCHASE_LOG_PATH) or {"purchases": []}
    log_data["purchases"].extend(changes)

    # 最新200件に制限
    if len(log_data["purchases"]) > 200:
        log_data["purchases"] = log_data["purchases"][-200:]

    save_json(PURCHASE_LOG_PATH, log_data)
    logger.info("📝 購入ログ更新: %d件追記", len(changes))


def main() -> None:
    """メインエントリポイント。"""
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("🔍 DRY-RUN モード（変更は保存されません）")

    holdings = load_json(HOLDINGS_PATH)
    if not holdings:
        logger.error("portfolio_holdings.json が読み込めません")
        sys.exit(1)

    changes = apply_tsumitate(holdings, dry_run=dry_run)

    if changes:
        if not dry_run:
            save_json(HOLDINGS_PATH, holdings)
            log_purchases(changes)
            logger.info("🎉 積立反映完了: %d件", len(changes))
        else:
            logger.info("🔍 [DRY-RUN] %d件の変更を検出", len(changes))
    else:
        logger.info("📅 今日は積立注文日ではありません")


if __name__ == "__main__":
    main()
