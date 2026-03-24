#!/usr/bin/env python3
"""
purchase_handler.py — 手動購入をポートフォリオに反映

この会話で「/buy MRVL 3 $88.50」のようなメッセージを受け取り、
portfolio_holdings.json に反映する。

使用:
    python purchase_handler.py "MRVL 3 88.50"       # 米国株（USD）
    python purchase_handler.py "1328 2 18200 JPY"    # 日本株（JPY）
    python purchase_handler.py --list                 # 現在のポジション一覧
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("purchase-handler")

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


def parse_buy_command(text: str) -> Optional[dict]:
    """
    購入コマンドをパースする。

    対応フォーマット:
        MRVL 3 88.50        → USD (デフォルト)
        MRVL 3 $88.50       → USD
        1328 2 18200 JPY    → JPY
        1328 2 ¥18200       → JPY

    Args:
        text: コマンド文字列（/buy プレフィックスは除去済み）

    Returns:
        パース済み辞書 or None
    """
    text = text.strip()
    # /buy プレフィックスを除去
    if text.lower().startswith("/buy"):
        text = text[4:].strip()

    # $ や ¥ を除去して通貨判定
    currency = "USD"
    if "¥" in text or "JPY" in text.upper():
        currency = "JPY"

    text = text.replace("$", "").replace("¥", "").replace(",", "")
    text = re.sub(r"\s+JPY\s*$", "", text, flags=re.IGNORECASE)

    parts = text.split()
    if len(parts) < 3:
        logger.error("パースエラー: 最低3つの要素が必要です (ティッカー 株数 単価)")
        return None

    ticker = parts[0].upper()
    try:
        shares = int(parts[1])
        price = float(parts[2])
    except ValueError:
        logger.error("パースエラー: 株数=%s, 単価=%s", parts[1], parts[2])
        return None

    # 日本株の判定（数字4桁のコード or .T suffix）
    if ticker.isdigit() or ticker.endswith(".T"):
        currency = "JPY"

    return {
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "currency": currency,
    }


def apply_us_stock_purchase(holdings: dict, ticker: str, shares: int, price: float) -> dict:
    """
    米国株の購入をポートフォリオに反映する。

    既存ポジションがある場合は加重平均で取得単価を更新する。
    新規銘柄の場合はNISA枠に追加する。

    Args:
        holdings: portfolio_holdings全体
        ticker: ティッカーシンボル
        shares: 購入株数
        price: 購入単価(USD)

    Returns:
        変更内容の辞書
    """
    # 全セクションから既存ポジションを検索
    for section_key in ["tokutei", "nisa"]:
        section = holdings.get("us_stocks", {}).get(section_key, [])
        for position in section:
            if position.get("ticker") == ticker:
                old_shares = position["shares"]
                old_cost = position["cost_basis_usd"]
                new_shares = old_shares + shares
                new_cost = (old_cost * old_shares + price * shares) / new_shares

                change = {
                    "action": "update",
                    "section": section_key,
                    "shares_before": old_shares,
                    "shares_after": new_shares,
                    "cost_before": old_cost,
                    "cost_after": round(new_cost, 2),
                }

                position["shares"] = new_shares
                position["cost_basis_usd"] = round(new_cost, 2)
                return change

    # 新規銘柄 → NISA枠に追加
    new_position = {
        "ticker": ticker,
        "name": ticker,
        "exchange": "UNKNOWN",
        "shares": shares,
        "cost_basis_usd": price,
        "current_price_usd": price,
        "unrealized_pnl_usd": 0,
        "unrealized_pnl_jpy": 0,
        "sector": "Unknown",
    }
    holdings.setdefault("us_stocks", {}).setdefault("nisa", []).append(new_position)
    return {"action": "new", "section": "nisa"}


def apply_jp_stock_purchase(holdings: dict, code: str, shares: int, price: float) -> dict:
    """
    日本株の購入をポートフォリオに反映する。

    Args:
        holdings: portfolio_holdings全体
        code: 銘柄コード (例: "1328")
        shares: 購入株数
        price: 購入単価(JPY)

    Returns:
        変更内容の辞書
    """
    code = code.replace(".T", "")
    section = holdings.get("japan_stocks", {}).get("nisa_growth", [])
    for position in section:
        if position.get("code") == code:
            old_shares = position["shares"]
            old_cost = position["cost_basis_jpy"]
            new_shares = old_shares + shares
            new_cost = (old_cost * old_shares + price * shares) / new_shares

            change = {
                "action": "update",
                "shares_before": old_shares,
                "shares_after": new_shares,
                "cost_before": old_cost,
                "cost_after": round(new_cost),
            }

            position["shares"] = new_shares
            position["cost_basis_jpy"] = round(new_cost)
            return change

    # 新規
    new_position = {
        "code": code,
        "name": code,
        "shares": shares,
        "cost_basis_jpy": price,
        "current_price_jpy": price,
        "unrealized_pnl_jpy": 0,
        "sector": "Unknown",
    }
    holdings.setdefault("japan_stocks", {}).setdefault("nisa_growth", []).append(new_position)
    return {"action": "new"}


def process_buy(text: str) -> bool:
    """
    /buy コマンドを処理しポートフォリオを更新する。

    Args:
        text: コマンド文字列

    Returns:
        処理成功したかどうか
    """
    parsed = parse_buy_command(text)
    if not parsed:
        return False

    holdings = load_json(HOLDINGS_PATH)
    if not holdings:
        return False

    ticker = parsed["ticker"]
    shares = parsed["shares"]
    price = parsed["price"]
    currency = parsed["currency"]

    logger.info("📦 購入処理: %s %d株 @ %s%.2f", ticker, shares,
                "¥" if currency == "JPY" else "$", price)

    if currency == "JPY":
        change = apply_jp_stock_purchase(holdings, ticker, shares, price)
    else:
        change = apply_us_stock_purchase(holdings, ticker, shares, price)

    save_json(HOLDINGS_PATH, holdings)

    # 購入ログに記録
    log_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type": "manual_buy",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "currency": currency,
        **change,
    }
    log_data = load_json(PURCHASE_LOG_PATH) or {"purchases": []}
    log_data["purchases"].append(log_entry)
    if len(log_data["purchases"]) > 200:
        log_data["purchases"] = log_data["purchases"][-200:]
    save_json(PURCHASE_LOG_PATH, log_data)

    if change.get("action") == "update":
        logger.info("✅ 更新: %s %d→%d株, 取得単価 %.2f→%.2f",
                     ticker, change["shares_before"], change["shares_after"],
                     change["cost_before"], change["cost_after"])
    else:
        logger.info("✅ 新規追加: %s %d株 @ %.2f", ticker, shares, price)

    return True


def show_positions() -> None:
    """現在のポジション一覧を表示する。"""
    holdings = load_json(HOLDINGS_PATH)
    if not holdings:
        return

    print("\n📊 現在のポジション一覧")
    print("=" * 60)

    print("\n■ 米国株:")
    for section in ["tokutei", "nisa"]:
        for p in holdings.get("us_stocks", {}).get(section, []):
            tag = "[特定]" if section == "tokutei" else "[NISA]"
            print(f"  {tag} {p['ticker']:6s} {p['shares']:4d}株 "
                  f"取得${p['cost_basis_usd']:.2f}")

    print("\n■ 日本株:")
    for p in holdings.get("japan_stocks", {}).get("nisa_growth", []):
        print(f"  [{p['code']}] {p['name'][:10]} {p['shares']}株 "
              f"取得¥{p['cost_basis_jpy']}")

    print()


if __name__ == "__main__":
    if "--list" in sys.argv:
        show_positions()
    elif len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        success = process_buy(cmd)
        sys.exit(0 if success else 1)
    else:
        print("使用方法:")
        print('  python purchase_handler.py "MRVL 3 88.50"')
        print('  python purchase_handler.py "1328 2 18200 JPY"')
        print("  python purchase_handler.py --list")
