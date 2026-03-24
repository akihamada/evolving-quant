# -*- coding: utf-8 -*-
"""
Portfolio Analysis Bot — メインオーケストレーター
================================================
1. config.json 読み込み
2. data_fetcher でマーケットデータ収集
3. ai_analyzer で Gemini 構造化分析
4. notifier で Discord Embed 通知

ローカル実行: python main.py
GitHub Actions: Cron で自動実行
"""

import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv(Path(__file__).resolve().parent / ".env")

import data_fetcher
import ai_analyzer
import notifier

# ==============================================================================
# ログ設定
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("portfolio-bot")


# ==============================================================================
# メイン処理
# ==============================================================================


def load_config() -> dict:
    """
    config.json をロードする。

    Returns:
        設定辞書

    Raises:
        SystemExit: ファイルが見つからない・パース失敗時
    """
    config_path = Path(__file__).resolve().parent / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.critical("config.json が見つかりません: %s", config_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical("config.json のパースに失敗: %s", e)
        sys.exit(1)


def main() -> None:
    """
    メイン処理フロー。

    データ収集 → AI分析 → Discord通知 の3ステップを順次実行。
    各ステップで致命的エラーが発生してもログを出して続行を試みる。
    """
    logger.info("=" * 60)
    logger.info("ポートフォリオ分析Bot 🚀 実行開始")
    logger.info("=" * 60)

    start_time = time.time()

    # --- Step 1: 設定読み込み ---
    config = load_config()
    holdings = config.get("portfolio", {}).get("holdings", [])
    logger.info("ポートフォリオ: %d銘柄", len(holdings))

    # --- Step 2: データ収集 ---
    try:
        market_data = data_fetcher.collect_all_data(config)
    except Exception as e:
        logger.critical("データ収集で致命的エラー: %s", e, exc_info=True)
        sys.exit(1)

    # --- Step 3: AI分析 ---
    try:
        analysis = ai_analyzer.analyze_portfolio(market_data)
        if analysis is None:
            logger.error("AI分析が失敗しました（Noneが返された）")
            sys.exit(1)
    except Exception as e:
        logger.critical("AI分析で致命的エラー: %s", e, exc_info=True)
        sys.exit(1)

    # --- Step 4: Discord通知 ---
    try:
        success = notifier.notify(analysis)
        if success:
            logger.info("✅ Discord通知 送信成功")
        else:
            logger.error("❌ Discord通知 一部または全体が失敗")
    except Exception as e:
        logger.error("Discord通知エラー: %s", e, exc_info=True)

    # --- 完了 ---
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("実行完了 ⏱ %.1f秒", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
