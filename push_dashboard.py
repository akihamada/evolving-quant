#!/usr/bin/env python3
"""
push_dashboard.py — ダッシュボードHTMLをGitHub Pagesに自動Push

generate_dashboard.py で生成した index.html を
GitHub APIで evolving-quant リポジトリに直接アップロードする。

使用:
    python push_dashboard.py
    python push_dashboard.py --dry-run
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("push-dashboard")

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_PATH = SCRIPT_DIR / "index.html"
REPO_OWNER = "akihamada"
REPO_NAME = "evolving-quant"
FILE_PATH = "index.html"
BRANCH = "main"


def _load_dotenv() -> None:
    """`.env` ファイルから環境変数を読み込む。"""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def get_current_sha() -> str:
    """GitHub上のファイルSHAを取得する。

    Returns:
        SHAハッシュ文字列（ファイルが存在しない場合は空文字列）
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data.get("sha", "")
    except urllib.error.HTTPError:
        return ""


def push_to_github(dry_run: bool = False) -> bool:
    """index.htmlをGitHub APIでPushする。

    Args:
        dry_run: Trueなら実際のPushを行わない

    Returns:
        成功したかどうか
    """
    _load_dotenv()
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error(
            "GITHUB_TOKEN が未設定です。.env に GITHUB_TOKEN=ghp_xxxx を追加してください"
        )
        return False

    if not HTML_PATH.exists():
        logger.error("index.html が見つかりません: %s", HTML_PATH)
        return False

    html_content = HTML_PATH.read_text(encoding="utf-8")
    html_size = len(html_content)
    logger.info("📄 index.html: %d bytes", html_size)

    if dry_run:
        logger.info("🔍 [DRY-RUN] Push をスキップ")
        return True

    sha = get_current_sha()
    content_b64 = base64.b64encode(html_content.encode("utf-8")).decode("ascii")

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    payload = {
        "message": "auto: update dashboard",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "Authorization": f"token {token}",
        },
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            new_sha = result.get("content", {}).get("sha", "?")
            logger.info("✅ GitHub Push完了: SHA=%s", new_sha[:12])
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("❌ GitHub Push失敗: %s — %s", e.code, body[:200])
        return False
    except Exception as e:
        logger.error("❌ Push失敗: %s", e)
        return False


def main() -> None:
    """メインエントリポイント。"""
    dry_run = "--dry-run" in sys.argv
    logger.info("🌐 ダッシュボード GitHub Push%s", " (dry-run)" if dry_run else "")
    ok = push_to_github(dry_run=dry_run)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
