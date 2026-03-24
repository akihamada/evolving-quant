#!/bin/bash
# ==============================================================================
# run_evolution.sh — 自己進化型クオンツ・ブートストラップスクリプト
# ==============================================================================
# 以下を自動実行:
#   0. launchd 自己登録（未登録なら plist コピー + load）
#   0.5. ログローテーション（肥大化防止）
#   1. venv の存在チェック → 無ければ自動構築
#   2. daily_evolution.py の実行
#   3. auto_prompt_cycle.py（Antigravity 経由でプロンプト送信）
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="/tmp/quant_venv"
PYTHON="/usr/bin/python3"
LOG_PREFIX="[quant-evolution]"
PLIST_NAME="com.akihome.quant-evolution.plist"
PLIST_SRC="${SCRIPT_DIR}/${PLIST_NAME}"
PLIST_DST="${HOME}/Library/LaunchAgents/${PLIST_NAME}"
STDOUT_LOG="/tmp/quant-evolution-stdout.log"
STDERR_LOG="/tmp/quant-evolution-stderr.log"
MAX_LOG_LINES=1000
KEEP_LOG_LINES=500

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} $1"
}

# --- Step 0: launchd 自己登録 ---
if [ -f "${PLIST_SRC}" ]; then
    if [ ! -f "${PLIST_DST}" ]; then
        log "📋 plist を LaunchAgents にコピー中..."
        cp "${PLIST_SRC}" "${PLIST_DST}" 2>/dev/null && \
            log "✅ plist コピー完了: ${PLIST_DST}" || \
            log "⚠️  plist コピー失敗（権限不足の可能性）"
    fi
    if ! launchctl list 2>/dev/null | grep -q "com.akihome.quant-evolution"; then
        log "🔧 launchd に登録中..."
        launchctl load "${PLIST_DST}" 2>/dev/null && \
            log "✅ launchd 登録完了" || \
            log "⚠️  launchd 登録スキップ（既存 or 権限不足）"
    else
        log "✅ launchd 登録済"
    fi
fi

# --- Step 0.5: ログローテーション（肥大化防止） ---
for LOG_FILE in "${STDOUT_LOG}" "${STDERR_LOG}"; do
    if [ -f "${LOG_FILE}" ]; then
        LINE_COUNT=$(wc -l < "${LOG_FILE}" 2>/dev/null || echo "0")
        if [ "${LINE_COUNT}" -gt "${MAX_LOG_LINES}" ]; then
            tail -n "${KEEP_LOG_LINES}" "${LOG_FILE}" > "${LOG_FILE}.tmp" && \
                mv "${LOG_FILE}.tmp" "${LOG_FILE}"
            log "🗑️ ログローテーション: ${LOG_FILE} (${LINE_COUNT}→${KEEP_LOG_LINES}行)"
        fi
    fi
done

# --- Step 1: venv チェック & 自動構築 ---
if [ ! -f "${VENV_DIR}/bin/python" ]; then
    log "⚙️  venv が見つかりません。自動構築開始..."
    ${PYTHON} -m venv "${VENV_DIR}"
    log "📦 依存パッケージをインストール中..."
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip --quiet 2>&1 || true
    "${VENV_DIR}/bin/pip" install \
        streamlit plotly scipy yfinance numpy pandas python-dotenv \
        --quiet 2>&1
    log "✅ venv 構築完了: ${VENV_DIR}"
else
    log "✅ venv 確認済: ${VENV_DIR}"
fi

# --- 土曜日判定: 週次サマリーのみ ---
if [ "$(date +%u)" -eq 6 ]; then
    log "📊 土曜日: 週次サマリー生成のみ実行..."
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/weekly_summary.py"
    log "🎉 週次サマリー完了"
    exit 0
fi

# --- Step 1.5: 積立自動反映チェック ---
log "📊 積立反映チェック..."
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/update_holdings.py"

# --- Step 2: daily_evolution.py 実行 ---
log "🧬 Daily Evolution Engine 起動..."
cd "${SCRIPT_DIR}"
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/daily_evolution.py"
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    log "🎉 Daily Evolution 正常完了 (exit=${EXIT_CODE})"

    # --- Step 3: 自動プロンプトサイクル ---
    log "🤖 Auto Prompt Cycle 起動..."
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/auto_prompt_cycle.py"
    PROMPT_EXIT=$?
    if [ ${PROMPT_EXIT} -eq 0 ]; then
        log "🎉 Auto Prompt Cycle 正常完了"
    else
        log "⚠️  Auto Prompt Cycle 失敗 (exit=${PROMPT_EXIT}) — 手動で Record タブから入力可能"
    fi
else
    log "❌ Daily Evolution 失敗 (exit=${EXIT_CODE})"
fi

exit ${EXIT_CODE}
