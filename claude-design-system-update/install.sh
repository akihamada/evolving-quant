#!/usr/bin/env bash
# Global Claude Code design system bootstrap.
#
# Usage (via curl / recommended):
#   curl -fsSL https://raw.githubusercontent.com/akihamada/claude-design-system/main/install.sh | bash
#
# Usage (local):
#   bash install.sh

set -euo pipefail

: "${DESIGN_REPO_RAW:=https://raw.githubusercontent.com/akihamada/claude-design-system/main}"

mkdir -p "$HOME/.claude/design" "$HOME/.claude/skills"

# 1. Global CLAUDE.md (backup if exists, then fetch latest from repo)
if [[ -f "$HOME/.claude/CLAUDE.md" ]]; then
  cp "$HOME/.claude/CLAUDE.md" "$HOME/.claude/CLAUDE.md.bak.$(date +%Y%m%d%H%M%S)"
  echo "backed up existing ~/.claude/CLAUDE.md" >&2
fi
curl -fsSL "$DESIGN_REPO_RAW/CLAUDE.md" -o "$HOME/.claude/CLAUDE.md"

# 2. Brand DESIGN.md (5 brands)
cd /tmp
for pair in "cursor:cursor" "notion:notion" "stripe:stripe" "linear.app:linear" "vercel:vercel"; do
  slug="${pair%%:*}"
  name="${pair##*:}"
  npx -y getdesign@latest add "$slug" --force --out "$HOME/.claude/design/${name}.md"
done

# 3. jp-ui-contracts (clone / pull)
if [[ -d "$HOME/.claude/design/jp-ui-contracts/.git" ]]; then
  git -C "$HOME/.claude/design/jp-ui-contracts" pull --ff-only
else
  rm -rf "$HOME/.claude/design/jp-ui-contracts"
  git clone --depth=1 https://github.com/hirokaji/jp-ui-contracts.git \
    "$HOME/.claude/design/jp-ui-contracts"
fi

# 4. TypeUI Clean skill
TMP_TYPEUI="$(mktemp -d)"
(cd "$TMP_TYPEUI" && npx -y typeui.sh pull clean --providers claude-code)
mkdir -p "$HOME/.claude/skills/design-system-clean"
cp "$TMP_TYPEUI/.claude/skills/design-system/SKILL.md" \
   "$HOME/.claude/skills/design-system-clean/SKILL.md"
rm -rf "$TMP_TYPEUI"

# 5. Google Cloud / AI skills (BigQuery, Cloud Run, Firebase, Gemini, GKE, AlloyDB, etc.)
npx -y skills add google/skills -g -y -s '*' -a claude-code 1>&2

# 6. Light mode resource controls (apply once if not already configured)
if command -v jq >/dev/null 2>&1; then
  if [[ ! -f "$HOME/.claude/settings.json" ]] || ! jq -e 'has("skillListingMaxDescChars")' "$HOME/.claude/settings.json" >/dev/null 2>&1; then
    LIGHT_SETTINGS_JSON='{
  "skillListingMaxDescChars": 256,
  "skillListingBudgetFraction": 0.005,
  "skillOverrides": {
    "alloydb-basics": "name-only",
    "bigquery-basics": "name-only",
    "cloud-run-basics": "name-only",
    "cloud-sql-basics": "name-only",
    "firebase-basics": "name-only",
    "gemini-api": "name-only",
    "gke-basics": "name-only",
    "google-cloud-recipe-auth": "name-only",
    "google-cloud-recipe-networking-observability": "name-only",
    "google-cloud-recipe-onboarding": "name-only",
    "google-cloud-waf-cost-optimization": "name-only",
    "google-cloud-waf-reliability": "name-only",
    "google-cloud-waf-security": "name-only"
  }
}'
    if [[ -f "$HOME/.claude/settings.json" ]]; then
      cp "$HOME/.claude/settings.json" "$HOME/.claude/settings.json.bak.$(date +%Y%m%d%H%M%S)"
      jq -s '.[0] * .[1]' "$HOME/.claude/settings.json" <(echo "$LIGHT_SETTINGS_JSON") > "$HOME/.claude/settings.json.tmp"
      mv "$HOME/.claude/settings.json.tmp" "$HOME/.claude/settings.json"
    else
      mkdir -p "$HOME/.claude"
      echo "$LIGHT_SETTINGS_JSON" > "$HOME/.claude/settings.json"
    fi
    echo "applied Light mode resource limits to ~/.claude/settings.json" >&2
  fi
else
  echo "warning: jq not found; skipping Light mode settings (install jq to enable)" >&2
fi

{
  echo
  echo "done."
  echo "  ~/.claude/CLAUDE.md (light: ~3 KB, no @-imports)"
  echo "  ~/.claude/design/{cursor,notion,stripe,linear,vercel}.md"
  echo "  ~/.claude/design/jp-ui-contracts/"
  echo "  ~/.claude/skills/design-system-clean/SKILL.md"
  echo "  ~/.claude/skills/google-* (14 skills)"
  echo "  ~/.claude/settings.json (Light mode applied if missing key)"
} >&2
