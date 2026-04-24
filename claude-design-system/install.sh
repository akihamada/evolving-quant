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

{
  echo
  echo "done."
  echo "  ~/.claude/CLAUDE.md"
  echo "  ~/.claude/design/{cursor,notion,stripe,linear,vercel}.md"
  echo "  ~/.claude/design/jp-ui-contracts/"
  echo "  ~/.claude/skills/design-system-clean/SKILL.md"
} >&2
