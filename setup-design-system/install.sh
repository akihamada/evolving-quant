#!/usr/bin/env bash
# Global design system bootstrap for Claude Code on Mac.
# Copies ~/.claude/CLAUDE.md and installs ~/.claude/design/ + skill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$HOME/.claude/design" "$HOME/.claude/skills"

# 1. Global CLAUDE.md (backup if exists)
if [[ -f "$HOME/.claude/CLAUDE.md" ]]; then
  cp "$HOME/.claude/CLAUDE.md" "$HOME/.claude/CLAUDE.md.bak.$(date +%Y%m%d%H%M%S)"
  echo "backed up existing ~/.claude/CLAUDE.md"
fi
cp "$SCRIPT_DIR/CLAUDE.md" "$HOME/.claude/CLAUDE.md"

# 2. Brand DESIGN.md (5 brands)
cd /tmp
for pair in "cursor:cursor" "notion:notion" "stripe:stripe" "linear.app:linear" "vercel:vercel"; do
  slug="${pair%%:*}"
  name="${pair##*:}"
  npx -y getdesign@latest add "$slug" --out "$HOME/.claude/design/${name}.md"
done

# 3. jp-ui-contracts (clone; updatable via `git -C ~/.claude/design/jp-ui-contracts pull`)
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

echo
echo "done."
echo "  ~/.claude/CLAUDE.md"
echo "  ~/.claude/design/{cursor,notion,stripe,linear,vercel}.md"
echo "  ~/.claude/design/jp-ui-contracts/"
echo "  ~/.claude/skills/design-system-clean/SKILL.md"
