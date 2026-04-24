#!/usr/bin/env bash
# Copy design-system bootstrap (setup-design-system/ + .claude/settings.json) into a target project repo.
# Usage: bash setup-design-system/propagate.sh <target-project-path>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target-project-path>" >&2
  exit 1
fi

TARGET="$(cd "$1" && pwd)"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "Error: $TARGET is not a git repository" >&2
  exit 1
fi

mkdir -p "$TARGET/.claude"

# setup-design-system/ (overwrites cleanly)
rm -rf "$TARGET/setup-design-system"
cp -r "$SRC/setup-design-system" "$TARGET/setup-design-system"

# .claude/settings.json (back up if exists — user merges manually if needed)
if [[ -f "$TARGET/.claude/settings.json" ]]; then
  BAK="$TARGET/.claude/settings.json.bak.$(date +%Y%m%d%H%M%S)"
  cp "$TARGET/.claude/settings.json" "$BAK"
  echo "warning: existing .claude/settings.json backed up to $(basename "$BAK")" >&2
  echo "         merge the SessionStart hook manually if the backup had other hooks" >&2
fi
cp "$SRC/.claude/settings.json" "$TARGET/.claude/settings.json"

cat <<EOF
done. copied to $TARGET:
  setup-design-system/
  .claude/settings.json

next:
  cd "$TARGET"
  git add setup-design-system .claude/settings.json
  git commit -m "feat: add global design system bootstrap"
  git push
EOF
