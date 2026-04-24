# claude-design-system

Claude Code 用グローバルデザインシステム。ローカル Mac でも web サンドボックスでも、一発で以下が揃う。

- `~/.claude/CLAUDE.md` — HTML/UI は Cursor ベース + 日本語UIは jp-ui-contracts を自動ロード
- `~/.claude/design/` — 5ブランド DESIGN.md（cursor / notion / stripe / linear / vercel）
- `~/.claude/design/jp-ui-contracts/` — 日本語UI契約キット（templates / recipes / validators）
- `~/.claude/skills/design-system-clean/SKILL.md` — TypeUI Clean スキル

## ローカルにインストール

```bash
curl -fsSL https://raw.githubusercontent.com/akihamada/claude-design-system/main/install.sh | bash
```

## Claude Code web サンドボックスで自動ブートストラップ

各プロジェクトの `.claude/settings.json` に以下を置くだけ:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "[[ -f ~/.claude/design/cursor.md ]] || curl -fsSL https://raw.githubusercontent.com/akihamada/claude-design-system/main/install.sh | bash 1>&2",
        "timeout": 300,
        "statusMessage": "Bootstrapping global design system..."
      }]
    }]
  }
}
```

web セッション開始時に hook が走り、`~/.claude/design/cursor.md` が無ければ install.sh が自動実行される。2回目以降は cursor.md 検出で即 skip（数ms）。

## 更新

install.sh を再実行すれば冪等に最新化:

```bash
curl -fsSL https://raw.githubusercontent.com/akihamada/claude-design-system/main/install.sh | bash
```

- `getdesign` は `--force` で上書き
- `jp-ui-contracts` は `git pull --ff-only`
- `typeui.sh` は再生成

既存の `~/.claude/CLAUDE.md` はタイムスタンプ付きで自動バックアップされる。

## 前提

- `node` / `npx`（getdesign, typeui 実行用）
- `git`（jp-ui-contracts clone 用）
- `curl`
- GitHub 到達性

## 構成要素

- [getdesign.md](https://github.com/getdesign-md/getdesign) — ブランド別 DESIGN.md
- [jp-ui-contracts](https://github.com/hirokaji/jp-ui-contracts) — 日本語UI契約キット
- [typeui.sh](https://typeui.sh) — スタイル方向性スキル

## ライセンス

MIT
