# Global User Instructions

## UI/デザイン実装ルール

UIを実装するときは必ず `~/.claude/design/` 配下の `DESIGN.md` を参照してスタイルを適用すること。

### HTML/UI のデフォルトベース

**HTML/UI 実装の最初に必ず Read すること:**
- `~/.claude/design/cursor.md`（デフォルトベース）

別ブランドが明示指定された場合は該当ファイルへ:

- `~/.claude/design/notion.md` — ドキュメント・LP・読み物系
- `~/.claude/design/stripe.md` — 決済・フォーム系
- `~/.claude/design/linear.md` — ダッシュボード・ツール系
- `~/.claude/design/vercel.md` — 汎用モダン

プロジェクト内に `./DESIGN.md` もしくは `./design/*.md` がある場合は、そちらをグローバルより優先すること。

### 日本語UI 共通契約

日本語UIを含む実装では併せて Read:

- `~/.claude/design/jp-ui-contracts/templates/base/DESIGN.md` — 最小共通契約

文脈に応じて追加 Read:

- `~/.claude/design/jp-ui-contracts/templates/media/DESIGN.md` — 記事・長文（line-height 1.75–2.0）
- `~/.claude/design/jp-ui-contracts/templates/saas/DESIGN.md` — 業務UI・管理画面
- `~/.claude/design/jp-ui-contracts/templates/docs/DESIGN.md` — 技術文書
- `~/.claude/design/jp-ui-contracts/templates/dashboard/DESIGN.md` — BI・高密度

CSS Recipe（必要時）:
- `~/.claude/design/jp-ui-contracts/recipes/{ja-text,mixed-script,headings,forms}.css`

生成後レビュー:
- `~/.claude/design/jp-ui-contracts/validators/review-checklist.md`
- `~/.claude/design/jp-ui-contracts/validators/lint-rules.md`

### 適用順序

ブランド系 DESIGN.md を見た目の土台にしつつ、日本語要素には jp-ui-contracts を重ねる。矛盾したら**日本語側を優先**（本文の読みやすさが最優先）。

### 禁止事項（jp-ui-contracts より）

- `word-break: break-all` を全体既定で使わない
- 日本語本文に強い letter-spacing を既定適用しない（0.02em を大きく超える tracking は禁止）
- 日本語本文で line-height 1.5 未満を既定にしない
- 日本語 fallback 未明示（ブラウザ既定任せ）にしない
- `palt` を本文に全体適用しない（見出し・ナビゲーションで実読確認後のみ）
- 表・フォームに本文ルールをそのまま継承させない

### スタイル方向性スキル

抽象的なスタイル方向性で作りたいときは Skills を使う。

- `design-system-clean` — 余白を広く、限定カラー、タイポ優先のミニマル

### ワークフロー

1. 契約（DESIGN.md + jp-ui-contracts テンプレ）を選ぶ
2. AIで生成
3. `validators/review-checklist.md` の観点で目視
4. 差分を DESIGN.md に戻す（契約修正）
5. 再生成

一発で当てるフローではなく、**契約 → 生成 → 目視 → 契約修正**のループで育てる。
