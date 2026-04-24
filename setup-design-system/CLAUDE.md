# Global User Instructions

## UI/デザイン実装ルール

UIを実装するときは必ず `~/.claude/design/` 配下の `DESIGN.md` を参照してスタイルを適用すること。

---

### HTML/UI のデフォルトベース（自動読み込み・最優先）

HTMLやUIを新規に作るとき、明示的に別ブランドの指定がない限り以下の Cursor DESIGN.md をベースとする。

@~/.claude/design/cursor.md

### 日本語UI 共通契約（自動読み込み）

日本語UIを実装するとき、欧文前提のDESIGN.mdだけでは和文フォールバック、行間、改行、和欧混植が崩れるため、以下を常に重ねる。

@~/.claude/design/jp-ui-contracts/templates/base/DESIGN.md

---

### 他ブランド（明示指定時に Read）

ユーザーが以下のブランド名を指示したら、該当ファイルを Read してから適用する。

- `~/.claude/design/notion.md` — ドキュメント・LP・読み物系
- `~/.claude/design/stripe.md` — 決済・フォーム系
- `~/.claude/design/linear.md` — ダッシュボード・ツール系
- `~/.claude/design/vercel.md` — 汎用モダン

プロジェクト内に `./DESIGN.md` もしくは `./design/*.md` がある場合は、そちらをグローバルより優先すること。

### 日本語UI 用途別プロファイル（文脈に応じて Read）

- `~/.claude/design/jp-ui-contracts/templates/media/DESIGN.md` — 記事・長文読み物（line-height 1.75–2.0）
- `~/.claude/design/jp-ui-contracts/templates/saas/DESIGN.md` — 業務UI・管理画面（密度と安定性優先）
- `~/.claude/design/jp-ui-contracts/templates/docs/DESIGN.md` — 技術文書（本文とコード）
- `~/.claude/design/jp-ui-contracts/templates/dashboard/DESIGN.md` — BI・高密度情報（走査性優先）

### 日本語UI Recipe（必要時に Read／CSSで使う）

- `~/.claude/design/jp-ui-contracts/recipes/ja-text.css` — 本文の改行・overflow・font-kerning
- `~/.claude/design/jp-ui-contracts/recipes/mixed-script.css` — 和欧混植、`word-break: auto-phrase`
- `~/.claude/design/jp-ui-contracts/recipes/headings.css` — 見出し専用 line-height
- `~/.claude/design/jp-ui-contracts/recipes/forms.css` — フォーム窮屈さ対策

### 生成後レビュー（UI完成後に参照）

- `~/.claude/design/jp-ui-contracts/validators/review-checklist.md` — 目視レビュー観点
- `~/.claude/design/jp-ui-contracts/validators/lint-rules.md` — Reject/Warn 判定規則

---

### 適用順序

ブランド系 DESIGN.md を見た目の土台にしつつ、日本語要素には jp-ui-contracts のテンプレート値・recipe・validator を重ねる。矛盾したら**日本語側を優先**（本文の読みやすさが最優先）。

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
