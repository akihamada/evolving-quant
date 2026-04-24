# Global User Instructions

## UI/デザイン実装ルール

UIを実装するときは必ず `~/.claude/design/` 配下の `DESIGN.md` を参照してスタイルを適用すること。

### HTML/UI のデフォルトベース（最優先）

**HTMLやUIを新規に作るとき、明示的に別ブランドの指定がない限り `~/.claude/design/cursor.md` をベースとして必ず読み込むこと。**

Cursor のデザイン特徴（warm off-white `#f2f1ed`、warm near-black `#26251e`、accent orange `#f54e00`、CursorGothic / jjannon serif / berkeleyMono の三書体、oklab ボーダー、full-pill 要素、8px ベース spacing）を土台として採用する。

### ブランド別 DESIGN.md（`~/.claude/design/`）

Cursor 以外を指定された場合、または用途が強く合致する場合は以下を選ぶ。

- `cursor.md` — **デフォルト**。warm minimalism + code-editor elegance
- `notion.md` — ドキュメント・LP・読み物系。Warm minimalism、serif見出し、soft surface
- `stripe.md` — 決済・フォーム系。Signature purple gradient、weight-300 の上品さ
- `linear.md` — ダッシュボード・ツール系。Ultra-minimal、precise、purple accent
- `vercel.md` — 汎用モダン。モノクロ精度、Geist フォント

プロジェクト内に `./DESIGN.md` もしくは `./design/*.md` がある場合は、そちらをグローバルより優先すること。

### 日本語UI（`~/.claude/design/jp-ui-contracts/`）

日本語UIを実装するときは、以下を併用する。欧文前提のDESIGN.mdだけでは和文フォールバック、行間、改行戦略、和欧混植が崩れるため。

- `templates/base/DESIGN.md` — 最小共通契約（最初に選ぶ骨格）
- `templates/media/DESIGN.md` — 記事・長文読み物（line-height 1.75–2.0）
- `templates/saas/DESIGN.md` — 業務UI・管理画面（密度と安定性優先）
- `templates/docs/DESIGN.md` — 技術文書（本文とコードの共存）
- `templates/dashboard/DESIGN.md` — BI・高密度情報（走査性優先）
- `recipes/ja-text.css` — 日本語本文の改行・overflow・font-kerning
- `recipes/mixed-script.css` — 和欧混植、`word-break: auto-phrase` の見出し扱い
- `recipes/headings.css` — 見出し専用 line-height
- `recipes/forms.css` — フォーム窮屈さ対策
- `validators/review-checklist.md` — 生成後の目視レビュー観点
- `validators/lint-rules.md` — Reject/Warn の判定規則

適用順序: ブランド系 DESIGN.md を見た目の土台にしつつ、日本語要素には jp-ui-contracts のテンプレート値・recipe・validator を重ねる。矛盾したら**日本語側を優先**（本文の読みやすさが最優先）。

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
