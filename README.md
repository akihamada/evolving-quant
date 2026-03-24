# 🧬 Evolving Quant — ユーザーマニュアル

## 概要

あなたの投資ポートフォリオを**自動分析・自動提案**するシステムです。

- 毎朝 **7:00** / 毎晩 **22:00** に自動実行
- マーケットデータ取得 → 数学的最適化 → Claudeに分析依頼 → 結果を自動記録
- ダッシュボードで視覚的に確認可能

---

## 🚀 何をしてくれるのか

### ① 保有銘柄アクション提案
全銘柄に対して **BUY / HOLD / SELL** を判定し、理由を1行で提示。

### ② 積立設定レビュー
毎月¥250,000の積立配分が適切か分析。変更すべきファンド・割合を提案。

### ③ 新規銘柄提案
ポートフォリオに不足するセクターから3〜5銘柄を提案。

### ④ リスクシナリオ
Bull / Base / Bear の3シナリオを確率付きで提示。

---

## 📊 ダッシュボードの見方

http://localhost:8501 にアクセス

| タブ | 内容 |
|---|---|
| **📊 Overview** | ポートフォリオ配分・レジーム判定 |
| **📈 Performance** | 累積リターンチャート（1M/3M/6M/1Y/2Y 切替可） |
| **🔥 Risk** | 相関ヒートマップ・ボラティリティ |
| **🎯 AI Report Card** | AIの過去予測の精度評価 |
| **🧬 Meta-Prompt** | 次回分析用プロンプト確認 |
| **📝 Record** | 手動入力用フォーム |

---

## 🛒 株を購入したとき

この会話に以下のフォーマットで送るだけ:

### 米国株
```
/buy MRVL 3 $88.50
/buy NVDA 10 $175
```

### 日本株
```
/buy 1328 2 ¥18200
/buy 1489 5 ¥3100
```

### フォーマット
```
/buy ティッカー 株数 単価
```

- `$` がなくても米国株として処理
- `¥` or `JPY` or 4桁数字コードで日本株と判定
- 既存銘柄 → 加重平均で取得コスト自動更新
- 新規銘柄 → NISA枠に自動追加

---

## 📅 積立の自動反映

毎月の積立購入は**自動でポートフォリオに反映**されます。

| 注文日 | ファンド | 月額 |
|---|---|---|
| **7日** | FANG+ / SBI S&P500(つみたて枠) | ¥100,000 |
| **27日** | eMAXIS S&P500 / インド株 / ゴールド | ¥150,000 |

- `run_evolution.sh` の実行時に自動チェック
- 注文日なら概算口数を追加し、next_order を翌月に更新
- 確定値と異なる場合は手動で修正可能

---

## ⏰ 自動実行スケジュール

**月・水・金の朝 7:00** に自動実行（pmset で 6:55 にスリープ解除）

```
月/水/金 7:00 → launchd が自動起動
  ├─ Step 0  : launchd 自己登録
  ├─ Step 0.5: ログローテーション
  ├─ Step 1  : venv チェック & 自動構築
  ├─ Step 1.5: 積立自動反映チェック
  ├─ Step 2  : daily_evolution.py（データ取得 & 最適化）
  └─ Step 3  : auto_prompt_cycle.py
               ├─ プロンプト生成（ポートフォリオ全データ込み）
               ├─ この会話にプロンプト送信
               ├─ Claudeの分析 → /tmp/quant_response.json
               └─ ai_track_record.json に自動記録
```

### 前提条件
- **Antigravity が起動していること**
- アクセシビリティ権限が付与されていること
- pmset でスリープ解除設定済み（`sudo pmset repeat wakeorpoweron MWF 06:55:00`）

---

## 📁 ファイル構成

| ファイル | 説明 |
|---|---|
| `run_evolution.sh` | メインブートストラップ（launchd から実行） |
| `daily_evolution.py` | データ取得 & 数学的最適化エンジン |
| `auto_prompt_cycle.py` | プロンプト生成 → 送信 → 応答記録 |
| `update_holdings.py` | 積立購入の自動反映 |
| `purchase_handler.py` | 手動購入の処理 |
| `evolving_quant_dashboard.py` | Streamlit ダッシュボード |
| `data/portfolio_holdings.json` | ポートフォリオ保有データ |
| `data/purchase_log.json` | 購入履歴ログ（最大200件） |
| `ai_track_record.json` | AI予測の記録（最大90件） |

---

## 🔧 トラブルシューティング

### プロンプトが送信されない
1. Antigravity が起動しているか確認
2. アクセシビリティ権限: `システム環境設定 → プライバシー → アクセシビリティ`

### ダッシュボードが表示されない
```bash
HOME=/tmp /tmp/quant_venv/bin/streamlit run evolving_quant_dashboard.py --server.port 8501
```

### 手動でフル実行
```bash
# ターミナル.app から実行（Antigravityのサンドボックス外）
/bin/bash /Users/akihome/Documents/akimachome_antigravity/portfolio-bot/run_evolution.sh
```

### ログ確認
```bash
tail -20 /tmp/quant-evolution-stdout.log
tail -20 /tmp/quant-evolution-stderr.log
```

---

## 💾 データ管理

| 項目 | 上限 | 自動管理 |
|---|---|---|
| AI予測レコード | 90件（約30週分） | 超過分は古い順に自動削除 |
| 購入ログ | 200件 | 超過分は自動削除 |
| ログファイル | 1,000行 | 500行に自動トリム |
| 一時ファイル | — | 完了後に自動削除 |
