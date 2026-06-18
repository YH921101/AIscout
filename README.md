# 全市場AIスカウトBOT

TDnetの当日開示を1〜5分おきに取得し、タイトル一次フィルタに通った開示だけPDF本文を抽出してOpenAI APIで採点します。スコア95点以上だけDiscord Webhookへ通知し、処理済みIDはJSONで保存して重複通知を防ぎます。

## エンドポイント

- `GET /health`: 稼働確認
- `GET /run?token=RUN_TOKEN`: cron-job.orgなどから呼び出す実行口

`RUN_TOKEN` を設定すると、`token` クエリまたは `X-Run-Token` ヘッダーが一致した場合だけ実行されます。

## Render設定

1. このディレクトリをRenderのWeb Serviceとしてデプロイします。
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
4. Environmentに `.env.example` の値を登録します。
5. cron-job.orgで `https://YOUR-RENDER-APP.onrender.com/run?token=RUN_TOKEN` を1〜5分おきに叩きます。

## 必須環境変数

- `OPENAI_API_KEY`: OpenAI APIキー
- `DISCORD_WEBHOOK_URL`: Discord Webhook URL
- `RUN_TOKEN`: cron実行用の秘密トークン

## 主な調整項目

- `OPENAI_MODEL`: 既定は `gpt-5.5`
- `OPENAI_REASONING_EFFORT`: 既定は `low`
- `NOTIFY_THRESHOLD`: Discord通知スコア。既定は `95`
- `LOG_THRESHOLD`: JSONLログ保存スコア。既定は `85`
- `MAX_AI_CANDIDATES_PER_RUN`: 1回の実行でAI判定する最大件数。既定は `12`
- `TDNET_LOOKBACK_DAYS`: 取得対象日数。既定は当日のみ `1`

## ローカル確認

```bash
pip install -r requirements.txt
copy .env.example .env
python app.py
```

OpenAIやDiscordを使わず流れだけ確認したい場合は `.env` で `DRY_RUN=true` にしてください。

## ログと重複排除

- `data/processed.json`: 処理済み開示ID
- `data/scout_log.jsonl`: 85点以上の候補、通知済み、Discord送信失敗などのログ

AI判定失敗やDiscord通知失敗は処理済みにしないため、次回cronでリトライします。
