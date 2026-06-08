# x.com コンテンツ取得ガイド

公式有料 API（Basic $200/月〜）を使わずに、X(旧 Twitter) のポストから
**本文テキスト・メディア・動画・字幕**を取得する実用手順をまとめる。

2026-06 時点で実地検証済み。X の仕様変更で非公式手法は突然死し得るため、
**複数手段を併用するのが前提**。

## ドキュメント一覧

| ファイル | 内容 |
| --- | --- |
| [fetch-x-content.md](fetch-x-content.md) | ポスト本文・メタ・メディアURL の取得（fxtwitter / vxtwitter JSON API ほか） |
| [download-video.md](download-video.md) | 動画ファイルのダウンロードと文字起こし（直リンク / yt-dlp / whisper） |

## 30秒クイックリファレンス

```bash
# 本文・メディアURL を JSON で（x.com を api.fxtwitter.com に置換するだけ）
curl "https://api.fxtwitter.com/{user}/status/{id}"
curl "https://api.vxtwitter.com/{user}/status/{id}"

# 動画を落とす（上記JSONの media[].url 直リンクを curl、または yt-dlp）
yt-dlp -f best "https://x.com/{user}/status/{id}"

# 動画を文字起こし
uvx --from openai-whisper whisper video.mp4 --language en --model small
```

## 手法の生死マップ（2026-06）

| 手法 | 状態 | 主用途 |
| --- | --- | --- |
| api.vxtwitter.com | ✅ 生存・最有力 | 本文・メディアURL(JSON) |
| api.fxtwitter.com | ✅ 生存・安定 | 本文・メディア(JSON, 構造リッチ) |
| yt-dlp | ✅ 生存・最有力 | 動画/音声/字幕DL |
| Grok x-search (`/x-search` スキル) | ✅ 生存 | 検索・要約・動画フレーム字幕 |
| gallery-dl | △ 要 cookie | 画像/動画一括DL |
| Nitter (xcancel.com 等) | △ 一部生存 | 閲覧用フォーク |
| 純正 Nitter / snscrape | ❌ 死滅 | — |
| cdn.syndication.twimg.com | ❌ ほぼ死滅 | (旧) 本文取得 |
| publish.twitter.com/oembed | △ 限定生存 | embed HTML のみ |

## 実例（このリポジトリでの利用）

[Liquid AI の学習インフラ対談動画](https://x.com/liquidai/status/2063658222121054643)
を取得し、`data/liquid-ai/liquidai_lfm_training_parallelism_interview.mp4`
に配置済み。手順は両ドキュメントの「実例」節を参照。

## 注意

- いずれも非公式・無保証。利用は自己責任で、X の利用規約・各サービスの規約を確認のこと。
- 取得したデータには出典（ポストURL・取得日・直リンク）をサイドカーで残す運用にする
  （例: `*.source.md`）。
