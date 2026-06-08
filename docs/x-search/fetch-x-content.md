# ポスト本文・メディアの取得

X のポスト URL（`https://x.com/{user}/status/{id}`）から、本文テキスト・
作者情報・メディア URL を、API キーなしで取得する手法。

## 手法A: fxtwitter / vxtwitter JSON API（推奨）

X の embed 修正サービス。URL のホスト名を置き換えるだけで JSON が返る。
**認証・cookie 不要**で、2026-06 時点で最も安定。

### fxtwitter（構造リッチ）

```bash
curl "https://api.fxtwitter.com/{user}/status/{id}"
# user 不明でも可:
curl "https://api.fxtwitter.com/i/status/{id}"
```

返る JSON の主なフィールド:

| パス | 内容 |
| --- | --- |
| `tweet.text` / `tweet.raw_text.text` | 本文（全文・改行込み） |
| `tweet.author.screen_name` / `.name` | 作者ハンドル / 表示名 |
| `tweet.created_at` / `.created_timestamp` | 投稿日時 |
| `tweet.views` / `.likes` / `.retweets` | エンゲージメント |
| `tweet.media.all[]` | メディア配列 |
| `tweet.media.all[].url` | **動画/画像の直リンク** |
| `tweet.media.all[].duration` | 動画長(秒) |
| `tweet.media.all[].formats[]` | HLS(.m3u8) など別形式 |

### vxtwitter（シンプル）

```bash
curl "https://api.vxtwitter.com/{user}/status/{id}"
```

| パス | 内容 |
| --- | --- |
| `text` | 本文 |
| `user_screen_name` / `user_name` | 作者 |
| `date` / `date_epoch` | 投稿日時 |
| `mediaURLs[]` | メディア直リンク（配列） |
| `media_extended[]` | メディア詳細（duration_millis, size 等） |

### jq で要点だけ抜く例

```bash
# 本文だけ
curl -s "https://api.fxtwitter.com/i/status/{id}" | jq -r '.tweet.text'

# 動画の直リンク MP4 だけ
curl -s "https://api.fxtwitter.com/i/status/{id}" \
  | jq -r '.tweet.media.all[] | select(.type=="video") | .url'

# vxtwitter でメディアURL一覧
curl -s "https://api.vxtwitter.com/i/status/{id}" | jq -r '.mediaURLs[]'
```

## 手法B: Grok x-search スキル（検索・要約・動画字幕）

このリポジトリ環境では `/x-search` スキル（`hermes-agent` の `x_search_tool`）が使える。
**生の検索 API ではなく Grok による要約・分析**を返す点に注意。
本文をそのまま欲しい場合はクエリで「本文をそのまま列挙して」と明示する。

```bash
# 基本
uvx --from hermes-agent python \
  "<skill_dir>/x_search.py" "クエリ"

# アカウント・期間で絞る / 全JSON / 動画フレーム字幕
uvx --from hermes-agent python "<skill_dir>/x_search.py" \
  --allow liquidai --from 2026-06-01 --to 2026-06-08 --raw --videos "クエリ"
```

- 用途: 横断検索・トレンド把握・特定アカウントの発言要約。
- 動画は **音声逐語ではなくフレーム上の字幕(キャプション)抽出**のため断片的。
  完全な文字起こしが要る場合は [download-video.md](download-video.md) の whisper を使う。

## 手法C: その他（フォールバック）

| 手法 | コマンド/URL | 状態 |
| --- | --- | --- |
| Nitter フォーク閲覧 | `https://xcancel.com/{user}/status/{id}` | △ 一部生存 |
| oembed (embed HTML) | `https://publish.twitter.com/oembed?url={post_url}` | △ 限定 |
| syndication | `https://cdn.syndication.twimg.com/tweet-result?id={id}` | ❌ ほぼ死滅 |
| snscrape / 純正 Nitter | — | ❌ 死滅 |

## 実例: Liquid AI 対談ポスト

```bash
curl -s "https://api.fxtwitter.com/liquidai/status/2063658222121054643" \
  | jq '{text: .tweet.text, video: (.tweet.media.all[0].url), dur: .tweet.media.all[0].duration}'
```

得られた本文（学習レシピの一次情報として有用）:

> Training LFMs at scale means solving parallelism across every layer of the
> architecture ... Data, tensor, pipeline, expert, and context parallelism, and
> how they make context parallelism work across hybrid architectures with both
> attention and convolution operators.

→ 「5D 並列」「ハイブリッドのコンテキスト並列化」が**公式ポスト本文で確定**。
