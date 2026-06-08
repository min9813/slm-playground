# 動画のダウンロードと文字起こし

X のポストに埋め込まれた動画を取得し、必要なら全文文字起こしする手順。

## ステップ1: 動画の直リンクを得る

[fetch-x-content.md](fetch-x-content.md) の fxtwitter/vxtwitter JSON API で
`media[].url`（`https://video.twimg.com/.../*.mp4`）を取得する。

```bash
VIDEO_URL=$(curl -s "https://api.fxtwitter.com/i/status/{id}" \
  | jq -r '.tweet.media.all[] | select(.type=="video") | .url')
echo "$VIDEO_URL"
```

複数解像度がある場合、`amplify_video/.../vid/avc1/{WxH}/...mp4` のパスに
解像度が入る。最高画質が欲しければ `formats[]` の HLS(.m3u8) を yt-dlp で
処理するか、最大解像度の MP4 を選ぶ。

## ステップ2: ダウンロード

### 方法1: 直リンクを curl（最速・依存なし）

```bash
curl -L -o video.mp4 "$VIDEO_URL"
file video.mp4   # => ISO Media, MP4 ... なら成功
```

### 方法2: yt-dlp（堅牢・字幕にも対応）

```bash
# 最高画質の動画本体
yt-dlp -f best -o "%(id)s.%(ext)s" "https://x.com/{user}/status/{id}"

# 字幕があれば字幕だけ（X は付かないことが多い）
yt-dlp --write-subs --sub-lang en --skip-download \
  -o "%(id)s.%(ext)s" "https://x.com/{user}/status/{id}"
```

yt-dlp が未導入なら `uvx yt-dlp ...` または `pipx install yt-dlp`。

## ステップ3: 文字起こし（X 動画に字幕が無い場合）

X の動画は字幕トラックが無いことが多い。音声を whisper で書き起こす。

```bash
# OpenAI whisper（CPUでも可・モデルは tiny/base/small/medium/large）
uvx --from openai-whisper whisper video.mp4 \
  --language en --model small --output_format txt --output_dir ./transcript

# 速い実装が良ければ faster-whisper
uvx --from faster-whisper ...   # API はパッケージ版に従う
```

- 音声抽出が別途必要なら `ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav`。
- 5分程度なら `small` で数十秒〜数分。精度優先なら `medium`/`large-v3`。

> 補足: Grok x-search の `--videos` はフレーム上の**字幕(キャプション)抽出**で
> あり、音声の逐語起こしではない。完全な逐語が要る場面ではこの whisper 手順を使う。

## 実例: Liquid AI 対談動画（このリポジトリ）

```bash
# 1. 直リンク特定（fxtwitter）
curl -s "https://api.fxtwitter.com/liquidai/status/2063658222121054643" \
  | jq -r '.tweet.media.all[0].url'
# => https://video.twimg.com/amplify_video/2063657994487787520/vid/avc1/1920x1080/X-W1mcgRc48XGZpA.mp4?tag=27

# 2. ダウンロード（33MB / 290.9s / 1920x1080）
curl -L -o liquidai_lfm_training_parallelism_interview.mp4 \
  "https://video.twimg.com/amplify_video/2063657994487787520/vid/avc1/1920x1080/X-W1mcgRc48XGZpA.mp4?tag=27"

# 3. 配置
mv liquidai_lfm_training_parallelism_interview.mp4 \
   ../../data/liquid-ai/      # = /home/min9813/projects/data/liquid-ai/
```

配置済みファイル:
`data/liquid-ai/liquidai_lfm_training_parallelism_interview.mp4`
（出典は同ディレクトリの `*.source.md`）

未実施: 音声の全文文字起こし（上記ステップ3を実行すれば
on-policy distillation / GPU 規模 等の言及有無を完全検証できる）。
