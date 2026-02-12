# code260212

## OpenAI API で動画フレームを自動アノテーション

`/Users/k.omote/Documents/New project/code260212/src/annotate_videos_with_openai.py` は以下を行います。

- 動画から一定間隔でフレーム抽出
- OpenAI API で7クラス判定（マルチラベル可）
- 複数ラベルなら同一フレームを複数クラスフォルダへコピー
- 対象外/判別不能なら `unknown` へコピー
- 実行ログをCSV保存

対象クラス:

- アナグマ
- アライグマ
- ハクビシン
- タヌキ
- ネコ
- ノウサギ
- テン

### Colab での実行例

```bash
!pip -q install openai opencv-python-headless
```

`OPENAI_API_KEY` は環境変数または Colab Secrets に登録済みの前提で実行してください（ノートブックにキー文字列を直接書かない）。

```bash
!python "/content/drive/MyDrive/code260212/src/annotate_videos_with_openai.py" \
  --input-dir "/content/drive/MyDrive/RG/input_images" \
  --output-root "/content/drive/MyDrive/RG/ai_annotated_frames" \
  --metadata-csv "/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv" \
  --model "gpt-5.2" \
  --sample-every-n-frames 45 \
  --max-frames-per-video 180
```

### 指定動画リストだけ実行する例

`/Users/k.omote/Documents/New project/code260212/metadata/rg_video_list.txt` に動画一覧を保存済みです。

```bash
!python "/content/drive/MyDrive/code260212/src/annotate_videos_with_openai.py" \
  --video-list-file "/content/drive/MyDrive/code260212/metadata/rg_video_list.txt" \
  --output-root "/content/drive/MyDrive/RG/ai_annotated_frames" \
  --metadata-csv "/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv" \
  --model "gpt-5.2" \
  --sample-every-n-frames 45 \
  --max-frames-per-video 180
```

APIエラーで止めたくない場合だけ `--continue-on-api-error` を付けてください（その場合は失敗フレームが `unknown` になります）。

出力先:

- 元フレーム: `/content/drive/MyDrive/RG/ai_annotated_frames/_frames/...`
- クラス別: `/content/drive/MyDrive/RG/ai_annotated_frames/<class_name>/...`
- ログCSV: `/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv`
