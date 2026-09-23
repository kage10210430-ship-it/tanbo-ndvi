# 田んぼNDVIマップ — 事前計算＋静的配信

県内の田んぼ（筆ポリゴン）ごとの NDVI を **週1回まとめて計算** し、結果を **静的ファイル** として公開する仕組みです。
利用者（農家・普及員）はブラウザで地図を開き、田んぼをタップするだけ。利用時に計算は発生しないため、
サーバー費用はほぼゼロで、利用者が増えても遅くなりません。

```
 筆ポリゴン(農水省)  Sentinel-2(Copernicus)
        │                  │
        ▼                  ▼
  build_cells.py     update_ndvi.py  ← Google Earth Engine で区画平均を計算
  (区画をグリッドに分割)  (未計算の観測日だけ追記)
        │                  │
        └──────┬───────────┘
               ▼
       site/data/           ← GitHub Actions が週1回更新し、GitHub Pages で配信
        ├ index.json          （セル一覧・更新日）
        └ cells/<id>/
           ├ parcels.geojson  （区画の形・有効画素の期待数）
           ├ inner.geojson    （計算用: 畦畔を避けて5m内側に縮めた形）
           └ ndvi.json        （観測日 × 区画の NDVI と画素数）
               │
               ▼
       site/index.html      ← 表示範囲のセルだけ読み込む地図ビューア
```

## できること（ビューア）
- 航空写真の上に区画。選んだ日付の NDVI で色分け（地域全体の生育の見え方）
- 区画をタップで最大6枚を比較。直近12か月／前年との重ね／移植後日数
- 近隣の田の平均（半径可変）との比較、要約表
- 雲・かすみで一斉に落ちた観測日を自動除外（詳細設定で戻せる）
- 「この場所のリンクをコピー」で、位置と選択した田んぼをそのまま共有

## セットアップ（1回だけ）

専門用語なしの手順書は `SETUP_やさしい版.md` を参照。

### 1. Google Earth Engine（計算エンジン）
1. Google Cloud プロジェクトを Earth Engine に登録（非営利・研究用途。試験場の研究として登録）
2. Cloud コンソール → IAM → **サービスアカウント** を作成し、JSON 鍵をダウンロード
3. そのサービスアカウントに `Earth Engine Resource Viewer` と `Service Usage Consumer` のロールを付与
4. Cloud コンソールで `Earth Engine API` を有効化（まだなら）

### 2. GitHub リポジトリ
1. このフォルダをそのままリポジトリにする（**Public** にする。無料プランでは Public リポジトリだけが Pages で公開でき、Actions の実行時間も無制限）
2. Settings → Secrets and variables → Actions に3つ登録
   - `EE_SERVICE_ACCOUNT` … サービスアカウントのメールアドレス
   - `EE_PRIVATE_KEY` … JSON 鍵ファイルの中身をそのまま貼る
   - `EE_PROJECT` … Cloud プロジェクトID
3. Settings → Pages → Source を `gh-pages` ブランチにする（初回の Actions 実行後にブランチができます）

### 3. 区画データ
`config.yaml` の `parcel_sources` に筆ポリゴンの入手先を書きます。
- 農林水産省「筆ポリゴンデータ」ページの **福井県 FlatGeobuf（分割ファイル）** の URL を並べる、または
- 筆ポリゴン公開サイト（open.fude.maff.go.jp）から市町ごとに GeoJSON を落として `parcels/` に置く
`bbox` で範囲を絞れます。まずは1つの市町か地区で試すのが安全です。

### 4. 初回実行
Actions → `weekly-ndvi` → **Run workflow**。初回は過去24か月分をさかのぼるので数時間かかります。
1回で終わらなくても、途中まで保存して次回に続きから計算します（`max_minutes` を超えると自動で切り上げ）。
終わったら `https://<ユーザー名>.github.io/<リポジトリ名>/` で開けます。

## 運用
- 毎週月曜 6:00（JST）に自動更新。手動実行も可
- 筆ポリゴンが年次更新されたら、`config.yaml` を直して **Run workflow の `rebuild_cells` にチェック**
- 1セル ≒ 2.2km×1.85km。県全体で数百セル。週次更新は数分〜数十分
- ログは Actions の実行画面で見られます。1セルの失敗は飛ばして続行します

## 設定の要点（config.yaml）
| 項目 | 意味 |
|---|---|
| `only_paddy` | 田（land_type=100）だけを対象にする |
| `edge_m` | 畦畔を避けて内側に縮める距離。10m画素なら5mが目安 |
| `history_months` | 初回にさかのぼる月数（前年重ね表示には24） |
| `max_scene_cloud` | シーン雲量がこれ以上の画像は使わない |
| `cell_deg_lon/lat` | 配信ファイルの分割単位 |

## ローカルでの動作確認（GEE 不要）
```bash
pip install -r requirements.txt
python pipeline/build_cells.py
python pipeline/update_ndvi.py --backend fake --fake-csv sample/genshu_center_ndvi.csv
cd site && python -m http.server 8000   # → http://localhost:8000/
```
`sample/` には原種センター17区画の GeoJSON と NDVI（2025/1〜2026/9）が入っています。

## ライセンス・出典
- 衛星: Contains modified Copernicus Sentinel data（無償・出典表記のみ）
- 区画: 農林水産省「筆ポリゴン」（利用規約に従い出典表記）
- 地図: 国土地理院タイル
- 計算: Google Earth Engine。**非営利（研究・教育）用途は無償、行政の実務運用は有償ライセンス** の対象になり得ます。
  試験場の研究として試作→本運用時に Earth Engine 商用ライセンスか、計算部分を Copernicus openEO に置き換える
  （`pipeline/update_ndvi.py` の `GEEBackend` を差し替えるだけ）のどちらかを判断してください。

## 限界
- 10m 画素なので、細長い区画や小さな区画は隣の影響を受けます（有効画素率で絞り込み可）
- 雲・かすみは完全には除けません。自動除外は「多くの区画が一斉に下がって次で戻る日」だけです
- NDVI は 0.8〜0.9 で頭打ちになるため、繁茂した圃場間の差は実際より小さく見えます
