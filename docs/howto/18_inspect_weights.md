# 18. 重みの素性をコマンドで引く

**台帳（[../reference/19_weights_catalog.md](../reference/19_weights_catalog.md)）に書いてある
sha256・公開/非公開・学習の種類・推論の設定は、すべてここのコマンドで引き直せる。**
**台帳の値が古いと疑ったときは、台帳を読まずにこれを打つ。**
**打つ場所はこのリポジトリの直下**（`.venv` がある所）。

**2026-09-12 に [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) の
「引き直すコマンド」の節を、そのままこの文書へ移した。**

## 0. 目次

- [1. HF にある重みと sha256 と公開・非公開を並べる](#1-hf-にある重みと-sha256-と公開非公開を並べる)
- [2. 提出に使った推論の設定を読む](#2-提出に使った推論の設定を読む)
- [3. 行動側だけの学習か、全パラメータかを見る](#3-行動側だけの学習か全パラメータかを見る)
- [4. HF の使用量を数える（枠は 100 GB）](#4-hf-の使用量を数える枠は-100-gb)
- [5. 落とさずにチェックポイントが別物かを確かめる](#5-落とさずにチェックポイントが別物かを確かめる)

## 1. HF にある重みと sha256 と公開・非公開を並べる

**どの重みが HF に残っていて、どれが公開かを 1 画面で見るためのコマンドである。**
**非公開だけが 100 GB の枠を食う**ので、枠を空ける相談はここから始める。

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author='siruku6'):
    i = api.model_info(m.id, files_metadata=True)
    for s in i.siblings:
        if s.rfilename.endswith('model.safetensors'):
            print(f\"{'非公開' if i.private else '公開  '} {s.lfs.sha256[:12]} {m.id}/{s.rfilename}\")"
```

**1 行につき「公開/非公開・sha256 の頭 12 桁・リポジトリ名/ファイル名」が出る。**

## 2. 提出に使った推論の設定を読む

**重みの隣に置いた `INFERENCE.md` が、その重みで出したときの n・infer・平滑化・samples の
一次資料である。**落とさずに中身だけ読める。

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(open(hf_hub_download('siruku6/pi05_full_runpod', 'INFERENCE.md')).read())"
```

## 3. 行動側だけの学習か、全パラメータかを見る

**名前からは分からない。**`pi05_full_local` の「full」は実行の名前であって、
全パラメータを動かしたという意味ではない。**`train_config.json` を見るのが唯一の決め手である。**

```bash
.venv/bin/python -c "
import json
d = json.load(open('work/artifacts/pi05_rp3000/train_config.json'))['policy']
print('行動側だけ' if d['train_expert_only'] else '全パラメータ')"
```

## 4. HF の使用量を数える（枠は 100 GB）

**次のチェックポイントを上げられるかを、上げる前に決めるための数である。**
**重み 1 点が 8.73 GiB なので、残りから「あと何点置けるか」が出る。**

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
t = 0
for m in api.list_models(author='siruku6'):
    i = api.model_info(m.id, files_metadata=True)
    s = sum(f.size or 0 for f in i.siblings); t += s
    print(f'{s/2**30:8.2f} GiB  {m.id}')
print(f'{t/2**30:8.2f} GiB  合計')"
```

## 5. 落とさずにチェックポイントが別物かを確かめる

**9.4 GB を落とさずに「手元のものと中身が違うか」を判定できる。**
HF の LFS が返すダイジェストが、落としたファイルに `sha256sum` を打った値と同じだからである。

**9/2 に、この一致を初めて端から端まで確かめた。**`ladder2500/001500` のダイジェストは
HF 側が `87d89a05…` を返し、Mac に落としてから打った `sha256sum` も
`87d89a055f166377bf201e6f72ffeef206cc6b2d28fa7511daed2f3375308436` だった。
**だからこの値をそのまま信じてよい。**

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
p = [f'ladder2500/{s}/model.safetensors' for s in ('001000','001500','002000','002500')]
for i in HfApi().get_paths_info('siruku6/pi05_trial1500', p):
    print(i.lfs.sha256[:12], i.path)"
```
