# 35. 変数のすぐ後ろに全角文字を置くと macOS の bash で名前が繋がる

**`$WANT_EPS）` と書くと、macOS の bash は全角括弧を変数名の一部として読む。**
値は正しく代入されているのに、読む側が別の名前を引くので
`unbound variable` になる（`set -u` のとき）。

```
build_initpose_datasets.sh: line 234: WANT_EPS?: unbound variable
```

**名前の後ろの文字化けがこれの印である。**Linux の bash では起きないので、
**書いた台では最後まで気づかない。**

## 何が困るか

**本当の原因が見えなくなる。**2026-09-18 に、この形のせいで

- 「データセットが出来ていないのでは」
- 「`hf download` が途中で終わったのでは」

と 2 手も回り道をした。**実際にはデータは揃っていて、表示の行だけが壊れていた。**

## 書き方

**日本語の文の中で変数を書くときは、必ず `${}` で括る。**

```bash
echo "本数 ${WANT_EPS}（期待）"     # よい
echo "本数 $WANT_EPS（期待）"       # macOS で壊れる
```

**半角が続くなら括らなくても動く**が、後から文を日本語に直したときに壊れるので、
**この木では常に括る。**

## 取り残しを探す

```bash
python3 -c "
import re, glob
pat = re.compile(r'\\\$([A-Za-z_][A-Za-z0-9_]*)(?=[^\x00-\x7f])')
print([f for f in glob.glob('solution/tools/**/*.sh', recursive=True)
       if pat.search(open(f, encoding='utf-8').read())] or '無し')"
```

2026-09-18 に 21 ファイル・37 箇所を直した。
