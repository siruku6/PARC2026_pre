#!/usr/bin/env bash
# Claude の設定一式を別のマシンへ持ち出す。
#
# GPU マシンで作業を続けるとき、skill と CLAUDE.md が無いと Claude は
# 「日本語は日常語で」「パスは英語で」「一次情報を出典に」を知らない状態から
# 始まる。持ち出すのは 30 KB 程度なので、毎回作り直せばよい。
#
#   bash solution/tools/export_claude_setup.sh
#   scp work/claude-setup.tgz <GPU マシン>:~/
#
# 受け取り側:
#   mkdir -p ~/.claude && tar xzf ~/claude-setup.tgz -C ~/.claude
#
# **認証情報は入れていない。** GPU マシンでは claude を改めて認証すること。
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${REPOSITORY_ROOT}/work/claude-setup.tgz"

# 持ち出すもの。**このコンペで実際に使ったものだけ**を挙げる。
# 使っていない skill を混ぜると、受け取り側で何が効いているのか分からなくなる。
ITEMS=(
    CLAUDE.md                        # 会話と作業の基本方針。**これが最重要**
    plain-japanese.md                # 使わない語と言い換えの表。**CLAUDE.md 1.1 が指す。
                                     # これが無いと 1.1 は「専門語か否か」の判断に戻る**
    skills/ml-competition-workflow   # 採点が見えない相手にスコアを詰める手順
    skills/torch-ecosystem-deps      # torch と周辺パッケージの版・CUDA 系統
    skills/notebook-runbook          # Colab の手順書と、結果を失わない作り
    skills/documentation-writing     # 読者目線・図表・用語定義
    skills/pdf-reading               # 参加者ガイド PDF を読むのに実際に使った
    skills/python-script-style       # 命名・関数分割・型注釈
    skills/multi-agent-survey        # 手法を広く調べるとき（未使用だが 11 番で使う）
)

mkdir -p "$(dirname "$OUTPUT")"

for item in "${ITEMS[@]}"; do
    if [[ ! -e "${HOME}/.claude/${item}" ]]; then
        echo "エラー: ~/.claude/${item} が無い" >&2
        exit 1
    fi
done

# macOS 由来のメタデータを載せない。
#
# macOS の tar（bsdtar）は既定でファイルの拡張属性を書き込む。Finder や
# Gatekeeper が付ける `com.apple.provenance` などである。これを GNU tar
# （Linux の既定）で展開すると、1 ファイルにつき 1 行こう警告が出る。
#
#   tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'
#
# **展開自体は成功している**（GNU tar は知らないヘッダを読み飛ばすだけ）。
# ただし失敗と見分けが付かないので、送る側で載せないようにする。
#   --no-xattrs       … 拡張属性を書かない
#   --no-mac-metadata … AppleDouble（`._` で始まる分身ファイル）を作らない
#
# GNU tar にこの 2 つは無いので、bsdtar のときだけ渡す。
TAR_OPTIONS=()
if tar --version 2>/dev/null | grep -q bsdtar; then
    TAR_OPTIONS=(--no-xattrs --no-mac-metadata)
fi

tar czf "$OUTPUT" "${TAR_OPTIONS[@]}" -C "${HOME}/.claude" "${ITEMS[@]}"

echo "作った: ${OUTPUT}  ($(du -h "$OUTPUT" | cut -f1))"
echo
echo "中身:"
tar tzf "$OUTPUT" | grep -E 'SKILL\.md|CLAUDE\.md' | sed 's/^/  /'
echo
echo "受け取り側:"
echo "  mkdir -p ~/.claude && tar xzf claude-setup.tgz -C ~/.claude"
