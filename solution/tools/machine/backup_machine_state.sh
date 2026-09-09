#!/usr/bin/env bash
# このマシンを作り替える前に、**失うと作り直せないものだけ**を 1 か所に集める。
#
# 集めるのは 2 種類ある。**どちらも小さい。重みもデータも集めない**（HF から
# 取り直せる。判定は docs/howto/10_rebuild_machine.md の 5 節）。
#
#   manifest … 復旧時に「元はこうだった」を照合するための実測値。
#              pip freeze・apt の手動導入一覧・docker inspect・nvidia-smi など
#   files    … git にも HF にも無い一次資料。訓練ログ・評価の数字・NOW.md など
#
# 秘密（SSH 秘密鍵・HF トークン・Claude の資格情報）は **わざと入れない**。
# 置き場所と作り直し方を SECRETS_TODO.md に書き出すだけにする。理由は
# docs/howto/10_rebuild_machine.md の 2 節。
#
#   bash solution/tools/machine/backup_machine_state.sh
#   bash solution/tools/machine/backup_machine_state.sh --out /tmp/backup
#   bash solution/tools/machine/backup_machine_state.sh --with-transcripts
#
# 出来るもの（既定では 1 つ目だけ）
#   <out>/machine_state_<日付>.tar.gz          約 70 MB
#   <out>/claude_transcripts_<日付>.tar.gz     約 190 MB（--with-transcripts のとき）
#
# 引数
#   --out DIR            置き場。既定 $HOME/machine_backup
#   --with-transcripts   ~/.claude/projects（過去セッションの記録）も別の tar にする
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$HOME/machine_backup"
WITH_TRANSCRIPTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        --with-transcripts) WITH_TRANSCRIPTS=1; shift ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "不明な引数: $1" >&2; exit 2 ;;
    esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/machine_state_$STAMP"
mkdir -p "$ROOT/manifest" "$ROOT/files"

log() { printf '[backup] %s\n' "$*"; }

# ---------------------------------------------------------------- manifest
# **ここは全部その場で測る。** 手で書き写した値は古くなるので置かない。
log "1/4 manifest —— 元の姿を実測して書き出す"

{
    echo "# 採取 $(date -Is)"
    echo
    echo "## OS / カーネル / CPU / メモリ"
    . /etc/os-release && echo "$PRETTY_NAME"
    uname -r
    echo "cores: $(nproc)"
    free -g | sed -n '1,2p'
    echo
    echo "## ディスク"
    df -h /
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || echo "nvidia-smi なし"
    echo
    echo "## vGPU ライセンス（期限は 1 日ごとに更新される。切れているかを見る欄ではない）"
    nvidia-smi -q 2>/dev/null | grep -E 'vGPU Software|Product Name|License Status' || true
    echo
    echo "## nvidia / cuda の dpkg"
    dpkg -l 2>/dev/null | grep -iE 'nvidia|cuda' | awk '{print $1, $2, $3}'
    echo
    echo "## docker"
    docker version --format 'server {{.Server.Version}}' 2>&1 || true
    docker info --format 'storage={{.Driver}}' 2>&1 || true
} > "$ROOT/manifest/host.md"

# apt で「自分で入れた」ことになっているものだけ。依存で入ったものは復元時に付いてくる。
apt-mark showmanual > "$ROOT/manifest/apt-manual.txt" 2>/dev/null || true

# apt のリポジトリ定義。nvidia-container-toolkit と Claude Desktop はここが要る。
{
    grep -hE '^deb ' /etc/apt/sources.list 2>/dev/null || true
    grep -rhE '^deb ' /etc/apt/sources.list.d/ 2>/dev/null || true
} > "$ROOT/manifest/apt-sources.txt"

# ホストの .venv は HF への上げ下げ専用の小さいもの。setup.sh が作る venv とは別物。
if [ -x "$REPOSITORY_ROOT/.venv/bin/pip" ]; then
    "$REPOSITORY_ROOT/.venv/bin/python" -V > "$ROOT/manifest/host-venv.txt" 2>&1
    "$REPOSITORY_ROOT/.venv/bin/pip" freeze >> "$ROOT/manifest/host-venv.txt" 2>&1
fi

# **常駐コンテナの起動条件はここにしか残っていない。** リポジトリのどの手順書にも
# `docker run -d --name quant_train ...` の完全な形は書かれていない（`...` で省略
# されている）。inspect の実測から起動コマンドを組み立てる。
docker images > "$ROOT/manifest/docker-images.txt" 2>&1 || true
docker ps -a > "$ROOT/manifest/docker-ps.txt" 2>&1 || true
for container in $(docker ps -aq 2>/dev/null); do
    name="$(docker inspect "$container" --format '{{.Name}}' | tr -d /)"
    docker inspect "$container" > "$ROOT/manifest/docker-inspect-$name.json" 2>&1 || true
    docker inspect "$container" --format '
docker run -d --name {{index (split .Name "/") 1}} \
  {{if .HostConfig.DeviceRequests}}--gpus all {{end}}\
  {{if .Config.User}}--user {{.Config.User}} {{end}}\
  --network {{.HostConfig.NetworkMode}} \
  {{range .HostConfig.Binds}}-v {{.}} \
  {{end}}{{.Config.Image}} {{range .Config.Cmd}}{{.}} {{end}}
' >> "$ROOT/manifest/docker-run-recovered.sh" 2>&1 || true
done

# HF キャッシュに何が居たか。**中身は入れない。ここから取り直すための一覧である。**
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    du -sh "$HOME/.cache/huggingface"/* 2>/dev/null > "$ROOT/manifest/hf-cache-sizes.txt" || true
    ls "$HOME/.cache/huggingface/hub" > "$ROOT/manifest/hf-cache-repos.txt" 2>/dev/null || true
fi

crontab -l > "$ROOT/manifest/crontab.txt" 2>/dev/null || echo "(cron の登録なし)" > "$ROOT/manifest/crontab.txt"

# git の取り残し。**復旧後ではなく、作り替える前にここを空にしておく。**
{
    for repository in "$REPOSITORY_ROOT" "$HOME/src"/*; do
        [ -d "$repository/.git" ] || continue
        echo "## $repository"
        git -C "$repository" status --short 2>&1 | sed 's/^/   /'
        git -C "$repository" stash list 2>&1 | sed 's/^/   stash: /'
        for branch in $(git -C "$repository" for-each-ref --format='%(refname:short)' refs/heads); do
            remote_count="$(git -C "$repository" branch -r --contains "$branch" 2>/dev/null | wc -l)"
            [ "$remote_count" -eq 0 ] && echo "   ⚠ $branch はどのリモート ref にも無い"
        done
        echo
    done
} > "$ROOT/manifest/git-unsaved.md"

# ---------------------------------------------------------------- files
log "2/4 files —— git にも HF にも無い一次資料を集める"

work="$REPOSITORY_ROOT/work"
mkdir -p "$ROOT/files/work"

# 6 時間走った学習のログ。走らせ直す以外に作る手が無い。
mkdir -p "$ROOT/files/work/train"
find "$work/train" -maxdepth 1 -name '*.log' -exec cp {} "$ROOT/files/work/train/" \; 2>/dev/null || true

# 評価 57 回分。**mp4 を外すと 1.3 GB が 43 MB になる。**数字は log/json/csv 側にある。
if [ -d "$work/gpu_eval" ]; then
    (cd "$work" && tar cf - --exclude='*.mp4' gpu_eval) | (cd "$ROOT/files/work" && tar xf -)
fi

for directory in logs quant_sweep sha256 episode_lists patch build_430 admin tier1_camera; do
    [ -d "$work/$directory" ] && cp -a "$work/$directory" "$ROOT/files/work/" || true
done

# src/ は git リポジトリではない。ここのファイルはどこにも控えが無い。
mkdir -p "$ROOT/files/src"
src="$(dirname "$REPOSITORY_ROOT")"
[ -f "$src/NOW.md" ] && cp "$src/NOW.md" "$ROOT/files/src/"
[ -d "$src/.claude" ] && cp -a "$src/.claude" "$ROOT/files/src/dot-claude"
[ -d "$src/tmp" ] && cp -a "$src/tmp" "$ROOT/files/src/tmp"

# 全プロジェクト共通の Claude 設定。言い回し集と memory はここにしか無い。
mkdir -p "$ROOT/files/home-claude"
for item in CLAUDE.md plain-japanese.md settings.json skills plans; do
    [ -e "$HOME/.claude/$item" ] && cp -a "$HOME/.claude/$item" "$ROOT/files/home-claude/" || true
done
# memory/ は 1 ファイル 1 事実の作りなので、プロジェクト名ごとに分けて平らに置く。
for memory_dir in "$HOME/.claude/projects"/*/memory; do
    [ -d "$memory_dir" ] || continue
    project="$(basename "$(dirname "$memory_dir")")"
    mkdir -p "$ROOT/files/home-claude/memory/$project"
    cp -a "$memory_dir"/. "$ROOT/files/home-claude/memory/$project/"
done

mkdir -p "$ROOT/files/home"
for item in .gitconfig .bash_history .selected_editor libero_plus_episodes.parquet; do
    [ -e "$HOME/$item" ] && cp -a "$HOME/$item" "$ROOT/files/home/" || true
done
[ -f "$HOME/.ssh/known_hosts" ] && cp "$HOME/.ssh/known_hosts" "$ROOT/files/home/ssh_known_hosts"

# ---------------------------------------------------------------- secrets
log "3/4 secrets —— 中身は入れず、置き場と作り直し方だけ書く"

cat > "$ROOT/SECRETS_TODO.md" <<'SECRETS'
# 秘密は tar に入っていない。新しいマシンで手当てする 4 件

**わざと外した。** この tar は HF や Mac を経由して運ぶので、秘密鍵を入れると
運んだ先すべてに鍵が残る。どれも数分で作り直せる。

| 何 | 元の置き場 | 新しいマシンでどうするか |
|---|---|---|
| SSH 秘密鍵 | `~/.ssh/id_ed25519`・`~/.ssh/id_rsa` | `ssh-keygen -t ed25519` で作り直し、公開鍵を GitHub の Settings → SSH keys に登録する。古い鍵は GitHub 側から消す |
| HF のトークン | `~/.cache/huggingface/token` | `hf auth login` で入れ直す。トークンは huggingface.co/settings/tokens で作る（write 権限が要る） |
| Claude Code の資格情報 | `~/.claude/.credentials.json` | `claude` を起動してログインし直すだけ |
| vGPU のライセンストークン | `/etc/nvidia/ClientConfigToken/`（root のみ読める） | **自分では復旧できない。** 手順書 10 の 0 節を読むこと |

`~/.ssh/known_hosts` は秘密ではないので `files/home/ssh_known_hosts` に入れてある。
SECRETS

# ---------------------------------------------------------------- 詰める
log "4/4 tar に詰める"
mkdir -p "$OUT_DIR"
core="$OUT_DIR/machine_state_$STAMP.tar.gz"
tar czf "$core" -C "$STAGE" "machine_state_$STAMP"

echo
log "できたもの:"
log "  $core  ($(du -h "$core" | cut -f1))"

if [ "$WITH_TRANSCRIPTS" -eq 1 ] && [ -d "$HOME/.claude/projects" ]; then
    transcripts="$OUT_DIR/claude_transcripts_$STAMP.tar.gz"
    tar czf "$transcripts" -C "$HOME/.claude" projects
    log "  $transcripts  ($(du -h "$transcripts" | cut -f1))"
else
    log "  ~/.claude/projects（過去セッションの記録）は入れていない。要るなら --with-transcripts"
fi

echo
log "集めていないもの —— どれも取り直せる。判定は docs/howto/10_rebuild_machine.md の 5 節"
log "  重み（HF の pi05_stage1_24000・pi05_full_runpod にある）"
log "  提出 zip（HF の parc2026-submission にある）"
log "  HF キャッシュ 29 GB・評価の mp4 470 本・work/scripted・docker image・.venv・LIBERO-plus"
echo
log "次にやること: SECRETS_TODO.md を読み、manifest/git-unsaved.md が空かを確かめる"
