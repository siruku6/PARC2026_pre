#!/usr/bin/env bash
# 部品から組み直すための一式を組み立てて、Hugging Face のデータセットへ置く。
#
# **混ぜたデータセットを消せるようにするためのものである。**手順・ソース・台帳を
# 1 か所に入れておけば、git が消えても部品から組み直せる。
#
#   bash solution/tools/regen/make_rebuild_bundle.sh                 # 組むだけ
#   UPLOAD=1 bash solution/tools/regen/make_rebuild_bundle.sh        # 上げる
#
# 上げ先は下の REPOS。**部品側（消さないほう）に必ず 1 つ置くこと**——
# 混合データを消したときに、手順ごと消えてしまわないようにするためである。
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

OUT="${OUT:-work/rebuild_bundle}"
UPLOAD="${UPLOAD:-0}"
REPOS=(siruku6/libero_plus_bowl5_initpose
       siruku6/libero_combined_bowl5_initpose_l90clean)
MANIFESTS=(l90clean_replace.json head_keep.json keep_replace_v2.json
           grader_texts.txt l90_dirty_summary.json)

COMMIT="$(git rev-parse --short HEAD)"
rm -rf "$OUT"
mkdir -p "$OUT/manifests"

# **ソースは git が追っているものだけを入れる。**work/ の中間物や __pycache__ が
# 混ざらないようにするため（`cp -r` ではそれが防げない）。
git archive --format=tar.gz --prefix=solution_src/ HEAD solution \
    > "$OUT/solution_${COMMIT}.tar.gz"
for name in "${MANIFESTS[@]}"; do
    cp "solution/tools/regen/manifests/$name" "$OUT/manifests/$name"
done
cp docs/cards/rebuild_bundle_README.md    "$OUT/README.md"
cp docs/cards/rebuild_bundle_README.ja.md "$OUT/README.ja.md"
echo "$COMMIT" > "$OUT/SOURCE_COMMIT.txt"

echo "組んだ: $OUT（$(du -sh "$OUT" | cut -f1)・版 $COMMIT）"

# **手順が名指ししているスクリプトが、本当に入っているかを確かめる。**
python3 - "$OUT/solution_${COMMIT}.tar.gz" "$OUT/README.ja.md" <<'PY'
import re
import sys
import tarfile

archive, recipe = sys.argv[1:3]
with tarfile.open(archive) as tar:
    inside = {name.removeprefix("solution_src/") for name in tar.getnames()}
named = set(re.findall(r"solution/[\w./-]+\.(?:sh|py)|solution/train/Dockerfile\.\w+",
                       open(recipe).read()))
missing = sorted(named - inside)
if missing:
    raise SystemExit(f"**手順が名指ししているのに入っていない: {missing}**")
print(f"  手順が名指しする {len(named)} 本すべてが入っている")
PY

[ "$UPLOAD" = 1 ] || { echo "（上げるなら UPLOAD=1）"; exit 0; }

python3 - "$OUT" "${REPOS[@]}" <<'PY'
import sys
from huggingface_hub import HfApi

api = HfApi()
folder, *repos = sys.argv[1:]
for repo in repos:
    url = api.upload_folder(folder_path=folder, path_in_repo="rebuild",
                            repo_id=repo, repo_type="dataset",
                            commit_message="部品から組み直すための一式を更新")
    print(f"  {repo}: {url}")
PY
