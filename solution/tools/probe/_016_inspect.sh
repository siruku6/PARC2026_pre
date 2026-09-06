#!/usr/bin/env bash
# 016 下調べ用の只read確認。docker run を script 経由で回す。
set -uo pipefail

docker run --rm parc2026-final bash -lc '
B=/workspace/LIBERO-plus/libero/libero/bddl_files/libero_10
echo "=== libero_10 の全 bddl（SCENE なしを含む） ==="
ls "$B" 2>/dev/null
echo
echo "=== SCENE 名のタスクだけ ==="
ls "$B" 2>/dev/null | grep SCENE
echo
echo "=== init_states がどこを指すか ==="
python3 -c "import os; os.environ[\"LIBERO_ROOT\"]=\"/workspace/LIBERO-plus\"; from libero.libero import get_libero_path; p=get_libero_path(\"init_states\"); print(p); import os as _o; print(sorted(_o.listdir(p))[:5] if _o.path.isdir(p) else \"NO DIR\")" 2>&1 | tail -3
'