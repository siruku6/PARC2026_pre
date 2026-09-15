#!/usr/bin/env bash
# 配布 T3 の 4 件について、指示文が init 50 行すべてで成り立つかを確かめる。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for t in book soup mug twomugs; do
  TASK=$t source "$here/_t3_container.sh"
  run_in_container check_instruction_coverage.py \
    --t3-bddl "$T3_BDDL" --t3-inits "$T3_INITS"
done
