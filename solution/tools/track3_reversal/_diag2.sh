#!/usr/bin/env bash
docker run --rm --entrypoint bash parc2026-final:latest -c "
  echo '=== compe tree ===';
  ls /workspace/compe 2>/dev/null; echo;
  echo '=== rollout.py location ===';
  find /workspace/compe -name 'rollout.py' 2>/dev/null | head;
  echo; echo '=== how obs.state / state is built in libero envs (broad) ===';
  grep -rn \"state\" /workspace/LIBERO/libero/envs/offscreen_env.py 2>/dev/null | head -20;
  echo '--- base env ---';
  grep -rn \"\\\"state\\\"\\|'state'\" /workspace/LIBERO/libero/envs/*.py 2>/dev/null | head -20;
"
