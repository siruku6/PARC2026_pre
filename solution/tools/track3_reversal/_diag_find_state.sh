#!/usr/bin/env bash
# find where observation.state is produced in the kit image
docker run --rm --entrypoint bash parc2026-final:latest -c "
  echo '=== LIBERO env state key definition ===';
  grep -rn \"\\\"state\\\"\" /workspace/LIBERO/libero/envs/ 2>/dev/null | grep -i 'sim\|state_vec\|eef\|flattened' | head -15;
  echo; echo '=== compe t3 rollout state usage ===';
  grep -rn \"observation.state\|\\\"state\\\"\|get_state\" /workspace/compe/t1 /workspace/compe/t2 /workspace/compe/t3 2>/dev/null | head -25;
"