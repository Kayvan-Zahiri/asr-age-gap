#!/bin/zsh
# Retry each arm: the run checkpoints per batch, so a Metal command-buffer
# failure costs only the batch in flight, and each restart resumes.
cd /Users/kayvan/Desktop/jobs/projects/asr-age-gap
run_arm () {
  local name=$1; shift
  for attempt in 1 2 3 4 5 6; do
    echo "=== $name (attempt $attempt) ==="
    python3 bench/run.py "$@" && return 0
    echo "=== $name FAILED, retrying from checkpoint ==="
    sleep 20
  done
  echo "=== $name GAVE UP ==="; return 1
}
run_arm "ARM 2: uncontrolled" --limit-shard 11 --accents any --no-match-accents \
        --per-bracket 1063 --tag uncontrolled
run_arm "ARM 3: eighties" --limit-shard 11 --accents native \
        --brackets twenties eighties --tag eighties
echo "=== ALL ARMS COMPLETE ==="
