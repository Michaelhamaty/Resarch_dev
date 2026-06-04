#!/usr/bin/env bash
# Over-match robustness run (fixed_2b @ 11 tiles, FinTab held-out) + scoring,
# then power the VM OFF no matter what — so a finished, failed, OR hung job
# never keeps billing.
#
# Design:
#   * The shutdown is in a `trap ... EXIT`, so it runs on success, on error,
#     and on timeout-kill. A plain `cmd && shutdown` would NOT shut down on
#     failure — the opposite of what you want for cost safety.
#   * `timeout` caps total wall-clock so a model runaway can't burn days.
#   * No `set -e`: we never want an early error to skip the shutdown trap.
#   * Output is tee'd to a timestamped log on the VM's (persistent) disk, so
#     you can read it after you restart the stopped instance.
#
# Usage (on the VM, inside tmux — see the launch command in chat):
#   ./scripts/scaleup/run_overmatch_and_shutdown.sh
#
# Pre-flight (run these ONCE before trusting it unattended):
#   sudo -n true && echo "passwordless sudo OK"   # else the shutdown will hang
#   # Confirm your cloud STOPS the instance on guest shutdown (stops billing),
#   # rather than just halting the OS while still charging. If it only halts,
#   # replace the shutdown line below with your cloud CLI stop, e.g.:
#   #   gcloud compute instances stop "$(hostname)" --zone=YOUR_ZONE -q
#   #   aws ec2 stop-instances --instance-ids YOUR_ID

set -uo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/overmatch_run_$(date -u +%Y%m%dT%H%M%SZ).log"
CONFIG="configs/experiment/scaleup_v2_overmatch.yaml"
TIMEOUT="4h"   # generous ceiling: one fixed system over 150 pages is ~1h even
               # with the matched-budget runaways noted in paper_measurables.md.

# Always power off on exit, whatever the reason. `sync` first so the log + any
# written artifacts are flushed to disk before the OS halts.
trap 'ec=$?; echo "[wrap] exit_code=$ec — powering off $(date -u)" | tee -a "$LOG"; sync; sudo shutdown -h now' EXIT

echo "[wrap] start $(date -u)  repo=$REPO_ROOT  log=$LOG" | tee -a "$LOG"

timeout --signal=TERM "$TIMEOUT" bash -lc "
  set -o pipefail
  uv run python scripts/scaleup/run_sweep.py \
      --config '$CONFIG' --datasets fintabnet --systems fixed_2b_matched &&
  uv run python scripts/scaleup/run_phase7_v2.py \
      --config '$CONFIG' --datasets fintabnet
" 2>&1 | tee -a "$LOG"

echo "[wrap] pipeline finished status=${PIPESTATUS[0]} $(date -u)" | tee -a "$LOG"
# trap fires here -> shutdown
