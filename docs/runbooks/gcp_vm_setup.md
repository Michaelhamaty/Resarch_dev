# GCP VM Setup — Scale-Up v2

**Purpose:** Reproducible setup for the single GCE L4-24GB VM that runs the scale-up v2 sweep.
**Plan:** [`docs/specs/scaleup_v2_plan.md`](../specs/scaleup_v2_plan.md) — Stage 1.
**Audience:** Author + professor; assumes no prior GCP experience.

---

## Target configuration

| Setting | Value | Why |
|---|---|---|
| Project ID | `hmaty-496720` | Professor's project with billing attached |
| Region | `us-central1` | Best L4 availability, lowest cost |
| Zone | `us-central1-a` (or any `-a/-b/-c` with L4 capacity at create time) | Inside chosen region |
| Machine type | `g2-standard-8` | 8 vCPU + 32 GB RAM + 1× L4 24 GB GPU |
| Image | `pytorch-latest-gpu` from `deeplearning-platform-release` | CUDA + PyTorch preinstalled, saves ~1 hour of driver pain |
| Boot disk | 200 GB SSD (`pd-ssd`) | Holds model checkpoints (~30 GB), datasets (~30 GB), outputs, headroom |
| Network | SSH only (no public ports) | Security |
| Provisioning | On-demand (NOT spot) | Resumability headache not worth ~2.5× savings on a 14-hour job |
| Estimated cost | ~$0.80/hr running, ~$0.04/GB/month stopped | Stop the VM whenever you're not running |

---

## Step 1 — Verify quotas (do this first)

Skip a quota request only when **both** of these are confirmed in
**Console → IAM & Admin → Quotas & System Limits**:

| Quota name | Filter | Required value | Status as of setup |
|---|---|---|---|
| `NVIDIA L4 GPUs` | region: `us-central1` | ≥ 1 | ✅ verified = 1 |
| `GPUs (all regions)` | (global, no region) | ≥ 1 | ✅ approved 2026-05-27 |

Both quotas confirmed; Stage 1 (VM creation) is unblocked.

---

## Step 2 — Install and authenticate `gcloud` (local laptop, one-time)

```bash
# macOS via Homebrew (other OSes: https://cloud.google.com/sdk/docs/install)
brew install --cask google-cloud-sdk

# Authenticate the local CLI with the account that owns hmaty-496720
gcloud auth login
gcloud config set project hmaty-496720
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Verify
gcloud config list
gcloud compute instances list   # should succeed (likely empty)
```

---

## Step 3 — Create the VM

Run from the laptop. Picks the first zone in `us-central1` that has L4 capacity at create time.

```bash
gcloud compute instances create scaleup-v2 \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform \
    --no-address \
    --tags=ssh-only
```

Flag notes:
- `--accelerator type=nvidia-l4,count=1` attaches the L4 GPU to the VM. Without this the `g2-standard-8` will fail to start.
- `--maintenance-policy=TERMINATE` is required for GPU VMs (they can't live-migrate).
- `--metadata="install-nvidia-driver=True"` triggers the Deep Learning VM image's driver install on first boot.
- `--no-address` skips assigning a public IPv4 (we SSH via IAP). Saves $0.005/hr and reduces attack surface.
- If `us-central1-a` returns `ZONE_RESOURCE_POOL_EXHAUSTED`, retry with `--zone=us-central1-b` or `-c`.

Expected: command returns within ~60s with `RUNNING` status. The driver install runs on first SSH login (~3 min extra).

---

## Step 4 — SSH in via IAP, accept the driver-install prompt

```bash
# First connection — accepts host key, triggers SSH config generation
gcloud compute ssh scaleup-v2 --tunnel-through-iap

# On first login the VM prompts:
#   "This VM requires Nvidia drivers to function correctly..."
#   Answer y. Wait ~3 min for the install + reboot.
# Reconnect after the reboot:
gcloud compute ssh scaleup-v2 --tunnel-through-iap
```

---

## Step 5 — Verify GPU + Python on the VM

Inside the SSH session:

```bash
# Confirm L4 visible
nvidia-smi
# Expected: "NVIDIA L4 ... 24564MiB" in the device table.

# Confirm PyTorch sees the GPU (Deep Learning VM image preinstalls it)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: "2.x.y True NVIDIA L4"
```

If `torch.cuda.is_available()` returns False, the driver install hasn't finished — `sudo reboot`, reconnect, retry.

---

## Step 6 — Install `uv`, clone repo, install Python deps

```bash
# Inside the VM SSH session

# uv (fast pip/venv replacement we use everywhere)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repo. Repo name is typoed upstream (Resarch_dev, no 'e').
# Using HTTPS + a GitHub Personal Access Token is the lowest-friction
# path; alternative is an SSH key in `~/.ssh/` registered on GitHub.
git clone https://github.com/Michaelhamaty/Resarch_dev.git
cd Resarch_dev
git checkout scaleup/v2

# Sync Python deps (creates .venv, installs from uv.lock)
uv sync --extra dev

# Tests should pass before any GPU work
uv run pytest -q
# Expected: "360 passed"
```

When `git clone` prompts for a password, paste a GitHub PAT (Settings → Developer settings → Personal access tokens; scope `repo` is sufficient).

---

## Step 7 — HuggingFace login (for InternVL2 model pull)

```bash
# Inside the VM SSH session, in the repo dir
uv run hf auth login
# Paste the read-token from your HF account when prompted.
# Confirm with: uv run hf whoami
```

InternVL2-2B and InternVL2-8B are public on HuggingFace; the login is needed for the model download cache + rate-limit headroom.

---

## Step 8 — Rsync data + splits from laptop to VM

The FinTabNet fixture (~250 PNGs + parquet-derived records) and the
scale-up v2 split manifests are not in git. Push them up from the
laptop:

```bash
# Run on LAPTOP, not the VM
LAPTOP_REPO=/Users/michaelhamaty/Developer/Research_claude

# Use IAP-tunneled rsync via gcloud (no public IP needed)
gcloud compute scp --recurse --tunnel-through-iap \
    "$LAPTOP_REPO/data/fintabnet" \
    "$LAPTOP_REPO/data/omnidocbench" \
    "$LAPTOP_REPO/data/splits" \
    scaleup-v2:~/Resarch_dev/data/
```

Confirm on the VM:

```bash
ls ~/Resarch_dev/data/fintabnet/   # images/  ground_truth.json  manifest.jsonl  records.json
ls ~/Resarch_dev/data/omnidocbench/
ls ~/Resarch_dev/data/splits/scaleup_v2/{omnidocbench,fintabnet}/
```

---

## Step 9 — tmux workflow (use this for every run > 5 min)

Every long-running job must run inside a tmux session so a dropped SSH
connection or laptop sleep does not kill the work.

```bash
# Inside the VM SSH session

# Start a named session (only needed once per logical job)
tmux new -s sweep
# (you are now inside tmux; everything you run here survives disconnects)

# Run your job, e.g.:
uv run python scripts/scaleup/smoke_8b_one_page.py

# DETACH (job keeps running): press Ctrl+B, then d
# You're back at the regular shell; the job continues.

# REATTACH later (from any SSH session):
tmux attach -t sweep

# LIST sessions:
tmux ls

# KILL a session you no longer need:
tmux kill-session -t sweep
```

**Verification ritual the first time you use tmux on this VM:** start a
session, run `sleep 60 && echo done`, detach with `Ctrl+B d`, close the
SSH connection entirely, re-SSH, `tmux attach -t sweep`, confirm
"done" was printed. Once verified you're safe for the overnight sweep.

---

## Step 10 — Stop the VM when not running

Billing for GPU time is per-second while the VM is `RUNNING`. Stop it
between sessions; the boot disk persists at ~$0.04/GB-month (~$8/month
for 200 GB).

```bash
# From the laptop
gcloud compute instances stop scaleup-v2

# Resume later (same disk, same data, same Python env)
gcloud compute instances start scaleup-v2
```

---

## Step 11 — Gate G1 checklist (must all pass before Stage 2)

- [ ] `nvidia-smi` on the VM shows NVIDIA L4 with 24 GB.
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- [ ] `uv run pytest -q` on the VM prints `360 passed`.
- [ ] `~/Resarch_dev/data/fintabnet/`, `data/omnidocbench/`, and `data/splits/scaleup_v2/` all populated.
- [ ] `uv run hf whoami` prints your HF username.
- [ ] tmux detach/reattach cycle verified.

When every box is checked, proceed to plan Stage 2 (8B smoke test).

---

## Troubleshooting

- **`ZONE_RESOURCE_POOL_EXHAUSTED`**: retry `gcloud compute instances create` with `--zone=us-central1-b` or `-c`.
- **`PERMISSION_DENIED: IAM Service Account ... cloud-platform`**: your account needs the `Compute Instance Admin (v1)` role on the project. Grant via Console → IAM.
- **SSH hangs on first connect**: the IAP tunnel needs ~30s the first time per VM. If it still hangs, ensure the project has the `iap.googleapis.com` API enabled (`gcloud services enable iap.googleapis.com`).
- **`nvidia-smi: command not found`**: driver install hasn't finished. `sudo reboot`, wait 90s, reconnect.
- **rsync via gcloud scp is slow**: ~250 PNGs at ~100 KB each is ~25 MB; should be < 30 s. If it stalls, check that `gcloud compute config-ssh` wrote a working host entry to `~/.ssh/config`.
- **Pre-existing VM with different name**: `gcloud compute instances list` shows what's running; if a leftover VM is consuming the L4 quota, stop or delete it before creating `scaleup-v2`.

---

## Cost expectations during Stage 1

| Activity | Wall clock | GPU-hr | Cost |
|---|---:|---:|---:|
| Steps 2–6 (create + driver install + repo + deps) | ~25 min | 0.4 | ~$0.35 |
| Step 7–8 (HF login + rsync) | ~5 min | 0.1 | ~$0.10 |
| Step 9 verification | ~5 min | 0.1 | ~$0.10 |
| **Stage 1 total** | **~35 min** | **0.6** | **~$0.55** |

Once Stage 1 is complete, *stop* the VM (Step 10) until you're ready
for Stage 2.

