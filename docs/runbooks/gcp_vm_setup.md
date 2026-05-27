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
| `GPUs (all regions)` | (global, no region) | ≥ 1 | ⏳ pending verification |

If `GPUs (all regions)` is 0, request an increase to 1:
- Tick the row's checkbox → **Edit Quota** at top
- Set new value to `1`, justify with: "Single L4 GPU VM for academic research on adaptive vision-language inference for table extraction. Budget ≤ $400."
- Submit; expect 4–24h wait. Stage 1 below blocks until approved.

---

(remaining sections will be filled in as we execute them — VM creation, SSH, repo clone, env setup, tmux workflow)
