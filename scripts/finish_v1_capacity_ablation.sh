#!/bin/zsh

# Completes the capacity ablation after the first run was interrupted.
#
# Already on disk and skipped here:
#   one-step h128b3, h128b6, and the nested 100K h64b3 point
#   multistep + eval + action comparison for h128b3
#
# Remaining: the 1.84M recursive stage, and the matched 255K baseline whose
# one-step checkpoint comes from the same nested 100K subset (not the
# random-selection checkpoint used by the horizon ablation).

set -euo pipefail

ablation_device=${MCWM_ABLATION_DEVICE:-mps}

finish() {
  local tag=$1 initial=$2
  if [[ ! -f "artifacts/v1-capacity-ablation/multistep-$tag/best.pt" ]]; then
    uv run mcwm train-spatial-dynamics \
      --initial-checkpoint "$initial" \
      --maximum-transitions 30000 \
      --rollout-steps 10 \
      --horizon-decay 0.8 \
      --gradient-clip 1.0 \
      --maximum-validation-sequences 5000 \
      --epochs 10 \
      --validation-every 1 \
      --patience 1000 \
      --batch-size 32 \
      --learning-rate 1e-4 \
      --output-dir "artifacts/v1-capacity-ablation/multistep-$tag" \
      --device "$ablation_device"
  fi

  if [[ ! -f "artifacts/v1-capacity-ablation/eval-$tag/metrics.json" ]]; then
    uv run mcwm evaluate-rollout \
      --dynamics-checkpoint "artifacts/v1-capacity-ablation/multistep-$tag/best.pt" \
      --output-dir "artifacts/v1-capacity-ablation/eval-$tag" \
      --horizons 1 2 5 10 20 \
      --maximum-examples 5000 \
      --split validation \
      --device "$ablation_device"
  fi

  if [[ ! -f "artifacts/v1-capacity-ablation/action-comparison-$tag.png" ]]; then
    uv run mcwm compare-actions \
      --dynamics-checkpoint "artifacts/v1-capacity-ablation/multistep-$tag/best.pt" \
      --camera-step 30 \
      --output "artifacts/v1-capacity-ablation/action-comparison-$tag.png" \
      --device "$ablation_device"
  fi
}

finish h128b6 artifacts/v1-capacity-ablation/one-step-h128b6/best.pt
finish h64b3  artifacts/v1-data-ablation/one-step-100000/best.pt
finish h128b3 artifacts/v1-capacity-ablation/one-step-h128b3/best.pt
