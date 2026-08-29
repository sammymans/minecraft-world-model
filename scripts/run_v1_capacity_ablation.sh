#!/bin/zsh

# Capacity ablation for the deterministic V1 spatial dynamics.
#
# Every point trains on the same nested 100,000-transition subset with the same
# twenty epochs and schedule; only --hidden-channels and --blocks change. The
# 64/3 point already exists as artifacts/v1-data-ablation/one-step-100000 and is
# not retrained.
#
# Recursive fine-tuning uses --rollout-steps 10, the optimum measured by the
# horizon ablation, so capacity is tested against the best known recipe.

set -euo pipefail

ablation_device=${MCWM_ABLATION_DEVICE:-mps}

for spec in 128:3 128:6; do
  hidden=${spec%%:*}
  blocks=${spec##*:}
  tag="h${hidden}b${blocks}"

  uv run mcwm train-spatial-dynamics \
    --maximum-transitions 100000 \
    --selection-policy nested_prefix \
    --hidden-channels "$hidden" \
    --blocks "$blocks" \
    --epochs 20 \
    --validation-every 2 \
    --patience 1000 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --output-dir "artifacts/v1-capacity-ablation/one-step-$tag" \
    --device "$ablation_device"

  uv run mcwm train-spatial-dynamics \
    --initial-checkpoint "artifacts/v1-capacity-ablation/one-step-$tag/best.pt" \
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

  uv run mcwm evaluate-rollout \
    --dynamics-checkpoint "artifacts/v1-capacity-ablation/multistep-$tag/best.pt" \
    --output-dir "artifacts/v1-capacity-ablation/eval-$tag" \
    --horizons 1 2 5 10 20 \
    --maximum-examples 5000 \
    --split validation \
    --device "$ablation_device"

  uv run mcwm compare-actions \
    --dynamics-checkpoint "artifacts/v1-capacity-ablation/multistep-$tag/best.pt" \
    --camera-step 30 \
    --output "artifacts/v1-capacity-ablation/action-comparison-$tag.png" \
    --device "$ablation_device"
done

# 64/3 baseline: reuse the existing nested 100K one-step checkpoint, but give it
# the same ten-step fine-tune so the comparison differs only in capacity.
uv run mcwm train-spatial-dynamics \
  --initial-checkpoint artifacts/v1-data-ablation/one-step-100000/best.pt \
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
  --output-dir artifacts/v1-capacity-ablation/multistep-h64b3 \
  --device "$ablation_device"

uv run mcwm evaluate-rollout \
  --dynamics-checkpoint artifacts/v1-capacity-ablation/multistep-h64b3/best.pt \
  --output-dir artifacts/v1-capacity-ablation/eval-h64b3 \
  --horizons 1 2 5 10 20 \
  --maximum-examples 5000 \
  --split validation \
  --device "$ablation_device"

uv run mcwm compare-actions \
  --dynamics-checkpoint artifacts/v1-capacity-ablation/multistep-h64b3/best.pt \
  --camera-step 30 \
  --output artifacts/v1-capacity-ablation/action-comparison-h64b3.png \
  --device "$ablation_device"
