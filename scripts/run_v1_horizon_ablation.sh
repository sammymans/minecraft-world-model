#!/bin/zsh

# Rollout-horizon ablation for the deterministic V1 spatial dynamics.
#
# Every point fine-tunes the same one-step checkpoint with the same 30,000
# windows, ten epochs, learning rate, gradient clip, and horizon decay. Only
# --rollout-steps changes. The five-step point already exists as the deployed
# artifacts/spatial-dynamics-v4-multistep checkpoint and is not retrained.

set -euo pipefail

ablation_device=${MCWM_ABLATION_DEVICE:-mps}
initial_checkpoint=artifacts/spatial-dynamics-v4/best.pt

for steps in 10 20; do
  uv run mcwm train-spatial-dynamics \
    --initial-checkpoint "$initial_checkpoint" \
    --maximum-transitions 30000 \
    --rollout-steps "$steps" \
    --horizon-decay 0.8 \
    --gradient-clip 1.0 \
    --maximum-validation-sequences 5000 \
    --epochs 10 \
    --validation-every 1 \
    --patience 1000 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --output-dir "artifacts/v1-horizon-ablation/rollout-steps-$steps" \
    --device "$ablation_device"
done

# The five-step baseline is the deployed checkpoint; score it through the same
# evaluator so all three points share one window set and one set of baselines.
uv run mcwm evaluate-rollout \
  --dynamics-checkpoint artifacts/spatial-dynamics-v4-multistep/best.pt \
  --output-dir artifacts/v1-horizon-ablation/eval-steps-5 \
  --horizons 1 2 5 10 20 \
  --maximum-examples 5000 \
  --split validation \
  --device "$ablation_device"

for steps in 10 20; do
  uv run mcwm evaluate-rollout \
    --dynamics-checkpoint "artifacts/v1-horizon-ablation/rollout-steps-$steps/best.pt" \
    --output-dir "artifacts/v1-horizon-ablation/eval-steps-$steps" \
    --horizons 1 2 5 10 20 \
    --maximum-examples 5000 \
    --split validation \
    --device "$ablation_device"
  uv run mcwm compare-actions \
    --dynamics-checkpoint "artifacts/v1-horizon-ablation/rollout-steps-$steps/best.pt" \
    --camera-step 30 \
    --output "artifacts/v1-horizon-ablation/action-comparison-steps-$steps.png" \
    --device "$ablation_device"
done
