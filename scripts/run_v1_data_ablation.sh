#!/bin/zsh

set -euo pipefail

ablation_device=${MCWM_ABLATION_DEVICE:-mps}

for spec in 10000:200:20 50000:40:4 100000:20:2; do
  transitions=${spec%%:*}
  remainder=${spec#*:}
  epochs=${remainder%%:*}
  validation_every=${remainder##*:}
  uv run mcwm train-spatial-dynamics \
    --maximum-transitions "$transitions" \
    --selection-policy nested_prefix \
    --epochs "$epochs" \
    --validation-every "$validation_every" \
    --patience 1000 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --output-dir "artifacts/v1-data-ablation/one-step-$transitions" \
    --device "$ablation_device"
done

for transitions in 10000 50000 100000; do
  uv run mcwm train-spatial-dynamics \
    --initial-checkpoint "artifacts/v1-data-ablation/one-step-$transitions/best.pt" \
    --maximum-transitions 30000 \
    --selection-policy nested_prefix \
    --rollout-steps 5 \
    --horizon-decay 0.8 \
    --gradient-clip 1.0 \
    --maximum-validation-sequences 5000 \
    --epochs 10 \
    --validation-every 1 \
    --patience 1000 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --output-dir "artifacts/v1-data-ablation/multistep-$transitions" \
    --device "$ablation_device"
done

for transitions in 10000 50000 100000; do
  uv run mcwm evaluate-rollout \
    --dynamics-checkpoint "artifacts/v1-data-ablation/multistep-$transitions/best.pt" \
    --output-dir "artifacts/v1-data-ablation/rollout-$transitions" \
    --horizons 1 2 5 10 20 \
    --maximum-examples 5000 \
    --split validation \
    --device "$ablation_device"
  uv run mcwm compare-actions \
    --dynamics-checkpoint "artifacts/v1-data-ablation/multistep-$transitions/best.pt" \
    --camera-step 30 \
    --output "artifacts/v1-data-ablation/action-comparison-$transitions.png" \
    --device "$ablation_device"
done
