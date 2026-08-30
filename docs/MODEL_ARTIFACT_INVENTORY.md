# Retained model and artifact inventory

Most of the local `artifacts/` directory is ignored by Git. The repository tracks
the selected V1 model, final figures, summary metrics, and ablation values. This
inventory records the additional models retained locally after the final cleanup.

| Stage | Retained artifacts |
|---|---|
| Initial flat model | `autoencoder/`, `dynamics/` |
| Data-scaled flat model | `autoencoder-v2/`, `dynamics-v2-new-ae/`, `rollout-v2/` |
| Spatial representation | `spatial-autoencoder-v3/` |
| Original spatial V1 | `spatial-dynamics-v4/`, `spatial-dynamics-v4-multistep/`, `spatial-rollout-v4-multistep/` |
| V1 data ablation | `v1-data-ablation/` |
| V1 horizon ablation | `v1-horizon-ablation/` |
| V1 capacity ablation and selected model | `v1-capacity-ablation/` |
| Final V2 diffusion experiment | `spatial-latent-diffusion-v2/full-v4/` |
| Writeup figures and video | `writeup-media/` |

The selected interactive model pair is:

```text
artifacts/spatial-autoencoder-v3/best.pt
artifacts/v1-capacity-ablation/multistep-h128b6/best.pt
```

The retained V2 checkpoint is:

```text
artifacts/spatial-latent-diffusion-v2/full-v4/best.pt
```

It is too large for regular GitHub storage and remains ignored locally. It can be
published separately as a GitHub Release asset.
