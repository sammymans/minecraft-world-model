# Tiny Minecraft World Model

A laptop-scale, action-conditioned world model for Minecraft. The project uses
public OpenAI VPT gameplay recordings to learn how movement and camera controls
change a compressed visual state, then recursively decodes its predictions into
future frames.

The selected V1 model combines a 253K-parameter spatial autoencoder with a
1.84M-parameter deterministic dynamics network. It responds meaningfully to
player actions, although its predictions become smoother over longer rollouts.
The repository also contains the larger V2 diffusion experiment and controlled
ablations over training-data size, recursive-training horizon, and model capacity.

## Interactive demo

```bash
uv sync
uv run mcwm play-rollout \
  --dynamics-checkpoint artifacts/v1-capacity-ablation/multistep-h128b6/best.pt \
  --sample-index 51575 \
  --camera-step 50
```

The playground uses a held-out processed episode as its initial scene, so the
VPT dataset must be downloaded and preprocessed first.

## Data pipeline

The CLI can download the public VPT data, preprocess it into synchronized
64x64 frame/action episodes, build a content-verified catalog, and optionally
publish that catalog to S3. Uploads are a dry run unless `--execute` is passed.

```bash
uv run mcwm dataset-download --manifest data/manifests/vpt_v4.jsonl
uv run mcwm dataset-preprocess \
  --manifest data/manifests/vpt_v4.jsonl \
  --output-dir data/processed/vpt_v4
```

Copy `.env.example` to `.env`, fill in temporary AWS credentials and a bucket,
then load those values into the shell before publishing. Boto3 also supports
the normal `~/.aws` profiles and IAM roles, so keys never need to enter the
repository.

```bash
set -a
source .env
set +a
uv run mcwm dataset-catalog
uv run mcwm dataset-publish-s3 artifacts/dataset-catalogs/vpt-v4.json \
  --bucket "$MCWM_S3_BUCKET" \
  --region "$AWS_DEFAULT_REGION" \
  --execute
```

Source code lives in `src/mcwm/`, dataset manifests live in `data/manifests/`,
and the local models and figures are summarized in
[the artifact inventory](docs/MODEL_ARTIFACT_INVENTORY.md). Large datasets,
intermediate checkpoints, and the V2 checkpoint are intentionally ignored by Git.
