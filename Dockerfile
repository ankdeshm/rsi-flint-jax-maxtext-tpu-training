# Dockerfile for Recursive Flint (39.8B MoE) JAX / MaxText TPU v5p Training
# Extends the project's MaxText TPU image in Artifact Registry and installs `flint_jax` + `safetensors` + `huggingface_hub`
FROM us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/maxtext/maxtext:latest

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface_hub>=0.26.0 \
    safetensors>=0.4.5 \
    flax>=0.10.0 \
    optax>=0.2.3 \
    orbax-checkpoint>=0.10.0

COPY flint_jax/ /app/flint_jax/
COPY maxtext/ /app/maxtext_flint/

ENV PYTHONPATH="/app:${PYTHONPATH:-}"
