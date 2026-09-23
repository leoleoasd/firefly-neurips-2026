# Base image: you must build or provide this image yourself and point REPO/BASE_TAG at it.
ARG REPO=your-registry.example.com/rl-web-agent-base
ARG BASE_TAG=base
FROM ${REPO}:${BASE_TAG}
RUN condax install uv

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install python3-dev cmake -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# (1) make the uv cache path explicit so it's stable for mounts
ENV XDG_CACHE_HOME=/root/.cache
ENV UV_CACHE_DIR=/root/.cache/uv

RUN python3.10 -m pip install --upgrade pip setuptools

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
RUN mkdir -p /workdir/src/nemo_finetuning && touch /workdir/src/nemo_finetuning/__init__.py

# (2) seed deps with a cache mount (builder-local persistent cache)
RUN --mount=type=cache,id=uv-cache-sft,target=/root/.cache/uv \
    cd /workdir && uv sync

ENV CUDNN_PATH=/workdir/.venv/lib/python3.10/site-packages/nvidia/cudnn

RUN --mount=type=cache,id=uv-cache-sft,target=/root/.cache/uv <<'SH'
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J"
echo "$MAX_JOBS"
cd /workdir
uv sync --extra compile
SH


# Secrets must be provided at runtime, not baked into the image:
#   docker run -e WANDB_API_KEY=...   # for wandb logging
COPY . /workdir
RUN ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts

# The original build pinned the source to a private git repo. To restore that
# behavior, point the clone at your own repo, e.g.:
# RUN git clone --bare git@github.com:your-org/your-repo.git /workdir/.git
# RUN git -C /workdir/.git/ config --unset core.bare
# RUN git -C /workdir/ restore --staged .

WORKDIR /workdir

# RUN rsync -av patch/ ./
