# Base image: you must build or provide this image yourself and point REPO/BASE_TAG at it.
ARG REPO=your-registry.example.com/rl-web-agent-base
ARG BASE_TAG=base
FROM ${REPO}:${BASE_TAG}

SHELL ["/bin/bash", "-c"]

RUN condax install uv

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install python3-dev cmake -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
COPY .python-version /workdir/.python-version
COPY thirdparty/verl/pyproject.toml /workdir/thirdparty/verl/pyproject.toml
COPY thirdparty/verl/uv.lock /workdir/thirdparty/verl/uv.lock
COPY thirdparty/webarena/setup.py /workdir/thirdparty/webarena/setup.py
COPY thirdparty/webarena/setup.cfg /workdir/thirdparty/webarena/setup.cfg
COPY thirdparty/webarena/requirements.txt /workdir/thirdparty/webarena/requirements.txt

RUN <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
conda activate base
cd /workdir
uv sync
SH
ENV CUDNN_PATH=/workdir/.venv/lib/python3.12/site-packages/nvidia/cudnn

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J"
echo "$MAX_JOBS"
cd /workdir
uv sync --extra compile --extra webarena
SH
RUN /workdir/.venv/bin/playwright install-deps
RUN /workdir/.venv/bin/playwright install


# Secrets must be provided at runtime, not baked into the image:
#   docker run -e WANDB_API_KEY=...   # for wandb logging


COPY . /workdir


WORKDIR /workdir
