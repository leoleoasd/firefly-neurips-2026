# Building a Custom Environment on Top of Slime

This document explains how slime_uv builds its environment on top of the upstream [Slime](https://github.com/THUDM/slime) RL training framework, and serves as a guide for anyone who wants to do the same.

## Background: How Slime Ships Its Environment

Slime's official Docker image (`thirdparty/slime/docker/Dockerfile`) does everything with `pip install` — no lockfile, no reproducibility. The build flow is:

1. Start from `slimerl/sglang:nightly-dev-*` (a pre-built sglang image with CUDA, torch, etc.)
2. `pip install` a long list of compiled dependencies (flash-attn, transformer_engine, apex, Megatron-LM, etc.)
3. `pip install -r requirements.txt` for lightweight Python deps (ray, wandb, transformers, etc.)
4. **Apply git patches** to Megatron-LM and sglang at build time (from `docker/patch/<version>/`)
5. `pip install -e .` slime itself

The key versions in the current stable release (see `thirdparty/slime/docker/README.md` and `thirdparty/slime/docker/Dockerfile`):

| Component | Version/Commit |
|---|---|
| sglang base image | `v0.5.9` (`bbe9c7ee`) |
| Megatron-LM | `3714d81d` (NVIDIA/Megatron-LM) |
| transformer_engine | 2.10.0 |
| flash-attn | 2.7.4.post1 |
| apex | `10417ace` |
| mbridge | `89eb1088` |
| torch_memory_saver | `a193d9dd` |
| Megatron-Bridge | `bridge` branch (radixark fork) |

The patches are the critical part — they modify sglang and Megatron to support Slime's RL training loop (weight update APIs, deterministic inference, MTP fixes, etc.). Without them, Slime won't work.

## What slime_uv Changes (and Why)

### 1. uv Instead of pip

**Problem:** Slime's pip-based build is not reproducible. Rebuilding the image a week later can pull different transitive dependency versions.

**Solution:** Use [uv](https://docs.astral.sh/uv/) with a `uv.lock` lockfile. Every dependency version is pinned, including transitive ones.

```toml
# pyproject.toml
[project]
requires-python = "==3.12.*"
dependencies = [
  "torch==2.11.0",
  "transformer_engine[pytorch]==2.10.0",
  "flash-attn==2.7.4.post1",
  # ... all deps declared here
]
```

### 2. Git Forks Instead of Runtime Patches

**Problem:** Slime applies `.patch` files to Megatron-LM and sglang during `docker build`. This is fragile — patches break when upstream changes, and you can't easily add your own modifications on top.

**Solution:** Maintain fork branches with the patches pre-applied. Declare them as uv git sources:

```toml
[tool.uv.sources]
megatron-core = {git = "https://github.com/leoleoasd/Megatron-LM.git", rev = "slime_patch_v5"}
sglang = {git = "https://github.com/leoleoasd/sglang.git", subdirectory = "python", rev = "slime_patch_v5"}
```

**How to create your own forks:**

1. Fork `NVIDIA/Megatron-LM`, checkout the commit Slime targets (e.g. `3714d81d`)
2. Apply the official patch: `git apply /path/to/slime/docker/patch/latest/megatron.patch`
3. Commit, push to your fork branch
4. Repeat for `sgl-project/sglang` at commit `bbe9c7ee`

**Our extra changes on top of the official patches:**

- **Megatron-LM:** None. Our fork (`leoleoasd/Megatron-LM@slime_patch_v5`) is upstream + the official `megatron.patch`. Zero additional modifications.
- **sglang:** One extra commit — makes the Mooncake transfer engine protocol configurable via `MOONCAKE_PROTOCOL` env var (defaults to `"rdma"`, needed for AWS EFA / TCP deployments). This is a one-line change in `disaggregation/mooncake/transfer_engine.py`.

### 3. Slime as a Git Subtree (Editable Install)

Instead of `git clone` + `pip install -e .` at build time, slime is vendored as a git subtree under `thirdparty/slime/` and declared as an editable path dependency:

```toml
[tool.uv.sources]
slime = {path = "thirdparty/slime", editable = true}
```

This means you can modify slime code locally and it takes effect immediately. To update slime, use `git subtree pull`.

### 4. Version Pins

We pin the core stack to **match Slime's official Dockerfile** (`thirdparty/slime/docker/Dockerfile`) so `uv sync` resolves the same binaries: torch **2.11.0+cu129**, transformer_engine **2.10.0**, flash-attn **2.7.4.post1**. `override-dependencies` in `pyproject.toml` is where these pins live (it forces every dependent onto the same torch/transformers build).

All git-pinned dependencies (apex, mbridge, torch_memory_saver, Megatron-Bridge) use **the exact same commits** as Slime upstream. The one deliberate upgrade is Python **3.12** (Slime targets 3.10+).

### 5. Additional Dependencies

slime_uv adds packages for its own agent workloads. These are irrelevant if you're just using Slime for training, but illustrate how to layer your own deps on top:

- Your own packages go in `[project] dependencies`
- Local packages use `[tool.setuptools.packages.find]`
- Vendored wheels go in a `wheels/` directory with path sources
- Thirdparty repos go in `thirdparty/` as subtrees with editable installs

## Step-by-Step: Building Your Own Environment

### Prerequisites

- A base Docker image with CUDA toolkit and Python 3.12
- [uv](https://docs.astral.sh/uv/) installed (or install it in the Dockerfile)

### Step 1: Create Your Project

```
my_project/
├── pyproject.toml
├── uv.lock              # generated by uv lock
├── .python-version      # "3.12"
├── Dockerfile
├── thirdparty/
│   └── slime/           # git subtree of THUDM/slime
├── wheels/              # vendored .whl files (if needed)
└── my_package/          # your code
```

### Step 2: Fork Megatron-LM and sglang

Check `thirdparty/slime/docker/README.md` for the current stable sglang and Megatron versions. Then:

```bash
# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout <MEGATRON_COMMIT>  # e.g. 3714d81d
git apply /path/to/slime/docker/patch/latest/megatron.patch --3way
git checkout -b my_slime_patch
git push origin my_slime_patch

# sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <SGLANG_COMMIT>  # e.g. bbe9c7ee
git apply /path/to/slime/docker/patch/latest/sglang.patch --3way
# Add your own changes here if needed
git checkout -b my_slime_patch
git push origin my_slime_patch
```

### Step 3: Write pyproject.toml

Key sections you need:

```toml
[project]
requires-python = "==3.12.*"
dependencies = [
  # === Slime's compiled deps (copy versions from slime/docker/Dockerfile) ===
  "torch==2.11.0",
  "flash-attn==2.7.4.post1",
  "transformer_engine[pytorch]==2.10.0",
  "apex",
  "megatron-core",
  "flash-linear-attention==0.4.1",
  "mbridge",
  "torch-memory-saver",
  "megatron-bridge",
  # === Slime's requirements.txt deps ===
  "accelerate", "blobfile", "datasets", "httpx[http2]",
  "omegaconf", "pillow", "pyyaml", "ray[default]",
  "ring-flash-attn", "sglang-router>=0.2.3",
  "tensorboard", "transformers", "wandb",
  "numpy<2",
  # === Slime itself ===
  "slime",
  # === Your own deps ===
  # ...
]

[tool.uv.sources]
# Your forks with slime patches pre-applied
megatron-core = {git = "https://github.com/YOU/Megatron-LM.git", rev = "my_slime_patch"}
sglang = {git = "https://github.com/YOU/sglang.git", subdirectory = "python", rev = "my_slime_patch"}
# Same commits as slime upstream
apex = {git = "https://github.com/NVIDIA/apex.git", rev = "10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"}
mbridge = {git = "https://github.com/ISEEKYAN/mbridge.git", rev = "89eb10887887bc74853f89a4de258c0702932a1c"}
megatron-bridge = {git = "https://github.com/radixark/Megatron-Bridge.git", rev = "bridge"}
torch-memory-saver = {git = "https://github.com/fzyzcjy/torch_memory_saver.git", rev = "a193d9dd1b877d33c64a41cfb3db9f867df2d926"}
# Slime as editable subtree
slime = {path = "thirdparty/slime", editable = true}
# torch from PyTorch cu129 index
torch = [{index = "pytorch-cu129", marker = "sys_platform == 'linux'"}]

[tool.uv]
# These packages need torch visible at compile time
no-build-isolation-package = [
  "flash-attn", "transformer-engine", "transformer-engine-cu12",
  "transformer-engine-torch", "apex",
]
# Pin cudnn to avoid incompatibilities
override-dependencies = ["nvidia-cudnn-cu12==9.16.0.29"]
prerelease = "allow"
# Build group: torch must be installed so native extensions can compile.
# Make it a default group so a plain `uv sync` includes it.
default-groups = ["build"]

[[tool.uv.index]]
explicit = true
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"

[dependency-groups]
build = ["torch==2.11.0", "nvidia-nvshmem-cu12>=3.3.20"]
```

### Step 4: Write the Dockerfile

A single `uv sync` is enough — the `build` dependency group is a default group, so torch is part of the same resolution and native extensions (flash-attn/apex/TE) find torch headers at compile time:

```dockerfile
FROM your-base-image-with-cuda

# Install uv
RUN condax install uv  # or pip install uv

# Copy only dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock .python-version /workdir/
COPY thirdparty/slime/pyproject.toml /workdir/thirdparty/slime/pyproject.toml

WORKDIR /workdir

# Full install with parallel compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    MAX_JOBS=$(nproc) NVCC_APPEND_FLAGS="--threads 4" \
    APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv sync

# Fix nvshmem symlinks (known issue)
RUN cd .venv/lib/python3.12/site-packages/nvidia/nvshmem/lib && \
    ln -sf libnvshmem_host.so.3 libnvshmem_host.so
    # ... other symlinks as needed

# Copy your code
COPY . /workdir
```

### Step 5: Lock and Build

```bash
# Generate lockfile
uv lock

# Build Docker image
docker buildx build -t my-slime-env .
```

## Gotchas and Tips

1. **`no-build-isolation-package` is critical.** flash-attn, apex, and transformer_engine all need to see torch at compile time. Without this, they'll fail to find CUDA headers.

2. **The `build` group must be a default group.** It carries torch (and `nvidia-nvshmem-cu12`) into every plain `uv sync`, so the `no-build-isolation-package` sdists compile against an env that already has torch — no separate bootstrap sync needed.

3. **nvshmem symlinks.** The nvidia-nvshmem-cu12 package ships `.so.3` files but some consumers expect unversioned `.so` names. You need to create symlinks manually.

4. **`override-dependencies` for cudnn.** Different packages may pull different cudnn versions. Pin it explicitly to avoid conflicts.

5. **flash-attn metadata override.** uv can't extract metadata from flash-attn's source build, so you need to declare it manually:
   ```toml
   [[tool.uv.dependency-metadata]]
   name = "flash-attn"
   requires-dist = ["torch", "einops"]
   version = "2.7.4.post1"
   ```

6. **When Slime updates patches,** you need to rebase your fork branches. Check `thirdparty/slime/docker/patch/` for new patch versions and re-apply them to your forks.

7. **`numpy<2` is required** by Megatron. Don't let it float to numpy 2.x.

8. **Slime's `requirements.txt`** (`thirdparty/slime/requirements.txt`) lists the lightweight Python deps that slime needs at runtime. Make sure they're all in your `[project] dependencies`.
