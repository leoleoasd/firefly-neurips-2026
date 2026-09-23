# Local Modifications to Slime

This document describes the changes we made to the vendored copy of [Slime](https://github.com/THUDM/slime) in `thirdparty/slime/`. Our baseline is Slime commit `41dc3b6` (latest as of 2026-05-11). The full patch is saved at `patches/slime_local.patch`.

## Summary

5 files modified, 1 file deleted (`setup.py`). The changes fall into three categories: **packaging**, **performance/reliability**, and **bug fixes**.

## Changes by File

### 1. `pyproject.toml` — Rewrite for uv compatibility

**Why:** The original `pyproject.toml` uses a legacy `setup.py`-based build with inline dependencies. We need slime to be installable as an editable path dependency via uv with `dependencies = []` (all deps managed by the outer project).

**What changed:**
- Converted to PEP 621 `[project]` metadata: `name = "slime"`, `version = "0.2.4"`, `requires-python = ">=3.10"`
- Set `dependencies = []` — slime's runtime deps are declared in the outer `pyproject.toml`
- Added `[tool.setuptools.packages.find]` with `include = ["slime*", "slime_plugins*"]`
- Deleted `setup.py` (no longer needed)

### 2. `slime/ray/placement_group.py` — Improved GPU placement sorting

**Why:** When nodes have different numbers of GPUs (e.g. mixed cluster), the original sort only considers IP + GPU ID. This could place training on a node with fewer GPUs, wasting resources.

**What changed:**
- `sort_key()` now sorts by **(GPU count per node descending, node IP, GPU ID)**, so nodes with more GPUs are assigned first
- Uses `collections.Counter` to count GPUs per node, passed via `functools.partial`

### 3. `slime/ray/rollout.py` — Engine count overflow fix

**Why:** `num_engines_on_this_node` could exceed the actual number of remaining engines when the total isn't evenly divisible by engines-per-node, causing index errors.

**What changed:**
- Added `min(num_engines_per_node - (rank % num_engines_per_node), num_engines - rank)` to cap engine count

### 4. `slime/utils/http_utils.py` — Async JSON decode + connection pool stats

**Why:** `response.json()` is synchronous and blocks the event loop on large responses (logprobs, token data). Also needed visibility into httpx connection pool state for debugging throughput issues.

**What changed:**
- JSON decode wrapped in `asyncio.to_thread(response.json)` to avoid blocking
- Added `_log_pool_stats()` helper and `ensure_pool_stats_loop()` that periodically logs pool state (total/in-use/idle/closed/queued connections)

### 5. `examples/fully_async/fully_async_rollout.py` — Dynamic sampling filter support

**Why:** The fully-async rollout example doesn't support dynamic sampling filters (DAPO-style), making it unusable for our training pipeline.

**What changed:**
- Return type changed from `list[list[Sample]]` to `RolloutFnTrainOutput`
- Integrated `call_dynamic_filter()` and `MetricGatherer` from slime's filter hub
- Changed `max_concurrent_tasks` from `rollout_batch_size` to `over_sampling_batch_size`

## Changes Dropped from Previous Version

These modifications existed in the previous slime baseline (`c259823`) but are no longer needed:

- **`slime/router/router.py`** — Upstream removed the built-in router entirely (#1773), switching to sglang-router. Our streaming proxy and `/get_router_load` endpoint are no longer applicable.
- **`scripts/run-glm4-9B.sh`** — Wandb arg uncommenting was dev convenience only.
- **`tools/convert_torch_dist_to_hf.py`** — Debug print line, upstream has its own improvements.
- **Mooncake protocol env var** (sglang fork) — The Mooncake disaggregation module was restructured in sglang v0.5.9; the code we patched no longer exists.

## Applying the Patch

To apply our modifications to a fresh copy of slime at commit `41dc3b6`:

```bash
cd thirdparty/slime
patch -p1 < ../../patches/slime_local.patch
```

## Updating Slime

When upstream slime releases a new version:

1. Replace `thirdparty/slime/` with the new version (delete + clone + remove `.git/`)
2. Reapply local modifications — check `patches/slime_local.patch` for what needs to be reapplied
3. Check if upstream has incorporated any of our fixes (e.g. the engine count fix, async JSON decode)
4. Rebuild sglang and Megatron forks with the new patch files from `docker/patch/<version>/`
5. Update `pyproject.toml` sources to point to new fork commits
6. Run `uv lock` and resolve any dependency conflicts
7. Regenerate the patch: `diff -ruN <upstream_slime> thirdparty/slime --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude='slime.egg-info' --exclude='setup.py' > patches/slime_local.patch`
