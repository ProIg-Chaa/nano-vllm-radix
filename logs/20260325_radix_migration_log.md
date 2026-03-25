# Nano-vLLM Radix Migration Log - 2026-03-25
@proig
## Purpose
This file is the running implementation log for integrating SGLang-style radix/prefix reuse ideas into `nano-vllm-radix`.
Future modifications, experiments, refactors, and benchmark observations should continue to be appended here.

## Current Goal
Migrate the project step by step toward cross-request KV cache reuse for shared prefixes, while keeping each intermediate step verifiable and low risk.

## Phase 1 - Instrument Existing Prefix Cache

### Objective
Before changing scheduling or cache structure, verify whether the current block-level prefix cache is actually working, and quantify how much reuse it is already achieving.

### Files Changed
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/llm_engine.py`

### What Was Added
#### 1. BlockManager statistics
Added a `stats` dictionary to track:
- `alloc_requests`
- `dealloc_requests`
- `queried_blocks`
- `hit_blocks`
- `miss_blocks`
- `reused_tokens`
- `new_blocks`

Also added:
- `reset_stats()`
- `get_stats()`

#### 2. Allocation-path instrumentation
Inside `BlockManager.allocate()`:
- count each block lookup
- count hit/miss blocks
- count reused tokens
- count newly allocated blocks

Inside `BlockManager.deallocate()`:
- count deallocation requests

#### 3. Per-run stats printing
Inside `LLMEngine.generate()`:
- reset stats at the beginning of each generation run
- print a single `[prefix-cache] ...` summary line at the end

### Why This Step Was Necessary
This step established whether the current implementation already performs useful block-level prefix reuse.
Without this measurement, later radix-related refactors would be speculative.

### Validation Script
Created helper scripts in project root:
- `run_prefix_cache_test.sh`
- `run_prefix_cache_test_small.sh`
- `enter_nano_vllm_env.sh`

Also created experiment log directory:
- `logs/experiments/`

The test scripts activate the `nano_vllm` micromamba environment, run a synthetic shared-prefix workload, and save logs with timestamps.

### Synthetic Request Construction
The main test workload used token-id prompts directly rather than natural language text:

```python
common_prefix = [123] * 768
prompts = [
    common_prefix + [1000 + i, 2000 + i, 3000 + i]
    for i in range(8)
]
```

Interpretation:
- 8 requests total
- first 768 tokens identical across all requests
- final 3 tokens differ per request
- with block size 256, this yields 3 full shared blocks + 1 partial tail block

### Observed Result
Run log:
- `logs/experiments/prefix_cache_test_20260324_123920.log`

Observed stats:

```text
[prefix-cache] alloc_reqs=8 dealloc_reqs=8 queried_blocks=32 hit_blocks=21 miss_blocks=11 hit_rate=65.62% reused_tokens=5376 new_blocks=11
```

### Interpretation
This exactly matched the expected block-level reuse pattern:
- request 0: 4 misses
- requests 1-7: first 3 blocks hit, last partial block misses
- total hits = 7 * 3 = 21
- total reused tokens = 21 * 256 = 5376

### Conclusion From Phase 1
The existing implementation already supports useful block-level shared-prefix KV reuse.
The current limitation is not "cache does not work", but rather:
- reuse granularity is fixed at full blocks
- partial trailing prefixes are not reusable
- scheduling still does not use prefix match information early enough

## Phase 2 - Refactor BlockManager Into Layers

### Objective
Separate physical KV block management from prefix lookup/indexing, while preserving current runtime behavior.
This is a structural refactor intended to make later radix-style evolution much safer.

### File Changed
- `nanovllm/engine/block_manager.py`

### Main Design Change
Previously, `BlockManager` directly handled all of the following in one class:
- free/used block bookkeeping
- ref counts
- block allocation and release
- prefix hash computation
- hash lookup
- block commit/update
- runtime statistics

This was refactored into three logical layers:

#### 1. `KVBlockAllocator`
Responsibilities:
- own all `Block` objects
- own `free_block_ids`
- own `used_block_ids`
- allocate free blocks
- increment/decrement ref counts
- return blocks to free list when `ref_count == 0`

This layer does **not** know anything about token sequences or prefix matching.

#### 2. `PrefixCache`
Responsibilities:
- compute chained prefix hashes
- only produce reusable hashes for full blocks
- look up cached block ids from logical block content
- commit a logical block into the cache index

This layer does **not** manage free lists or ref counts.

#### 3. `BlockManager`
Responsibilities:
- preserve the original public interface used by `Scheduler`
- orchestrate calls between `PrefixCache` and `KVBlockAllocator`
- keep instrumentation/stats in one place

To preserve compatibility, these aliases were intentionally kept:
- `self.blocks`
- `self.free_block_ids`
- `self.used_block_ids`
- `self.hash_to_block_id`

### Why This Refactor Was Necessary
The original code mixed two different concerns:
- logical prefix matching
- physical memory/block lifecycle management

That coupling would make future steps risky.
For example, replacing the current hash-chain prefix lookup with a radix tree should mainly affect the prefix-cache layer, not the physical allocator.

This refactor establishes a cleaner future evolution path:
- change prefix lookup structure in `PrefixCache`
- change eviction behavior in allocator-related logic
- later adjust scheduling to match prefixes before checking allocation

### Important Constraint
This refactor was intentionally **behavior-preserving**.
It did **not** change:
- full-block-only reuse semantics
- cache hit criteria
- append semantics during decode
- scheduler call pattern

So the purpose of Phase 2 was not immediate speedup.
Its purpose was to make subsequent changes isolated and easier to reason about.

### Verification After Refactor
Ran the smaller shared-prefix test after the refactor.
Observed log:
- `logs/experiments/prefix_cache_test_small_20260324_130733.log`

Observed stats:

```text
[prefix-cache] alloc_reqs=4 dealloc_reqs=4 queried_blocks=16 hit_blocks=9 miss_blocks=7 hit_rate=56.25% reused_tokens=2304 new_blocks=7
```

Interpretation:
- request 0: 4 misses
- requests 1-3: each contributes 3 hits + 1 miss
- total hits = 9
- reused tokens = 9 * 256 = 2304

This matched the expected behavior exactly, confirming that the refactor preserved semantics.

## What Has Been Learned So Far
1. The current project already has a functioning block-level prefix cache path.
2. The real current limitation is granularity and scheduling, not absence of reuse.
3. A stable allocator/cache separation is necessary before attempting a radix-tree-style prefix structure.
4. Future work should proceed incrementally, with each step independently verifiable.

## Recommended Next Step
Phase 3 should make scheduling prefix-aware earlier in the decision process:
- match prefix first
- compute how many tail blocks actually need allocation
- then decide whether the request can enter the batch

This is the first step that should begin to produce systemic scheduling benefits beyond raw block reuse.


## Phase 3 - Make Prefill Scheduling Prefix-Aware

### Objective
Move prefix matching earlier in the prefill scheduling path so that scheduling decisions are based on the true uncached tail and the true number of free blocks still required.

### Files Changed
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`

### Problem Before This Step
Before Phase 3, the scheduler admitted prefill requests using the raw request size:
- batched token accounting used `len(seq)` before allocation
- block-capacity checks used `seq.num_blocks`

This missed an important distinction:
- some blocks may already be cached and currently used by another request
- some blocks may be cached but currently free and therefore only need to be re-reserved
- only the uncached tail contributes actual prefill compute

So the old scheduler was prefix-cache-aware only after allocation had already happened.

### Main Design Change
Added an explicit prefill planning phase in `BlockManager`.

#### New structures
Introduced:
- `PlanStepKind`
- `PlanStep`
- `PrefillPlan`

These describe how each logical prompt block should be handled during prefill allocation.

#### `BlockManager.make_prefill_plan(seq)`
This new method performs a dry-run prefix match and returns a `PrefillPlan` containing:
- `steps`: one step per logical block
- `cached_tokens`: how many prompt tokens are already reusable from prefix cache
- `required_free_blocks`: how many blocks still need to come out of the allocator free pool

The plan distinguishes three cases:
- `HIT_USED`: matching cached block is already pinned by another active request, so only `incref()` is needed
- `HIT_FREE`: matching cached block exists but is currently free, so it must be re-reserved from the free list
- `MISS`: no reusable block is available, so a new free block must be allocated

### Scheduler Change
`Scheduler.schedule()` now does this in prefill mode:
1. build a `PrefillPlan` first
2. compute `uncached_tokens = len(seq) - plan.cached_tokens`
3. check `max_num_batched_tokens` using `uncached_tokens`
4. check block capacity using `plan.required_free_blocks`
5. only then call `block_manager.allocate(seq, plan)`

This means prefix reuse now influences admission decisions before allocation happens.

### Why `required_free_blocks` Is Not Just Tail Miss Count
A subtle but important detail:
- if a cached block is currently free, it still needs to be taken out of `free_block_ids` again before reuse
- if a cached block is currently in use, reuse only requires `incref()` and consumes no new free slot

So the true capacity cost is:
- misses
- plus cached-hit blocks that are currently free

This is why the plan tracks `HIT_USED` vs `HIT_FREE` separately.

### Why This Step Matters
This is the first step where prefix caching begins to affect the system at the scheduling level, not only at execution time.

Previously, prefix cache reduced compute after a request was admitted.
Now, prefix cache also helps determine whether the request should be admitted into the prefill batch in the first place.

This is an important bridge toward more advanced radix-style scheduling because it introduces a clean "match first, allocate second" workflow.

### Behavior Preservation Notes
This step still intentionally preserves the current reuse semantics:
- only full blocks are reusable
- once a miss happens, later blocks in the same request are not considered cache hits
- decode append logic is unchanged
- the external `BlockManager` API still exists

So this is still an incremental step, not yet a radix-tree implementation.

### Validation After Phase 3
Re-ran the shared-prefix small regression test.
Observed log:
- `logs/experiments/prefix_cache_test_small_20260325_232020.log`

Observed stats:

```text
[prefix-cache] alloc_reqs=4 dealloc_reqs=4 queried_blocks=16 hit_blocks=9 miss_blocks=7 hit_rate=56.25% reused_tokens=2304 new_blocks=7
```

This matched the previous expected behavior exactly, confirming that the scheduling refactor preserved correctness for the current block-level cache model.

### What Was Learned From Phase 3
1. Prefix matching and physical allocation can now be reasoned about as two separate phases.
2. Free-block pressure depends on more than just cache misses; reusing a currently free cached block still consumes allocator capacity.
3. The project now has the right control-flow shape for later radix-style evolution: match first, then allocate.

### Recommended Next Step
Phase 4 should improve the prefix index itself, most likely by evolving `PrefixCache` beyond the current chained-hash lookup into a structure that is easier to extend toward radix-tree-style prefix matching.
