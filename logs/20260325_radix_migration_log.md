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


## Phase 4 - Evolve PrefixCache Into a Block-Level Prefix Tree

### Objective
Replace the current flat chained-hash prefix index with a tree-shaped block-prefix structure, while preserving the current block-level reuse semantics and the Phase 3 scheduling flow.

### File Changed
- `nanovllm/engine/block_manager.py`

### Problem Before This Step
After Phase 3, the scheduler already used a prefix-aware planning step, but the prefix index itself was still logically flat:
- prefix lookup still depended on chained hash lookup
- future extension toward radix-style structure was not yet represented directly in the data structure

So the control flow had become "match first, allocate second", but the prefix cache itself had not yet become tree-shaped.

### Main Design Change
Introduced a block-level prefix tree inside `PrefixCache`.

#### New structure
Added `PrefixTreeNode` with:
- `block_id`
- `block_hash`
- `token_ids`
- `children`

The cache now has a root node and stores full-block prefixes by walking parent -> child edges keyed by full block token tuples.

### How Lookup Works Now
For each full prompt block:
- start from the current prefix node
- use the block token tuple as the child key
- if that child exists, the corresponding cached block can be considered for reuse
- if it does not exist, the prefix path ends there

This makes the logical prefix structure explicit rather than implicit in a flat hash dictionary.

### How Commit Works Now
When a block becomes committable:
- find or create the child node under the current prefix node
- update that node with the latest `block_id` and `block_hash`
- store the block tokens on the node
- keep the old `hash_to_block_id` flat mapping as a compatibility/debugging view

So after this step:
- the tree is the source of truth for prefix traversal
- the flat hash map is retained as a compatibility mirror

### Why This Matters
This is the first step where the prefix cache data structure itself becomes structurally closer to a radix-style cache.

It is still block-granular and not yet a full SGLang-style radix tree, but it changes the representation from:
- flat mapping over chained hashes

to:
- explicit prefix path over block tokens

That makes later work easier in several ways:
- longest-prefix-style traversal becomes more natural
- prefix state is now attached to a path, not just a flat key
- future node-level policies such as eviction metadata or subtree statistics become easier to add

### Important Constraint
This phase still intentionally preserves current semantics:
- only full blocks enter the tree
- partial tail blocks are still not reusable
- once a miss occurs inside a request, later blocks are still treated as misses for the current implementation path
- scheduler interface and prefill plan interface stay unchanged

So this is still an intermediate architectural step, not yet full radix attention.

### Validation After Phase 4
Ran the shared-prefix small regression test.
Observed log:
- `logs/experiments/prefix_cache_test_small_20260326_172607.log`

Observed stats:

```text
[prefix-cache] alloc_reqs=4 dealloc_reqs=4 queried_blocks=16 hit_blocks=9 miss_blocks=7 hit_rate=56.25% reused_tokens=2304 new_blocks=7
```

This matched the previous expected result exactly, showing that the tree-based prefix cache preserves the existing block-level behavior.

### What Was Learned From Phase 4
1. The control flow and the prefix data structure are now aligned: both are prefix-aware.
2. The cache has moved from an implicit prefix representation to an explicit path representation.
3. The project is now materially closer to radix-style evolution, even though the granularity is still full block rather than arbitrary token spans.

### Recommended Next Step
Phase 5 should decide whether to continue in one of two directions:
- enhance the tree with richer node metadata and eventual eviction/pinning policy
- reduce the logical reuse granularity beyond full blocks, which would require a more substantial change to addressing and cache representation


## Phase 5 - Make The Prefix Tree Ownership-Safe

### Objective
Make the block-level prefix tree safer to maintain by ensuring each physical `block_id` has a single authoritative owner node in the tree, and by pruning stale subtrees when a block is reused for a different prefix path.

### File Changed
- `nanovllm/engine/block_manager.py`

### Problem Before This Step
After Phase 4, the cache had an explicit tree structure, but it still had a correctness/maintenance risk:
- when a free block was later reused for a different prefix path
- the old tree node could still keep pointing to that same `block_id`
- deeper descendants under that stale node could remain in the tree even though the path was no longer valid

The runtime lookup was still guarded by token equality checks, so many stale cases would degrade into cache misses rather than silent wrong reuse.
However, the tree would gradually accumulate outdated ownership relationships, making future eviction or policy logic harder and less trustworthy.

### Main Design Change
Added ownership and maintenance metadata to `PrefixTreeNode` and `PrefixCache`.

#### New node metadata
`PrefixTreeNode` now records:
- `parent`
- `key_from_parent`
- `depth`
- `touch_count`
- `last_access_tick`

This turns the tree into a structure with explicit parent/child identity and access history, rather than only forward child edges.

#### New cache metadata
`PrefixCache` now tracks:
- `block_to_node`: reverse mapping from physical `block_id` to its owning tree node
- `access_tick`: monotonic counter for node touch history

### New Safety Rule
A physical cache block may only have **one** authoritative owner node in the prefix tree at a time.

When `commit()` assigns a block to a node:
- look up whether that block id already belongs to another node
- if so, and it is not the same node, prune the old subtree first
- then bind the block to the new node

This guarantees that the tree does not keep multiple conflicting prefix paths attached to the same physical block id.

### Subtree Pruning
Added `_prune_subtree()` which recursively:
- removes descendant ownership information
- removes flat hash mirror entries when they still point to the stale node
- removes reverse block ownership entries
- detaches the subtree from its parent

This is important because if an internal block in a prefix path becomes invalid, the descendants under that path are no longer semantically usable either.

### Why This Matters
This phase does not change current hit-rate semantics, but it makes the tree structurally trustworthy for later work.

Without this step, later features such as:
- node-level eviction metadata
- subtree-based policies
- radix-style maintenance logic
would be built on top of a tree that can retain stale ownership history.

After this step, the prefix tree is no longer just explicit; it is also ownership-safe.

### Additional Benefit
The new metadata also gives the project a place to hang future policies:
- `touch_count` and `last_access_tick` can later support cache retention / recency heuristics
- `parent` and `depth` make subtree operations and ancestor-aware policies easier

### Validation After Phase 5
Re-ran the shared-prefix small regression test.
Observed log:
- `logs/experiments/prefix_cache_test_small_20260326_190538.log`

Observed stats:

```text
[prefix-cache] alloc_reqs=4 dealloc_reqs=4 queried_blocks=16 hit_blocks=9 miss_blocks=7 hit_rate=56.25% reused_tokens=2304 new_blocks=7
```

This matched the expected block-level behavior exactly, confirming that ownership-safe tree maintenance did not change the current reuse semantics.

### What Was Learned From Phase 5
1. Explicit prefix structure is not enough; ownership consistency also matters.
2. Guarding correctness only at lookup time is not sufficient if the goal is to evolve the cache into a richer tree-managed system.
3. Reverse ownership and subtree pruning are natural prerequisites for later eviction or retention policy work.

### Recommended Next Step
Phase 6 should choose between two directions:
- add an actual node-level retention / eviction policy using the metadata added in Phase 5
- start attacking the bigger architectural gap: reuse granularity finer than full blocks

## Phase 6 - Return To Stage A And Add Minimal Retention / Eviction

### Route Choice
At this point the project deliberately returned to **Stage A** as the mainline.
The rationale was:
- the long-term target is still SGLang-style radix integration
- but the current system still needs a more operational block-level cache manager before finer-grained reuse is introduced
- therefore the next step should not yet attack partial-block reuse
- instead it should give the existing prefix tree a minimal, explicit retention / eviction policy

This keeps the cache manager evolution incremental:
1. stabilize the block-level tree as a managed cache
2. then move toward finer logical reuse granularity and more SGLang-like addressing

### Objective
Add a low-risk node-level retention framework so the prefix tree can explicitly drop cold cached leaves under allocation pressure, rather than keeping every free cached path indefinitely.

### Files Changed
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/llm_engine.py`

### Main Design Change
Introduced the first minimal eviction behavior for the tree.
The tree now distinguishes between:
- active blocks that are still in use by requests
- free cached leaves that may be retained
- free cached leaves that may be evicted when free-block pressure becomes high

This is still block-granular and intentionally conservative.
It does not change prefix hit semantics.

### New Prefix Cache Capabilities
`PrefixTreeNode` now exposes `is_leaf`, and `PrefixCache` now supports:
- iterating leaves
- collecting evictable leaves
- selecting a cold eviction candidate
- evicting a single retained leaf

A leaf is considered evictable only if:
- it corresponds to a real cached block id
- it has no children
- its physical block is not currently in active use
- it is not protected by the current allocation plan

### Eviction Policy
Added a minimal low-watermark retention rule in `BlockManager`.

`BlockManager` now has:
- `retention_low_watermark`
- `apply_retention_policy(required_free_blocks, protected_block_ids)`

Behavior:
- before prefill allocation, protect the blocks referenced by the current plan
- if projected free blocks would fall below the watermark, prune a small number of cold retained leaves
- when decode needs a new physical block, apply the same policy before allocating the next free block

Candidate selection is recency-biased:
- older `last_access_tick` is evicted first
- lower `touch_count` is evicted first on ties
- deeper leaves are preferred on ties to avoid prematurely dropping shallower shared structure

### Why This Matters
This phase does not increase reuse granularity.
Partial blocks are still not reusable.
However, it gives the tree a real cache-management behavior instead of treating every freed full block as retained forever.

That matters for the next stages because:
- the tree now has an explicit retained-vs-evicted lifecycle
- later policies can extend this instead of inventing eviction from scratch
- the project has a clearer separation between active use, retention, and discard

### New Observability
Added two stats to the prefix-cache printout:
- `eviction_passes`
- `evicted_leaves`

This makes retention behavior visible in experiments.

### Expected Validation
The shared-prefix regression should still preserve the old hit behavior in the common case.
For the small synthetic regression, eviction counters are expected to stay at or near zero because the test is not pressure-heavy.

### What Was Learned From Phase 6
1. Returning to Stage A was a deliberate engineering choice, not a retreat from the SGLang goal.
2. Before attacking finer-grained reuse, the block-level tree needs explicit cache-lifecycle semantics.
3. A minimal leaf-eviction policy can be added without disturbing the existing scheduler or block-level reuse behavior.

### Recommended Next Step
Continue Stage A only far enough to make the cache manager trustworthy:
- verify retention / eviction behavior under stronger pressure workloads
- decide whether node pin metadata is needed explicitly
- then move to the larger Stage B/C work: decoupling logical reuse granularity from physical block storage

## Phase 7 - Begin Stage B With Logical Page Metadata

### Route Choice
After stabilizing the block-level cache manager in Stage A, the project now begins **Stage B**.
This transition is intentionally conservative.
The immediate goal is not yet to make partial blocks reusable.
Instead, the first Stage B step is to introduce an explicit **logical page** axis so the system can start distinguishing:
- physical KV storage block size
- logical prefix reuse granularity

This preserves the current execution path while preparing the codebase for later finer-grained prefix reuse.

### Objective
Add the first minimal Stage B skeleton:
- a configurable `logical_page_size`
- `Sequence` helpers for logical-page views
- prefill-plan metadata that reports reuse in logical pages as well as tokens/blocks
- observability that shows logical-page reuse without changing current runtime semantics

### Files Changed
- `nanovllm/config.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/llm_engine.py`

### Main Design Change
Introduced a new configuration field:
- `logical_page_size`, default `32`

Validation rules:
- it must be positive
- it must evenly divide `kvcache_block_size`

This means the system now has two distinct granularities:
- physical block size: still the KV/cache allocation unit
- logical page size: a smaller semantic unit for future prefix matching and accounting

### Sequence-Level Changes
`Sequence` now exposes logical-page metadata and slicing helpers:
- `logical_page_size`
- `num_logical_pages`
- `num_cached_logical_pages`
- `last_logical_page_num_tokens`
- `logical_page(i)`

This is the first place where the project can talk about a request in page units without yet changing the actual KV layout.

### Scheduler / Runner Alignment
Both `Scheduler` and `ModelRunner` now push the config values back into `Sequence` class-level settings so the whole runtime sees a consistent:
- `block_size`
- `logical_page_size`

This avoids hard-coding `256`/`32` assumptions in scattered places.

### Prefill Plan Extension
`PrefillPlan` now carries both block-oriented and logical-page-oriented metadata:
- `cached_tokens`
- `cached_logical_pages`
- `required_free_blocks`
- `required_logical_pages`

Current scheduling behavior is still unchanged.
The scheduler continues to admit work based on tokens and free blocks.
However, the planning path now computes page-level accounting that later stages can consume.

### Observability
Prefix-cache stats now additionally report:
- `reused_logical_pages`
- `logical_page_size`

This makes Stage B progress visible even before the runtime begins reusing partial blocks.

### Why This Matters
This phase still does **not** make partial blocks reusable.
The actual reuse semantics are still block-level.
But the system now has a first-class place to represent a smaller logical reuse unit.

That is the critical first move for Stage B, because later changes will need to answer:
- how many logical pages were reused
- how many logical pages remain uncached
- how a request maps from logical units to physical KV storage

Without this intermediate layer, later addressing work would have to be introduced all at once.

### Expected Validation
Existing regression tests should continue to pass.
The only visible difference should be extra prefix-cache fields in the output, showing logical-page accounting derived from the old block-level hits.

### Recommended Next Step
The next Stage B step should stay incremental:
- add a request-side logical-page mapping structure
- then let the planning path reason about page-aligned cached spans more explicitly
- only after that begin touching the actual KV addressing path in `ModelRunner`

## Phase 8 - Add Request-Side Logical Page Mapping Skeleton

### Objective
Continue Stage B without touching the actual KV addressing path yet.
The goal of this step is to make logical pages first-class on the **request side**, so later work can reason about page-level cached spans before the runtime starts storing or addressing KV at page granularity.

### Files Changed
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/block_manager.py`

### Main Design Change
Added a request-side logical-page mapping structure.
`Sequence` now owns `logical_page_table`, where each entry records:
- `block_id`
- `block_offset`
- `page_tokens`
- `cached`

This means a request can now explicitly describe how each logical page relates to the current block-level KV layout, even though the physical storage path is still block-based.

### Sequence-Level Additions
Introduced `LogicalPageRef` and new `Sequence` helpers:
- `logical_page_table`
- `logical_pages_per_block`
- `clear_logical_page_table()`
- `sync_logical_page_table()`

`sync_logical_page_table()` derives the page mapping from the current:
- `block_table`
- `num_tokens`
- `num_cached_tokens`
- `logical_page_size`

This gives later stages a stable request-local source of truth for page layout.

### Prefill Plan Extension
`PrefillPlan` now also carries page-level planning metadata:
- `page_block_ids`
- `cached_page_mask`

At planning time:
- cached full-block hits are expanded into logical-page entries that point to the reused `block_id`
- uncached spans are represented with `-1` block ids and `False` in the page mask

This is still derived from the current block-level semantics, but it makes the cached/uncached page boundary explicit.

### Lifecycle Integration
The page table is now synchronized when:
- prefill allocation finishes
- decode append mutates the block layout
- deallocation clears the request state

So the request-side mapping stays aligned with the current block-level runtime state.

### Why This Matters
The previous Stage B step only introduced logical-page counting.
This step goes one level deeper: the system can now represent a request's page layout, not just count pages.

That is the missing bridge between:
- page-level planning semantics
- and future page-level or token-level KV addressing

Without a request-side mapping table, later work would have to jump directly from counters to backend layout changes.

### Expected Validation
Current block-level behavior should remain unchanged.
Regression output may not change much beyond preserving the Stage B page statistics, because this step mainly adds internal structure for later use.

### Recommended Next Step
The next Stage B step should connect this request-side mapping to a more explicit planning abstraction:
- reason about contiguous cached page spans instead of only block hits
- then start threading page-layout metadata into `ModelRunner.prepare_prefill()` as read-only structure before changing the actual KV storage layout

## Phase 9 - Add Page Span Planning And Read-Only Runner Consumption

### Objective
Continue Stage B by promoting page metadata from flat per-page tables to a span-oriented planning abstraction, then let `ModelRunner.prepare_prefill()` consume that metadata in a read-only way before any actual KV-addressing refactor.

### Files Changed
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/model_runner.py`

### Main Design Change
Introduced explicit logical-page spans.
The system can now describe contiguous cached and uncached page regions instead of only storing a per-page mask.

This is important because later KV-addressing work will almost certainly reason about reused prefix **ranges**, not isolated page entries.

### Sequence-Level Additions
Added `LogicalPageSpan` plus `Sequence.logical_page_spans()`.
This groups `logical_page_table` into contiguous spans, each carrying:
- `start_page`
- `end_page`
- `start_token`
- `end_token`
- `cached`

So the request-side page layout now has both:
- a fine-grained table view
- a higher-level span view

### Prefill Plan Extension
`PrefillPlan` now carries:
- `cached_page_spans`
- `uncached_page_spans`

These are built directly from `cached_page_mask` during planning.
That means the planning path can now explicitly describe:
- the reused page prefix region
- the uncached page tail region

without changing the underlying block-level allocation path.

### ModelRunner Integration
`ModelRunner.prepare_prefill()` now reads the request-side page metadata through `prepare_logical_page_metadata()`.
This step is intentionally read-only.
It currently uses spans to:
- validate that cached/uncached page spans agree with `num_cached_tokens`
- derive the uncached start token from the uncached span boundary
- keep the existing slot-mapping behavior unchanged

So the runner has started consuming page-layout information, but it still writes KV using the old block-level layout.

### Why This Matters
This phase is the first real bridge between Stage B planning and the runner.
Before this, page metadata existed only on the request/planning side.
Now the execution-preparation path can see that metadata and verify it is self-consistent.

That reduces the gap to the next step, where page-layout metadata can begin to influence how prefill addressing is prepared.

### Expected Validation
Regression behavior should remain unchanged.
The point of this phase is not to increase reuse yet, but to make page-span structure available all the way up to `prepare_prefill()`.

### Recommended Next Step
The next Stage B step should begin a controlled influence of page metadata on addressing preparation:
- first thread page-span metadata deeper into prefill slot-planning
- then experiment with page-aligned cached-boundary handling before any physical KV page allocator change
## Phase 10 - Complete Stage B Boundary 2 With Page-Aware Prefill Preparation

### Route Choice
Stage B is now explicitly completed at the **Page-Aware Prefill Preparation** boundary.
That means:
- physical KV storage remains block-based
- attention/context interfaces remain block-table and slot-mapping based
- but page/span metadata now materially influences how prefill preparation is derived

This deliberately stops short of true partial-block reuse.
That next bridge belongs to the following addressing/storage phase.

### Files Changed
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/llm_engine.py`
- `run_page_aware_prefill_checks.sh`

### Main Design Change
Added a request-side `prefill_layout` object on `Sequence` as the single source of truth for prefill page boundaries.
It now stores:
- `page_block_ids`
- `cached_page_mask`
- `cached_page_spans`
- `uncached_page_spans`
- `uncached_start_token`
- `uncached_end_token`
- `uncached_start_page`
- `uncached_num_pages`
- `uncached_num_tokens`

`BlockManager.allocate()` now writes this layout back to the request before the request enters the running queue.
`deallocate()` and decode append paths clear it so stale prefill metadata does not persist.

### Prefill Planning Changes
`PrefillPlan` now carries the explicit uncached token/page boundary fields used by the scheduler and runner:
- `uncached_start_token`
- `uncached_num_tokens`
- `uncached_start_page`
- `uncached_num_pages`

The scheduler now uses `uncached_num_tokens` directly instead of recomputing the uncached cost from `cached_tokens`.

### Runner Changes
`ModelRunner.prepare_prefill()` now consumes the request-side prefill layout as its primary boundary source.
It no longer derives the uncached start from `num_cached_tokens` except as a consistency assertion.

Slot mapping is now generated page-by-page from:
- request `logical_page_table`
- `prefill_layout.uncached_*` boundaries
- `block_id + block_offset + page_tokens`

The physical result is still equivalent to the old block-range logic, and the runner asserts that equivalence during Stage B.

### New Validation
Added `run_page_aware_prefill_checks.sh`.
This is a lightweight, non-model validation script that checks:
- aligned block-boundary shared prefixes
- non-aligned shared prefixes that still only reuse whole blocks
- no-cache single request
- full-hit complete-block reuse
- lifecycle consistency for `prefill_layout` and `logical_page_table`
- equality between legacy and page-aware prefill slot mapping

### Observability
Added lightweight per-run counters:
- `prefill_cached_pages`
- `prefill_uncached_pages`

These are printed alongside the existing prefix-cache stats.

### Why This Closes Stage B Boundary 2
At this point page/span metadata is no longer passive.
It affects prefill preparation in the runner, while the physical KV storage path still remains block-based.
That is exactly the intended Stage B endpoint.

### Recommended Next Step
The next phase should target the true logic-to-storage bridge:
- relax the assumption that cached boundaries align to full blocks
- thread page-aware addressing deeper than a read-only equivalence layer
- then decide whether the allocator/context/backend must grow a physical page abstraction

