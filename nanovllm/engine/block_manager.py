from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import xxhash
import numpy as np

from nanovllm.engine.sequence import LogicalPageSpan, PhysicalCopySpan, Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class KVBlockAllocator:

    def __init__(self, num_blocks: int):
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def has_free_blocks(self, num_blocks: int) -> bool:
        return self.num_free_blocks() >= num_blocks

    def is_used(self, block_id: int) -> bool:
        return block_id in self.used_block_ids

    def allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def allocate_next_free_block(self) -> tuple[int, Block]:
        block_id = self.free_block_ids[0]
        return block_id, self.allocate_block(block_id)

    def incref(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count > 0
        block.ref_count += 1
        return block

    def decref(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)


@dataclass
class PrefixTreeNode:
    parent: "PrefixTreeNode | None" = None
    key_from_parent: tuple[int, ...] = ()
    depth: int = 0
    block_id: int = -1
    block_hash: int = -1
    token_ids: tuple[int, ...] = ()
    touch_count: int = 0
    last_access_tick: int = 0
    children: dict[tuple[int, ...], "PrefixTreeNode"] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class PrefixCache:

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = PrefixTreeNode()
        self.hash_to_block_id: dict[int, int] = dict()
        self.block_to_node: dict[int, PrefixTreeNode] = dict()
        self.access_tick = 0

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def compute_block_hash(self, token_ids: list[int], prefix: int = -1) -> int:
        return self.compute_hash(token_ids, prefix) if len(token_ids) == self.block_size else -1

    def _touch(self, node: PrefixTreeNode):
        self.access_tick += 1
        node.last_access_tick = self.access_tick
        node.touch_count += 1

    def get_child(self, parent: PrefixTreeNode, token_ids: list[int]) -> PrefixTreeNode | None:
        if len(token_ids) != self.block_size:
            return None
        node = parent.children.get(tuple(token_ids))
        if node is not None:
            self._touch(node)
        return node

    def _prune_subtree(self, node: PrefixTreeNode):
        for child in list(node.children.values()):
            self._prune_subtree(child)
        if node.block_hash != -1 and self.hash_to_block_id.get(node.block_hash) == node.block_id:
            del self.hash_to_block_id[node.block_hash]
        if node.block_id != -1 and self.block_to_node.get(node.block_id) is node:
            del self.block_to_node[node.block_id]
        node.children.clear()
        node.block_id = -1
        node.block_hash = -1
        node.token_ids = ()
        node.touch_count = 0
        node.last_access_tick = 0
        if node.parent is not None:
            node.parent.children.pop(node.key_from_parent, None)

    def _ensure_node(self, parent: PrefixTreeNode, token_ids: list[int]) -> PrefixTreeNode:
        key = tuple(token_ids)
        node = parent.children.get(key)
        if node is None:
            node = PrefixTreeNode(parent=parent, key_from_parent=key, depth=parent.depth + 1)
            parent.children[key] = node
        return node

    def _iter_leaves(self, node: PrefixTreeNode | None = None):
        node = self.root if node is None else node
        if node.is_leaf:
            if node is not self.root:
                yield node
            return
        for child in node.children.values():
            yield from self._iter_leaves(child)

    def collect_evictable_leaves(
        self,
        allocator: KVBlockAllocator,
        protected_block_ids: set[int] | None = None,
    ) -> list[PrefixTreeNode]:
        protected_block_ids = set() if protected_block_ids is None else protected_block_ids
        leaves = []
        for leaf in self._iter_leaves():
            if leaf.block_id == -1:
                continue
            if leaf.block_id in protected_block_ids:
                continue
            if allocator.is_used(leaf.block_id):
                continue
            leaves.append(leaf)
        return leaves

    def select_eviction_candidate(
        self,
        allocator: KVBlockAllocator,
        protected_block_ids: set[int] | None = None,
    ) -> PrefixTreeNode | None:
        candidates = self.collect_evictable_leaves(allocator, protected_block_ids)
        if not candidates:
            return None
        return min(candidates, key=lambda node: (node.last_access_tick, node.touch_count, -node.depth, node.block_id))

    def evict_one_leaf(
        self,
        allocator: KVBlockAllocator,
        protected_block_ids: set[int] | None = None,
    ) -> int | None:
        candidate = self.select_eviction_candidate(allocator, protected_block_ids)
        if candidate is None:
            return None
        block_id = candidate.block_id
        self._prune_subtree(candidate)
        return block_id

    def commit(self, parent: PrefixTreeNode, block: Block, block_hash: int, token_ids: list[int]) -> PrefixTreeNode:
        if block_hash == -1:
            return self.root
        node = self._ensure_node(parent, token_ids)
        old_owner = self.block_to_node.get(block.block_id)
        if old_owner is not None and old_owner is not node:
            self._prune_subtree(old_owner)
            node = self._ensure_node(parent, token_ids)
        node.block_id = block.block_id
        node.block_hash = block_hash
        node.token_ids = tuple(token_ids)
        self._touch(node)
        block.update(block_hash, token_ids)
        self.hash_to_block_id[block_hash] = block.block_id
        self.block_to_node[block.block_id] = node
        return node


class PlanStepKind(Enum):
    HIT_USED = auto()
    HIT_FREE = auto()
    PARTIAL_HIT_FREE = auto()
    PARTIAL_HIT_USED_COPY = auto()
    MISS = auto()


@dataclass
class PlanStep:
    kind: PlanStepKind
    block_id: int
    block_hash: int
    token_ids: list[int]
    shared_prefix_tokens: int = 0
    prefix_hash_before_block: int = -1


@dataclass
class PrefillPlan:
    steps: list[PlanStep]
    cached_tokens: int
    cached_logical_pages: int
    required_free_blocks: int
    required_logical_pages: int
    page_block_ids: list[int]
    cached_page_mask: list[bool]
    page_cached_tokens: list[int]
    cached_page_spans: list[LogicalPageSpan]
    uncached_page_spans: list[LogicalPageSpan]
    uncached_start_token: int
    uncached_num_tokens: int
    uncached_start_page: int
    uncached_num_pages: int


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, logical_page_size: int | None = None):
        self.block_size = block_size
        self.logical_page_size = block_size if logical_page_size is None else logical_page_size
        assert self.block_size % self.logical_page_size == 0
        self.allocator = KVBlockAllocator(num_blocks)
        self.prefix_cache = PrefixCache(block_size)
        self.blocks = self.allocator.blocks
        self.free_block_ids = self.allocator.free_block_ids
        self.used_block_ids = self.allocator.used_block_ids
        self.hash_to_block_id = self.prefix_cache.hash_to_block_id
        self.retention_low_watermark = max(1, num_blocks // 8)
        self.partial_prefix_to_block_ids: dict[int, set[int]] = dict()
        self.partial_block_to_prefix: dict[int, int] = dict()
        self.stats = {
            "alloc_requests": 0,
            "dealloc_requests": 0,
            "queried_blocks": 0,
            "hit_blocks": 0,
            "miss_blocks": 0,
            "reused_tokens": 0,
            "partial_reused_tokens": 0,
            "reused_logical_pages": 0,
            "prefill_cached_pages": 0,
            "prefill_uncached_pages": 0,
            "new_blocks": 0,
            "eviction_passes": 0,
            "evicted_leaves": 0,
        }

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self):
        stats = dict(self.stats)
        queried_blocks = stats["queried_blocks"]
        stats["hit_rate"] = stats["hit_blocks"] / queried_blocks if queried_blocks else 0.0
        stats["logical_page_size"] = self.logical_page_size
        return stats

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        return PrefixCache.compute_hash(token_ids, prefix)

    @staticmethod
    def common_prefix_len(lhs: list[int], rhs: list[int]) -> int:
        matched = 0
        max_len = min(len(lhs), len(rhs))
        while matched < max_len and lhs[matched] == rhs[matched]:
            matched += 1
        return matched

    def unregister_partial_block(self, block_id: int):
        prefix_hash = self.partial_block_to_prefix.pop(block_id, None)
        if prefix_hash is None:
            return
        block_ids = self.partial_prefix_to_block_ids.get(prefix_hash)
        if block_ids is None:
            return
        block_ids.discard(block_id)
        if not block_ids:
            del self.partial_prefix_to_block_ids[prefix_hash]

    def register_partial_block(self, block_id: int, prefix_hash_before_block: int, token_ids: list[int]):
        self.unregister_partial_block(block_id)
        if not token_ids or len(token_ids) == self.block_size:
            return
        self.blocks[block_id].update(-1, token_ids)
        self.partial_prefix_to_block_ids.setdefault(prefix_hash_before_block, set()).add(block_id)
        self.partial_block_to_prefix[block_id] = prefix_hash_before_block

    def get_materialized_partial_info(self, seq: Sequence) -> tuple[int, int, list[int]] | None:
        if not seq.block_table or seq.num_materialized_tokens == 0:
            return None
        if seq.last_materialized_block_num_tokens in (0, self.block_size):
            return None
        block_idx = seq.num_materialized_blocks - 1
        block_id = seq.block_table[block_idx]
        prefix_hash = -1
        for i in range(block_idx):
            prefix_hash = self.prefix_cache.compute_block_hash(seq.materialized_block(i), prefix_hash)
        return block_id, prefix_hash, seq.materialized_block(block_idx)

    def clear_materialized_partial_block(self, seq: Sequence):
        info = self.get_materialized_partial_info(seq)
        if info is None:
            return
        block_id, _, _ = info
        self.unregister_partial_block(block_id)

    def sync_materialized_partial_block(self, seq: Sequence):
        info = self.get_materialized_partial_info(seq)
        if info is None:
            return
        block_id, prefix_hash_before_block, token_ids = info
        self.register_partial_block(block_id, prefix_hash_before_block, token_ids)

    def find_best_partial_hit(self, prefix_hash_before_block: int, token_ids: list[int]) -> tuple[int, int, bool]:
        best_block_id = -1
        best_shared_prefix_tokens = 0
        best_is_used = False
        for block_id in self.partial_prefix_to_block_ids.get(prefix_hash_before_block, set()):
            shared_prefix_tokens = self.common_prefix_len(self.blocks[block_id].token_ids, token_ids)
            is_used = self.allocator.is_used(block_id)
            if (
                shared_prefix_tokens > best_shared_prefix_tokens
                or (
                    shared_prefix_tokens == best_shared_prefix_tokens
                    and shared_prefix_tokens > 0
                    and best_block_id != -1
                    and best_is_used
                    and not is_used
                )
            ):
                best_block_id = block_id
                best_shared_prefix_tokens = shared_prefix_tokens
                best_is_used = is_used
        return best_block_id, best_shared_prefix_tokens, best_is_used

    def build_page_spans(self, cached_tokens: int, num_tokens: int, num_logical_pages: int) -> tuple[list[LogicalPageSpan], list[LogicalPageSpan]]:
        spans = []
        if cached_tokens > 0:
            spans.append(LogicalPageSpan(
                0,
                (cached_tokens + self.logical_page_size - 1) // self.logical_page_size,
                0,
                cached_tokens,
                True,
            ))
        if cached_tokens < num_tokens:
            spans.append(LogicalPageSpan(
                cached_tokens // self.logical_page_size,
                num_logical_pages,
                cached_tokens,
                num_tokens,
                False,
            ))
        cached_spans = [span for span in spans if span.cached]
        uncached_spans = [span for span in spans if not span.cached]
        return cached_spans, uncached_spans

    def make_prefill_plan(self, seq: Sequence) -> PrefillPlan:
        steps = []
        prefix_hash = -1
        prefix_node = self.prefix_cache.root
        cache_miss = False
        cached_tokens = 0
        required_free_blocks = 0
        page_block_ids = []
        cached_page_mask = []
        page_cached_tokens = []

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            prefix_hash_before_block = prefix_hash
            block_hash = self.prefix_cache.compute_block_hash(token_ids, prefix_hash)
            node = None if cache_miss else self.prefix_cache.get_child(prefix_node, token_ids)
            block_id = -1 if node is None else node.block_id
            is_hit = block_id != -1 and self.blocks[block_id].token_ids == token_ids
            num_pages = (len(token_ids) + self.logical_page_size - 1) // self.logical_page_size
            if is_hit:
                cached_tokens += self.block_size
                if self.allocator.is_used(block_id):
                    steps.append(PlanStep(PlanStepKind.HIT_USED, block_id, block_hash, token_ids, len(token_ids), prefix_hash_before_block))
                else:
                    required_free_blocks += 1
                    steps.append(PlanStep(PlanStepKind.HIT_FREE, block_id, block_hash, token_ids, len(token_ids), prefix_hash_before_block))
                page_block_ids.extend([block_id] * num_pages)
                cached_page_mask.extend([True] * num_pages)
                page_cached_tokens.extend([
                    min(self.logical_page_size, len(token_ids) - page_idx * self.logical_page_size)
                    for page_idx in range(num_pages)
                ])
                prefix_node = node
            else:
                partial_block_id, shared_prefix_tokens, partial_block_is_used = (-1, 0, False)
                if not cache_miss and len(token_ids) < self.block_size:
                    partial_block_id, shared_prefix_tokens, partial_block_is_used = self.find_best_partial_hit(
                        prefix_hash_before_block,
                        token_ids,
                    )
                if partial_block_id != -1 and shared_prefix_tokens > 0:
                    cache_miss = True
                    cached_tokens += shared_prefix_tokens
                    required_free_blocks += 1
                    steps.append(PlanStep(
                        PlanStepKind.PARTIAL_HIT_USED_COPY if partial_block_is_used else PlanStepKind.PARTIAL_HIT_FREE,
                        partial_block_id,
                        -1,
                        token_ids,
                        shared_prefix_tokens,
                        prefix_hash_before_block,
                    ))
                    page_block_ids.extend([partial_block_id] * num_pages)
                    for page_idx in range(num_pages):
                        page_start = page_idx * self.logical_page_size
                        page_tokens = min(self.logical_page_size, len(token_ids) - page_start)
                        cached_in_page = min(max(shared_prefix_tokens - page_start, 0), page_tokens)
                        page_cached_tokens.append(cached_in_page)
                        cached_page_mask.append(cached_in_page == page_tokens)
                else:
                    cache_miss = True
                    required_free_blocks += 1
                    steps.append(PlanStep(PlanStepKind.MISS, -1, block_hash, token_ids, 0, prefix_hash_before_block))
                    page_block_ids.extend([-1] * num_pages)
                    cached_page_mask.extend([False] * num_pages)
                    page_cached_tokens.extend([0] * num_pages)
            prefix_hash = block_hash

        uncached_tokens = len(seq) - cached_tokens
        cached_logical_pages = sum(
            cached_in_page == self.logical_page_size
            for cached_in_page in page_cached_tokens
        )
        required_logical_pages = seq.num_logical_pages - (cached_tokens // self.logical_page_size)
        assert len(page_block_ids) == seq.num_logical_pages
        assert len(cached_page_mask) == seq.num_logical_pages
        assert len(page_cached_tokens) == seq.num_logical_pages
        cached_page_spans, uncached_page_spans = self.build_page_spans(cached_tokens, len(seq), seq.num_logical_pages)
        uncached_start_token = cached_tokens
        uncached_start_page = cached_tokens // self.logical_page_size
        uncached_num_tokens = len(seq) - uncached_start_token
        uncached_num_pages = seq.num_logical_pages - uncached_start_page
        assert uncached_start_token == cached_tokens
        assert uncached_num_pages == required_logical_pages
        return PrefillPlan(
            steps,
            cached_tokens,
            cached_logical_pages,
            required_free_blocks,
            required_logical_pages,
            page_block_ids,
            cached_page_mask,
            page_cached_tokens,
            cached_page_spans,
            uncached_page_spans,
            uncached_start_token,
            uncached_num_tokens,
            uncached_start_page,
            uncached_num_pages,
        )

    def can_allocate(self, seq: Sequence, plan: PrefillPlan | None = None) -> bool:
        if plan is None:
            plan = self.make_prefill_plan(seq)
        return self.allocator.has_free_blocks(plan.required_free_blocks)

    def apply_retention_policy(
        self,
        required_free_blocks: int,
        protected_block_ids: set[int] | None = None,
    ) -> int:
        projected_free_blocks = self.allocator.num_free_blocks() - required_free_blocks
        if projected_free_blocks > self.retention_low_watermark:
            return 0
        budget = max(1, min(required_free_blocks or 1, self.retention_low_watermark - projected_free_blocks + 1))
        protected_block_ids = set() if protected_block_ids is None else protected_block_ids
        self.stats["eviction_passes"] += 1
        evicted = 0
        for _ in range(budget):
            block_id = self.prefix_cache.evict_one_leaf(self.allocator, protected_block_ids)
            if block_id is None:
                break
            evicted += 1
        self.stats["evicted_leaves"] += evicted
        return evicted

    def allocate(self, seq: Sequence, plan: PrefillPlan | None = None):
        assert not seq.block_table
        if plan is None:
            plan = self.make_prefill_plan(seq)

        self.stats["alloc_requests"] += 1
        self.stats["prefill_cached_pages"] += plan.cached_logical_pages
        self.stats["prefill_uncached_pages"] += plan.uncached_num_pages
        protected_block_ids = {step.block_id for step in plan.steps if step.block_id != -1}
        self.apply_retention_policy(plan.required_free_blocks, protected_block_ids)
        seq.clear_prefill_layout()
        seq.num_cached_tokens = plan.cached_tokens
        prefix_node = self.prefix_cache.root
        copy_spans = []

        for step in plan.steps:
            self.stats["queried_blocks"] += 1
            token_ids = step.token_ids
            if step.kind == PlanStepKind.MISS:
                self.stats["miss_blocks"] += 1
                block_id, block = self.allocator.allocate_next_free_block()
                self.unregister_partial_block(block_id)
                self.stats["new_blocks"] += 1
            elif step.kind == PlanStepKind.HIT_USED:
                self.stats["hit_blocks"] += 1
                self.stats["reused_tokens"] += step.shared_prefix_tokens
                self.stats["reused_logical_pages"] += step.shared_prefix_tokens // self.logical_page_size
                block_id = step.block_id
                block = self.allocator.incref(block_id)
            elif step.kind == PlanStepKind.HIT_FREE:
                self.stats["hit_blocks"] += 1
                self.stats["reused_tokens"] += step.shared_prefix_tokens
                self.stats["reused_logical_pages"] += step.shared_prefix_tokens // self.logical_page_size
                block_id = step.block_id
                block = self.allocator.allocate_block(block_id)
            elif step.kind == PlanStepKind.PARTIAL_HIT_FREE:
                block_id = step.block_id
                self.unregister_partial_block(block_id)
                block = self.allocator.allocate_block(block_id)
                self.stats["reused_tokens"] += step.shared_prefix_tokens
                self.stats["partial_reused_tokens"] += step.shared_prefix_tokens
                self.stats["reused_logical_pages"] += step.shared_prefix_tokens // self.logical_page_size
            else:
                src_block_id = step.block_id
                block_start_token = len(seq.block_table) * self.block_size
                block_id, block = self.allocator.allocate_next_free_block()
                self.unregister_partial_block(block_id)
                self.stats["new_blocks"] += 1
                self.stats["reused_tokens"] += step.shared_prefix_tokens
                self.stats["partial_reused_tokens"] += step.shared_prefix_tokens
                self.stats["reused_logical_pages"] += step.shared_prefix_tokens // self.logical_page_size
                copy_spans.append(PhysicalCopySpan(
                    src_block_id=src_block_id,
                    dst_block_id=block_id,
                    src_block_offset=0,
                    dst_block_offset=0,
                    num_tokens=step.shared_prefix_tokens,
                    start_token=block_start_token,
                    end_token=block_start_token + step.shared_prefix_tokens,
                ))

            if step.block_hash != -1:
                prefix_node = self.prefix_cache.commit(prefix_node, block, step.block_hash, token_ids)
            else:
                block.update(-1, token_ids)
            seq.block_table.append(block_id)
        seq.sync_logical_page_table()
        seq.sync_prefill_layout(copy_spans=copy_spans)
        layout = seq.prefill_layout
        assert layout is not None
        assert layout.cached_page_mask == plan.cached_page_mask
        assert layout.page_cached_tokens == plan.page_cached_tokens
        assert layout.cached_page_spans == plan.cached_page_spans
        assert layout.uncached_page_spans == plan.uncached_page_spans
        assert layout.uncached_start_token == plan.uncached_start_token
        assert layout.uncached_end_token == len(seq)
        assert layout.uncached_start_page == plan.uncached_start_page
        assert layout.uncached_num_pages == plan.uncached_num_pages
        assert layout.uncached_num_tokens == plan.uncached_num_tokens

    def deallocate(self, seq: Sequence):
        self.stats["dealloc_requests"] += 1
        self.clear_materialized_partial_block(seq)
        self.sync_materialized_partial_block(seq)
        for block_id in reversed(seq.block_table):
            self.allocator.decref(block_id)
        seq.num_materialized_tokens = 0
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.clear_logical_page_table()
        seq.clear_prefill_layout()

    def can_append(self, seq: Sequence) -> bool:
        return self.allocator.has_free_blocks(len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        self.clear_materialized_partial_block(seq)
        seq.clear_prefill_layout()
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            self.apply_retention_policy(1, set(block_table))
            block_id, _ = self.allocator.allocate_next_free_block()
            self.unregister_partial_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            block_hash = self.prefix_cache.compute_block_hash(token_ids, prefix)
            prefix_node = self.prefix_cache.root
            for block_id in block_table[:-1]:
                block = self.blocks[block_id]
                prefix_node = self.prefix_cache.get_child(prefix_node, block.token_ids)
                assert prefix_node is not None
            self.prefix_cache.commit(prefix_node, last_block, block_hash, token_ids)
        else:
            assert last_block.hash == -1
        seq.sync_logical_page_table()
