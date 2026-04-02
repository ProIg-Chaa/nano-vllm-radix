from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class LogicalPageRef:
    block_id: int
    block_offset: int
    page_tokens: int
    cached_tokens: int

    @property
    def cached(self) -> bool:
        return self.cached_tokens == self.page_tokens

    @property
    def uncached_tokens(self) -> int:
        return self.page_tokens - self.cached_tokens

    @property
    def is_partial(self) -> bool:
        return 0 < self.cached_tokens < self.page_tokens


@dataclass
class LogicalPageSpan:
    start_page: int
    end_page: int
    start_token: int
    end_token: int
    cached: bool


@dataclass
class PhysicalAddressSpan:
    start_page: int
    end_page: int
    start_token: int
    end_token: int
    slot_start: int
    slot_end: int
    cached: bool


@dataclass
class PhysicalCopySpan:
    src_block_id: int
    dst_block_id: int
    src_block_offset: int
    dst_block_offset: int
    num_tokens: int
    start_token: int
    end_token: int


@dataclass
class RequestPrefillLayout:
    page_block_ids: list[int]
    cached_page_mask: list[bool]
    page_cached_tokens: list[int]
    cached_page_spans: list[LogicalPageSpan]
    uncached_page_spans: list[LogicalPageSpan]
    cached_physical_spans: list[PhysicalAddressSpan]
    uncached_physical_spans: list[PhysicalAddressSpan]
    copy_spans: list[PhysicalCopySpan]
    uncached_start_token: int
    uncached_end_token: int
    uncached_start_page: int
    uncached_num_pages: int
    uncached_num_tokens: int


class Sequence:
    block_size = 256
    logical_page_size = 32
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_materialized_tokens = 0
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.logical_page_table: list[LogicalPageRef] = []
        self.prefill_layout: RequestPrefillLayout | None = None
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_materialized_blocks(self):
        return (self.num_materialized_tokens + self.block_size - 1) // self.block_size if self.num_materialized_tokens else 0

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def last_materialized_block_num_tokens(self):
        if self.num_materialized_tokens == 0:
            return 0
        return self.num_materialized_tokens - (self.num_materialized_blocks - 1) * self.block_size

    @property
    def num_cached_logical_pages(self):
        return self.num_cached_tokens // self.logical_page_size

    @property
    def num_logical_pages(self):
        return (self.num_tokens + self.logical_page_size - 1) // self.logical_page_size

    @property
    def last_logical_page_num_tokens(self):
        return self.num_tokens - (self.num_logical_pages - 1) * self.logical_page_size

    @property
    def logical_pages_per_block(self):
        return self.block_size // self.logical_page_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def materialized_block(self, i):
        assert 0 <= i < self.num_materialized_blocks
        end = min((i + 1) * self.block_size, self.num_materialized_tokens)
        return self.token_ids[i * self.block_size: end]

    def logical_page(self, i):
        assert 0 <= i < self.num_logical_pages
        return self.token_ids[i * self.logical_page_size: (i + 1) * self.logical_page_size]

    def clear_logical_page_table(self):
        self.logical_page_table.clear()

    def clear_prefill_layout(self):
        self.prefill_layout = None

    def set_prefill_layout(self, layout: RequestPrefillLayout):
        self.prefill_layout = layout

    def sync_logical_page_table(self, cached_tokens: int | None = None):
        cached_tokens = self.num_cached_tokens if cached_tokens is None else cached_tokens
        logical_page_table = []
        for page_idx in range(self.num_logical_pages):
            page_start = page_idx * self.logical_page_size
            block_idx = page_start // self.block_size
            block_offset = page_start % self.block_size
            block_id = self.block_table[block_idx] if block_idx < len(self.block_table) else -1
            page_tokens = min(self.logical_page_size, self.num_tokens - page_start)
            page_cached_tokens = min(max(cached_tokens - page_start, 0), page_tokens)
            logical_page_table.append(LogicalPageRef(block_id, block_offset, page_tokens, page_cached_tokens))
        self.logical_page_table = logical_page_table

    def logical_page_spans(self, cached: bool | None = None) -> list[LogicalPageSpan]:
        if self.num_tokens == 0:
            return []
        spans = []
        if self.num_cached_tokens > 0:
            cached_end_page = (self.num_cached_tokens + self.logical_page_size - 1) // self.logical_page_size
            spans.append(LogicalPageSpan(
                0,
                cached_end_page,
                0,
                self.num_cached_tokens,
                True,
            ))
        if self.num_cached_tokens < self.num_tokens:
            uncached_start_page = self.num_cached_tokens // self.logical_page_size
            spans.append(LogicalPageSpan(
                uncached_start_page,
                self.num_logical_pages,
                self.num_cached_tokens,
                self.num_tokens,
                False,
            ))
        if cached is None:
            return spans
        return [span for span in spans if span.cached is cached]

    def physical_address_spans(self, cached: bool | None = None) -> list[PhysicalAddressSpan]:
        if not self.logical_page_table:
            return []
        spans = []
        span_specs = []
        for page_idx, page_ref in enumerate(self.logical_page_table):
            page_start_token = page_idx * self.logical_page_size
            slot_base = page_ref.block_id * self.block_size + page_ref.block_offset
            if page_ref.cached_tokens > 0:
                span_specs.append(PhysicalAddressSpan(
                    page_idx,
                    page_idx + 1,
                    page_start_token,
                    page_start_token + page_ref.cached_tokens,
                    slot_base,
                    slot_base + page_ref.cached_tokens,
                    True,
                ))
            if page_ref.uncached_tokens > 0:
                span_specs.append(PhysicalAddressSpan(
                    page_idx,
                    page_idx + 1,
                    page_start_token + page_ref.cached_tokens,
                    page_start_token + page_ref.page_tokens,
                    slot_base + page_ref.cached_tokens,
                    slot_base + page_ref.page_tokens,
                    False,
                ))
        if not span_specs:
            return []
        current = span_specs[0]
        for spec in span_specs[1:]:
            if (
                spec.cached == current.cached
                and spec.slot_start == current.slot_end
                and spec.start_token == current.end_token
            ):
                current = PhysicalAddressSpan(
                    current.start_page,
                    spec.end_page,
                    current.start_token,
                    spec.end_token,
                    current.slot_start,
                    spec.slot_end,
                    current.cached,
                )
                continue
            spans.append(current)
            current = spec
        spans.append(current)
        if cached is None:
            return spans
        return [span for span in spans if span.cached is cached]

    def sync_prefill_layout(self, copy_spans: list[PhysicalCopySpan] | None = None):
        if not self.block_table:
            self.prefill_layout = None
            return
        cached_page_mask = [page_ref.cached for page_ref in self.logical_page_table]
        cached_page_spans = self.logical_page_spans(cached=True)
        uncached_page_spans = self.logical_page_spans(cached=False)
        cached_physical_spans = self.physical_address_spans(cached=True)
        uncached_physical_spans = self.physical_address_spans(cached=False)
        if cached_page_spans:
            assert len(cached_page_spans) == 1
            assert cached_page_spans[0].start_page == 0
            assert cached_page_spans[0].end_token == self.num_cached_tokens
        uncached_start_page = uncached_page_spans[0].start_page if uncached_page_spans else self.num_logical_pages
        uncached_start_token = uncached_page_spans[0].start_token if uncached_page_spans else self.num_tokens
        uncached_end_token = self.num_tokens
        uncached_num_pages = sum(span.end_page - span.start_page for span in uncached_page_spans)
        uncached_num_tokens = uncached_end_token - uncached_start_token
        assert uncached_start_token == self.num_cached_tokens
        self.prefill_layout = RequestPrefillLayout(
            page_block_ids=[page_ref.block_id for page_ref in self.logical_page_table],
            cached_page_mask=cached_page_mask,
            page_cached_tokens=[page_ref.cached_tokens for page_ref in self.logical_page_table],
            cached_page_spans=cached_page_spans,
            uncached_page_spans=uncached_page_spans,
            cached_physical_spans=cached_physical_spans,
            uncached_physical_spans=uncached_physical_spans,
            copy_spans=[] if copy_spans is None else copy_spans,
            uncached_start_token=uncached_start_token,
            uncached_end_token=uncached_end_token,
            uncached_start_page=uncached_start_page,
            uncached_num_pages=uncached_num_pages,
            uncached_num_tokens=uncached_num_tokens,
        )

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.clear_prefill_layout()
        if self.block_table:
            self.sync_logical_page_table()

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_materialized_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_materialized_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        self.logical_page_table = []
        self.prefill_layout = None
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
            if self.block_table:
                self.sync_logical_page_table()
                self.sync_prefill_layout()
        else:
            self.last_token = state[-1]
