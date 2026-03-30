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
    cached: bool


@dataclass
class LogicalPageSpan:
    start_page: int
    end_page: int
    start_token: int
    end_token: int
    cached: bool


@dataclass
class RequestPrefillLayout:
    page_block_ids: list[int]
    cached_page_mask: list[bool]
    cached_page_spans: list[LogicalPageSpan]
    uncached_page_spans: list[LogicalPageSpan]
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
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

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
            cached = page_start + page_tokens <= cached_tokens
            logical_page_table.append(LogicalPageRef(block_id, block_offset, page_tokens, cached))
        self.logical_page_table = logical_page_table

    def logical_page_spans(self, cached: bool | None = None) -> list[LogicalPageSpan]:
        if not self.logical_page_table:
            return []
        spans = []
        start_page = 0
        current_cached = self.logical_page_table[0].cached
        for page_idx, page_ref in enumerate(self.logical_page_table[1:], start=1):
            if page_ref.cached != current_cached:
                spans.append(LogicalPageSpan(
                    start_page,
                    page_idx,
                    start_page * self.logical_page_size,
                    min(page_idx * self.logical_page_size, self.num_tokens),
                    current_cached,
                ))
                start_page = page_idx
                current_cached = page_ref.cached
        spans.append(LogicalPageSpan(
            start_page,
            len(self.logical_page_table),
            start_page * self.logical_page_size,
            self.num_tokens,
            current_cached,
        ))
        if cached is None:
            return spans
        return [span for span in spans if span.cached is cached]

    def sync_prefill_layout(self):
        if not self.block_table:
            self.prefill_layout = None
            return
        assert self.num_cached_tokens % self.logical_page_size == 0
        cached_page_mask = [page_ref.cached for page_ref in self.logical_page_table]
        cached_page_spans = self.logical_page_spans(cached=True)
        uncached_page_spans = self.logical_page_spans(cached=False)
        if cached_page_spans:
            assert len(cached_page_spans) == 1
            assert cached_page_spans[0].start_page == 0
            assert cached_page_spans[0].end_token == self.num_cached_tokens
        uncached_start_page = uncached_page_spans[0].start_page if uncached_page_spans else self.num_logical_pages
        uncached_start_token = uncached_page_spans[0].start_token if uncached_page_spans else self.num_tokens
        if cached_page_spans and uncached_page_spans:
            assert cached_page_spans[0].end_page == uncached_start_page
        uncached_end_token = self.num_tokens
        uncached_num_pages = sum(span.end_page - span.start_page for span in uncached_page_spans)
        uncached_num_tokens = uncached_end_token - uncached_start_token
        assert uncached_start_token == self.num_cached_tokens
        self.prefill_layout = RequestPrefillLayout(
            page_block_ids=[page_ref.block_id for page_ref in self.logical_page_table],
            cached_page_mask=cached_page_mask,
            cached_page_spans=cached_page_spans,
            uncached_page_spans=uncached_page_spans,
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
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        self.logical_page_table = []
        self.prefill_layout = None
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
            if self.block_table:
                self.sync_logical_page_table()
                self.sync_prefill_layout()
        else:
            self.last_token = state[-1]
