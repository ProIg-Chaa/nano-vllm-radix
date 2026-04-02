#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/experiments"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/page_aware_prefill_checks_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

eval "$(micromamba shell hook -s bash)"
micromamba activate nano_vllm

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

echo "[run] log_file=$LOG_FILE" | tee "$LOG_FILE"
python - <<'INNER_PY' 2>&1 | tee -a "$LOG_FILE"
from nanovllm.engine.block_manager import BlockManager, PlanStepKind
from nanovllm.engine.model_runner import build_legacy_prefill_slot_mapping, build_page_aware_prefill_slot_mapping
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 256
PAGE_SIZE = 32


def make_seq(token_ids):
    Sequence.block_size = BLOCK_SIZE
    Sequence.logical_page_size = PAGE_SIZE
    return Sequence(token_ids, SamplingParams(max_tokens=1, ignore_eos=True))


def validate_allocated_seq(seq, plan):
    layout = seq.prefill_layout
    assert layout is not None
    assert seq.logical_page_table
    assert layout.uncached_start_token == plan.uncached_start_token
    assert layout.uncached_num_tokens == plan.uncached_num_tokens
    assert layout.uncached_start_page == plan.uncached_start_page
    assert layout.uncached_num_pages == plan.uncached_num_pages
    assert layout.cached_page_spans == plan.cached_page_spans
    assert layout.uncached_page_spans == plan.uncached_page_spans
    assert layout.cached_page_mask == plan.cached_page_mask
    assert layout.page_cached_tokens == plan.page_cached_tokens
    assert sum(span.num_tokens for span in layout.copy_spans) <= plan.cached_tokens
    assert sum(layout.page_cached_tokens) == plan.cached_tokens
    assert all(page_cached_tokens in (0, PAGE_SIZE) for page_cached_tokens in layout.page_cached_tokens[:-1])
    assert sum(span.slot_end - span.slot_start for span in layout.cached_physical_spans) == plan.cached_tokens
    assert sum(span.slot_end - span.slot_start for span in layout.uncached_physical_spans) == plan.uncached_num_tokens
    if layout.cached_physical_spans:
        assert layout.cached_physical_spans[0].start_page == 0
        assert layout.cached_physical_spans[-1].end_token == plan.cached_tokens
    if layout.uncached_physical_spans:
        assert layout.uncached_physical_spans[0].start_token == plan.uncached_start_token
        assert layout.uncached_physical_spans[-1].end_token == len(seq)
    page_aware_slot_mapping = build_page_aware_prefill_slot_mapping(seq, BLOCK_SIZE)
    legacy_slot_mapping = build_legacy_prefill_slot_mapping(seq, BLOCK_SIZE)
    assert len(page_aware_slot_mapping) == plan.uncached_num_tokens
    if plan.uncached_start_token % BLOCK_SIZE == 0:
        assert page_aware_slot_mapping == legacy_slot_mapping
    else:
        assert len(page_aware_slot_mapping) < len(legacy_slot_mapping)


def run_case(name, first_prompt, second_prompt=None, expected_cached_tokens=None):
    manager = BlockManager(64, BLOCK_SIZE, PAGE_SIZE)
    seq1 = make_seq(first_prompt)
    plan1 = manager.make_prefill_plan(seq1)
    manager.allocate(seq1, plan1)
    validate_allocated_seq(seq1, plan1)
    seq1.num_materialized_tokens = len(seq1)
    manager.sync_materialized_partial_block(seq1)
    seq1.append_token(999)
    assert seq1.prefill_layout is None
    manager.deallocate(seq1)
    assert not seq1.logical_page_table
    assert seq1.prefill_layout is None

    result = {"case": name, "first_cached_tokens": plan1.cached_tokens}
    if second_prompt is None:
        return result

    seq2 = make_seq(second_prompt)
    plan2 = manager.make_prefill_plan(seq2)
    if expected_cached_tokens is not None:
        assert plan2.cached_tokens == expected_cached_tokens, (name, plan2.cached_tokens, expected_cached_tokens)
    manager.allocate(seq2, plan2)
    validate_allocated_seq(seq2, plan2)
    result.update(
        second_cached_tokens=plan2.cached_tokens,
        uncached_start_token=plan2.uncached_start_token,
        uncached_num_tokens=plan2.uncached_num_tokens,
        uncached_start_page=plan2.uncached_start_page,
        uncached_num_pages=plan2.uncached_num_pages,
    )
    manager.deallocate(seq2)
    assert not seq2.logical_page_table
    assert seq2.prefill_layout is None
    return result


def run_active_partial_copy_case():
    manager = BlockManager(64, BLOCK_SIZE, PAGE_SIZE)
    common_nonaligned = [77] * 600
    seq1 = make_seq(common_nonaligned + [1, 2])
    plan1 = manager.make_prefill_plan(seq1)
    manager.allocate(seq1, plan1)
    seq1.num_materialized_tokens = len(seq1)
    manager.sync_materialized_partial_block(seq1)

    seq2 = make_seq(common_nonaligned + [3, 4])
    plan2 = manager.make_prefill_plan(seq2)
    assert plan2.cached_tokens == 600
    assert any(step.kind == PlanStepKind.PARTIAL_HIT_USED_COPY for step in plan2.steps)
    manager.allocate(seq2, plan2)
    validate_allocated_seq(seq2, plan2)
    assert seq2.prefill_layout is not None
    assert len(seq2.prefill_layout.copy_spans) == 1
    copy_span = seq2.prefill_layout.copy_spans[0]
    assert copy_span.src_block_id == seq1.block_table[-1]
    assert copy_span.dst_block_id == seq2.block_table[-1]
    assert copy_span.num_tokens == 88
    manager.deallocate(seq2)
    manager.deallocate(seq1)
    return {
        "case": "active_partial_tail_copy_on_write",
        "second_cached_tokens": plan2.cached_tokens,
        "copy_span_tokens": copy_span.num_tokens,
        "uncached_start_token": plan2.uncached_start_token,
        "uncached_num_tokens": plan2.uncached_num_tokens,
    }


cases = []
cases.append(run_case("single_request_no_cache", list(range(771))))
common_aligned = [123] * 768
cases.append(run_case(
    "aligned_prefix_tail_miss",
    common_aligned + [1000, 2000, 3000],
    common_aligned + [1001, 2001, 3001],
    expected_cached_tokens=768,
))
common_nonaligned = [77] * 600
cases.append(run_case(
    "nonaligned_prefix_partial_tail_reuse",
    common_nonaligned + [1, 2],
    common_nonaligned + [3, 4],
    expected_cached_tokens=600,
))
common_full = [55] * 768
cases.append(run_case(
    "full_hit_complete_blocks",
    common_full,
    common_full,
    expected_cached_tokens=768,
))
common_partial = [88] * 512
cases.append(run_case(
    "multi_block_prefix_then_tail",
    common_partial + [9] * 259,
    common_partial + [10] * 259,
    expected_cached_tokens=512,
))
cases.append(run_active_partial_copy_case())

for case in cases:
    print(case)
print("page_aware_prefill_checks: PASS")
INNER_PY

echo "[run] finished log_file=$LOG_FILE" | tee -a "$LOG_FILE"
