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
from nanovllm.engine.block_manager import BlockManager
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
    assert build_page_aware_prefill_slot_mapping(seq, BLOCK_SIZE) == build_legacy_prefill_slot_mapping(seq, BLOCK_SIZE)


def run_case(name, first_prompt, second_prompt=None, expected_cached_tokens=None):
    manager = BlockManager(64, BLOCK_SIZE, PAGE_SIZE)
    seq1 = make_seq(first_prompt)
    plan1 = manager.make_prefill_plan(seq1)
    manager.allocate(seq1, plan1)
    validate_allocated_seq(seq1, plan1)
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
    "nonaligned_prefix_still_block_only",
    common_nonaligned + [1, 2],
    common_nonaligned + [3, 4],
    expected_cached_tokens=512,
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

for case in cases:
    print(case)
print("page_aware_prefill_checks: PASS")
INNER_PY

echo "[run] finished log_file=$LOG_FILE" | tee -a "$LOG_FILE"
