"""Timing and same-model comparisons for the quantization pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import math
from time import perf_counter


@contextmanager
def timed_stage(timings: dict, name: str):
    """Record elapsed wall time even when a stage raises; exceptions propagate."""
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = perf_counter() - start


def compare_candidate(baseline: dict, candidate: dict) -> dict:
    """Unknown measurements remain null. Speedup >1 means faster."""
    output = dict(speedup=None, latency_change_pct=None, success_drop_pp=None,
                  quantization_break_even_calls=None)
    if candidate.get('status') not in {'complete', 'engine_benchmarked', 'engine_verified', 'engine_measured'}:
        return output
    base_ms, quant_ms = baseline.get('p50_ms'), candidate.get('p50_ms')
    if base_ms is not None and quant_ms is not None and base_ms > 0 and quant_ms > 0:
        output['speedup'] = base_ms / quant_ms
        output['latency_change_pct'] = 100 * (quant_ms / base_ms - 1)
        cost = candidate.get('quantize_seconds')
        if cost is not None and cost >= 0 and quant_ms < base_ms:
            output['quantization_break_even_calls'] = math.ceil(cost * 1000 / (base_ms - quant_ms))
    if baseline.get('status') == candidate.get('status') == 'complete':
        base_rate, quant_rate = baseline.get('success_rate'), candidate.get('success_rate')
        if base_rate is not None and quant_rate is not None:
            output['success_drop_pp'] = 100 * (base_rate - quant_rate)
    return output


def comparisons(results: list[dict]) -> list[dict]:
    output = []
    for candidate in results:
        baseline = next((row for row in results
                         if row.get('model') == candidate.get('model')
                         and row['preset'] in {'bf16', 'float_reference'}
                         and row.get('status') in {'complete', 'engine_benchmarked', 'engine_verified', 'engine_measured'}), None)
        values = compare_candidate(baseline or {}, candidate)
        output.append({'model': candidate.get('model'), 'preset': candidate['preset'], **values})
    return output
