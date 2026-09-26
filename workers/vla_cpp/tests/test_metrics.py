import json

import pytest

from policykit import metrics
from policykit.metrics import compare_candidate, comparisons, timed_stage
from policykit.performance_report import build


def test_cost_and_quality_are_separate():
    baseline = {'status': 'complete', 'p50_ms': 100, 'success_rate': .8}
    candidate = {'status': 'complete', 'p50_ms': 60, 'quantize_seconds': 5, 'success_rate': .75}
    result = compare_candidate(baseline, candidate)
    assert result['speedup'] == pytest.approx(100/60)
    assert result['latency_change_pct'] == pytest.approx(-40)
    assert result['success_drop_pp'] == pytest.approx(5)
    assert result['quantization_break_even_calls'] == 125


def test_slower_or_unknown_is_not_a_break_even_or_quality_claim():
    baseline = {'status': 'engine_verified', 'p50_ms': 100}
    candidate = {'status': 'engine_verified', 'p50_ms': 110, 'quantize_seconds': 5, 'success_rate': .8}
    result = compare_candidate(baseline, candidate)
    assert result['speedup'] < 1
    assert result['quantization_break_even_calls'] is None
    assert result['success_drop_pp'] is None
    assert compare_candidate({}, candidate)['speedup'] is None
    candidate['status'] = 'failed'
    assert all(value is None for value in compare_candidate(baseline, candidate).values())


def test_baselines_are_model_specific():
    results = [{'model':'a','preset':'bf16','status':'complete','p50_ms':100},
               {'model':'b','preset':'q4','status':'complete','p50_ms':50}]
    assert comparisons(results)[1]['speedup'] is None


def test_stage_records_failure_and_propagates(monkeypatch):
    clock = iter([10, 12.5])
    monkeypatch.setattr(metrics, 'perf_counter', lambda: next(clock))
    timings = {}
    with pytest.raises(RuntimeError), timed_stage(timings, 'quantize_seconds'):
        raise RuntimeError('failed')
    assert timings == {'quantize_seconds': 2.5}


def test_pipeline_estimate_and_missing_values(tmp_path):
    rows = [{'preset':'float_reference','status':'engine_verified','p50_ms':100,'quantize_seconds':0},
            {'preset':'q4','status':'engine_verified','p50_ms':60,'quantize_seconds':5},
            {'preset':'q8','status':'running'}]
    result = build({'results':rows}, tmp_path/'results.json', 1000)
    assert result['results'][0]['estimated_quantize_plus_inference_seconds'] == 100
    assert result['results'][1]['estimated_quantize_plus_inference_seconds'] == 65
    assert result['results'][2]['estimated_quantize_plus_inference_seconds'] is None
    assert 'estimated_quantize_plus_inference_seconds' not in rows[0]
