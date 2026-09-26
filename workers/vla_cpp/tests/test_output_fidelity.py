import json
import math

import pytest

from policykit.output_fidelity import compare_actions, from_pairs, from_run, render
from policykit.performance_report import build


def test_identical_output_and_zero_reference():
    m = compare_actions([[0,1],[2,3]], [[0,1],[2,3]])
    assert m['agreement']['all_exact'] is True
    assert m['agreement']['rmse'] == 0
    assert m['agreement']['within_tolerance_fraction'] == 1
    assert m['target_error'] is None


def test_known_errors_and_channel_specific_tolerances():
    m = compare_actions([[0,0],[0,0]], [[0,2],[0,4]], atol=[0,3],rtol=0)
    assert m['agreement']['mae'] == 1.5
    assert m['agreement']['rmse'] == math.sqrt(5)
    assert m['agreement']['max_abs_error'] == 4
    assert m['agreement']['p95_abs_error'] == 4
    assert m['agreement']['exact_match_fraction'] == .5
    assert m['agreement']['within_tolerance_fraction'] == .75
    assert m['agreement']['steps_within_tolerance_fraction'] == .5
    assert m['agreement']['all_within_tolerance'] is False
    assert m['per_channel'][0]['mae'] == 0
    assert m['per_channel'][1]['mae'] == 3


def test_relative_tolerance_uses_reference_and_boundary_is_inclusive():
    m = compare_actions([[0,100]], [[1,111]],atol=1,rtol=.1)
    assert m['agreement']['all_within_tolerance'] is True
    assert compare_actions([[0]],[[.01]],atol=0,rtol=.1)['agreement']['all_within_tolerance'] is False


def test_target_accuracy_can_improve_even_if_baseline_agreement_worsens():
    m = compare_actions([[2,2]], [[1,1]], targets=[[0,0]])
    assert m['agreement']['all_exact'] is False
    assert m['target_error']['baseline']['mae'] == 2
    assert m['target_error']['candidate']['mae'] == 1
    assert m['target_error']['mae_increase'] == -1
    assert m['target_error']['per_channel'][0]['candidate']['rmse'] == 1


@pytest.mark.parametrize('reference,candidate', [([],[]), ([[1]],[[1,2]]),
    ([[1],[2]],[[1]]), ([[1],[2,3]],[[1],[2]]), ([[float('nan')]],[[1]]),
    ([[1]],[[float('inf')]]), ([[True]],[[1]]), ([[1]],[['1']])])
def test_invalid_data_is_rejected(reference,candidate):
    with pytest.raises(ValueError):
        compare_actions(reference,candidate)


@pytest.mark.parametrize('atol,rtol', [(-1,0), (0,float('nan')), ([0,0],0), (0,-1), (float('inf'),0)])
def test_invalid_tolerance_is_rejected(atol,rtol):
    with pytest.raises(ValueError):
        compare_actions([[0]],[[0]],atol=atol,rtol=rtol)


def test_target_shape_rejected():
    with pytest.raises(ValueError,match='Target'):
        compare_actions([[1]],[[1]],targets=[[1,2]])


def test_dataset_cases_require_unique_pair_ids(tmp_path):
    path = tmp_path/'pairs.json'
    case = {'id':'obs-1','reference':[[1]],'candidate':[[2]],'targets':[[2]]}
    path.write_text(json.dumps({'cases':[case]}))
    report = from_pairs(path,.001,.01)
    assert report['cases'][0]['output_fidelity']['target_error']['candidate']['mae'] == 0
    path.write_text(json.dumps({'cases':[case,case]}))
    with pytest.raises(ValueError,match='unique'):
        from_pairs(path,.001,.01)


def test_saved_evidence_integration_and_stale_result_rejection(tmp_path):
    path = tmp_path/'results.json'
    payload = {'results':[
        {'preset':'float_reference','sha256':'baseline','status':'engine_verified'},
        {'preset':'lm_q4','sha256':'candidate','status':'engine_verified'}]}
    path.write_text(json.dumps(payload))
    (tmp_path/'float_reference-actions.json').write_text(json.dumps([0]*350))
    (tmp_path/'lm_q4-actions.json').write_text(json.dumps([.01]*350))
    fidelity = from_run(path,.001,.01)
    assert fidelity['results'][1]['output_fidelity']['agreement']['mae'] == pytest.approx(.01)
    assert '0.00%' in render(fidelity)
    result = build(payload,path,1000,fidelity)
    assert result['results'][1]['output_fidelity']['values'] == 350
    path.write_text(json.dumps({**payload,'changed':True}))
    with pytest.raises(ValueError,match='does not match'):
        build(payload,path,1000,fidelity)


def test_accuracy_report_shows_supplied_target_error():
    payload = {'scope':'test', 'cases':[{'case_id':'one', 'output_fidelity':
        compare_actions([[2]],[[1]],targets=[[0]])}]}
    page = render(payload)
    assert 'Error against supplied target actions' in page
    assert '| one | 2 | 1 | 2 | 1 |' in page


def test_timing_control_exposes_drift_without_changing_raw_ratios():
    from policykit.performance_report import add_control
    original = {'hardware':{'cpu':'test'},'source':{'revision':'one'},'binary_sha256':'bin','patch_sha256':'patch',
                'results':[{'preset':'float_reference','status':'engine_verified','sha256':'model','p50_ms':100}]}
    control = {**original,'results':[{**original['results'][0],'p50_ms':125}]}
    report = {}
    add_control(report,original,control)
    assert report['baseline_control']['change_pct'] == 25
    assert original['results'][0]['p50_ms'] == 100
    control['binary_sha256'] = 'different'
    with pytest.raises(ValueError,match='binary_sha256'):
        add_control({},original,control)
