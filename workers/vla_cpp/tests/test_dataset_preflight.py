import pytest

from policykit.dataset_preflight import assess


def fixtures():
    info={'total_episodes':30,'total_frames':4500,'fps':30,'features':{
        'action':{'shape':[6]},'observation.state':{'shape':[6]},
        'observation.images.front':{'shape':[1080,1920,3],'dtype':'video'}}}
    model={'input_features':{'observation.state':{'shape':[8]},
        'observation.images.image':{'type':'VISUAL'},'observation.images.image2':{'type':'VISUAL'}},
        'output_features':{'action':{'shape':[7]}}}
    profile={'dataset':{'repository':'example/data','revision':'rev','expected_action_dim':6,
                       'expected_state_dim':6,'camera':'observation.images.front'},'evaluation':{}}
    return info,model,profile


def test_libero_checkpoint_is_not_so101_compatible():
    result=assess(*fixtures())
    assert len(result['feature_mismatches']) == 3
    assert result['offline_evaluation_status'] == 'blocked_incompatible_checkpoint'
    assert result['task_success_drop_pp'] is None
    assert result['closed_loop_status'] == 'not_configured'


def test_matching_shapes_do_not_prove_simulator_or_semantic_compatibility():
    info,model,profile=fixtures()
    model={'input_features':{'observation.state':{'shape':[6]},
                            'observation.images.front':{'type':'VISUAL'}},
           'output_features':{'action':{'shape':[6]}}}
    result=assess(info,model,profile)
    assert result['feature_mismatches'] == []
    assert result['offline_evaluation_status'] == 'requires_checkpoint_semantics_and_split_validation'
    assert result['task_success_drop_pp'] is None


def test_dataset_contract_rejects_changed_dimensions():
    info,model,profile=fixtures()
    info['features']['action']['shape']=[7]
    with pytest.raises(ValueError,match='dimensions'):
        assess(info,model,profile)
