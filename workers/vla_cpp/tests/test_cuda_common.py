import pytest

from policykit.cuda_common import parse_actions


def test_action_metrics_check_padding_but_score_only_real_channels():
    values = [str(channel) for _ in range(50) for channel in range(32)]
    parsed = parse_actions('action_len=1600\n'+'\n'.join(values))
    assert len(parsed) == 350
    assert parsed[:7] == list(range(7))
    values[-1] = 'nan'
    with pytest.raises(ValueError, match='finite'):
        parse_actions('action_len=1600\n'+'\n'.join(values))


@pytest.mark.parametrize('length', [1599, 1601])
def test_declared_shape_must_match_even_when_1600_values_follow(length):
    with pytest.raises(ValueError, match='declared'):
        parse_actions(f'action_len={length}\n'+'\n'.join(['0']*1600))
