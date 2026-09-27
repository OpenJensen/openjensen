from pathlib import Path

import pytest
from firebird_distill.contracts import teacher_info


def test_simulator_control_contract_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="does not yet preserve simulator control"):
        teacher_info(Path("not-opened"), {"control-contract.json": {}})
