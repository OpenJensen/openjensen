import pytest
from patch_vllm_pooling import NEW, OLD, patch


def test_patch_is_idempotent_and_preserves_backup(tmp_path):
    path = tmp_path / "runner.py"
    path.write_text(OLD)
    assert patch(path)
    assert path.read_text() == NEW
    assert not patch(path)
    assert path.with_suffix(".py.original").read_text() == OLD


def test_unknown_source_is_rejected_without_changes(tmp_path):
    path = tmp_path / "runner.py"
    path.write_text("unknown version")
    with pytest.raises(RuntimeError, match="Unexpected"):
        patch(path)
    assert path.read_text() == "unknown version"
