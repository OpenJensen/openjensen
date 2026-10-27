"""Shared OpenAI credentials feed provider profiles without a paid request."""

import pytest
from pydantic import SecretStr
from vla_platform.agent_control.contracts import ProviderInput
from vla_platform.agent_control.settings import ProfileError, Profiles
from vla_platform.api_connections import ApiConnections
from vla_platform.modules import initialize_modules, set_module_installed
from vla_platform.settings import Settings

KEY = "private-shared-provider-fixture-12345"


def test_provider_uses_saved_shared_key_and_rechecks_after_key_change(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    initialize_modules(
        Settings(data_dir=tmp_path), tmp_path / "robot-lab-connection.json", existing=False
    )
    set_module_installed(tmp_path, "openai", True)
    keys = ApiConnections(tmp_path)
    keys.save("openai", SecretStr(KEY))
    profiles = Profiles(tmp_path)
    profiles.save("openai", ProviderInput(provider="openai_responses"))
    original = profiles.get("openai")
    assert original["api_key"] == KEY
    profiles.checked("openai", original["revision"], ["text"])
    assert profiles.get("openai")["checked"]
    keys.save("openai", SecretStr(KEY + "-changed"))
    changed = profiles.get("openai")
    assert changed["api_key"] == KEY + "-changed"
    assert changed["revision"] != original["revision"] and not changed["checked"]
    assert KEY not in profiles.public("openai").model_dump_json()
    with pytest.raises(ProfileError, match="changed"):
        profiles.checked("openai", original["revision"], ["text"])
    profiles.checked("openai", changed["revision"], ["text"])
    assert profiles.get("openai")["checked"]
    set_module_installed(tmp_path, "openai", False)
    assert [item.id for item in profiles.list()] == ["mock"]
    with pytest.raises(ProfileError, match="Install OpenAI"):
        profiles.get("openai")
    assert keys.key("openai") == KEY + "-changed"
