from search_assistant.config import Settings


def test_settings_loads_defaults_for_local_mode(tmp_path):
    settings = Settings.from_env({"SEARCH_ASSISTANT_DATA_DIR": str(tmp_path)})

    assert settings.data_dir == tmp_path
    assert settings.database_path == tmp_path / "assistant.sqlite3"
    assert settings.feishu_enabled is False
