from self_evolution_agent.config import Settings


def test_qwen35_vision_defaults_include_optional_adapter() -> None:
    settings = Settings(_env_file=None)

    assert settings.vision_model_name == "Qwen/Qwen3.5-2B"
    assert settings.vision_adapter_path == ""
    assert settings.vision_model_version == "qwen3.5-2b-fridge-qlora-v1"


def test_vision_adapter_path_can_be_configured() -> None:
    settings = Settings(_env_file=None, vision_adapter_path="outputs/fridge-adapter")

    assert settings.vision_adapter_path == "outputs/fridge-adapter"
