from pathlib import Path

from app.config import PROJECT_ROOT, load_app_config


def test_load_app_config_reads_yaml_and_resolves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "server:",
                "  port: 9100",
                "models:",
                "  asr:",
                "    model_name_or_path: models/custom_asr",
                "pipeline:",
                "  task_instruction: Reply naturally.",
            ]
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.server.port == 9100
    assert config.pipeline.task_instruction == "Reply naturally."
    assert config.resolve_reference(
        config.models.asr.model_name_or_path,
        local_only=True,
    ) == str((PROJECT_ROOT / "models/custom_asr").resolve())
    assert config.server.cors_allow_origins == ["*"]
