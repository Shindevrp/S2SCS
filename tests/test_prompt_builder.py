from app.cs_detection.cs_features import (
    CodeSwitchMetrics,
    EmbeddedLanguageIsland,
)
from app.dialect.camel_dialect import DialectSignal
from app.llm.prompt_builder import CodeSwitchPromptBuilder, ResponsePromptInput


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.87,
        raw_label="JED" if label == "Hejazi" else "MSA",
        raw_city="Jeddah" if label == "Hejazi" else None,
        raw_country="Saudi Arabia" if label == "Hejazi" else None,
        raw_region="Gulf" if label == "Hejazi" else "Modern Standard Arabic",
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.1, "Hejazi": 0.8},
        is_fallback=False,
    )


def test_prompt_builder_includes_all_structured_signals() -> None:
    metrics = CodeSwitchMetrics(
        cs_index=0.5,
        switch_count=2,
        valid_transition_count=4,
        matrix_language="AR",
        secondary_language="EN",
        embedded_language_islands=[
            EmbeddedLanguageIsland(
                language="EN",
                start_token_index=1,
                end_token_index=2,
                start_char=4,
                end_char=15,
                tokens=["machine", "learning"],
                text="machine learning",
                mean_score=0.91,
            )
        ],
        language_token_count=5,
        dialect_label="Hejazi",
    )
    prompt_input = ResponsePromptInput(
        normalized_text="انا احب machine learning",
        dialect_signal=build_signal("Hejazi"),
        code_switch_metrics=metrics,
        task_instruction="Reply warmly and briefly.",
    )

    prompt = CodeSwitchPromptBuilder().build(prompt_input)

    assert "انا احب machine learning" in prompt
    assert "Hejazi" in prompt
    assert "0.870" in prompt
    assert "0.500" in prompt
    assert "Arabic" in prompt
    assert "English" in prompt
    assert "machine learning" in prompt
    assert "Reply warmly and briefly." in prompt
    assert "Output only the final response text." in prompt
