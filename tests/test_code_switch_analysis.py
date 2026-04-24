import pytest

from app.cs_detection.cs_features import (
    EmbeddedLanguageIsland,
    TokenPrediction,
    analyze_code_switch,
    compute_cs_index,
    detect_matrix_language,
    extract_features,
    extract_embedded_language_islands,
)


def build_prediction(
    token: str,
    label: str,
    start_char: int,
    end_char: int,
    score: float = 0.9,
    dialect_label: str = "Gulf",
) -> TokenPrediction:
    return TokenPrediction(
        token=token,
        label=label,
        score=score,
        start_char=start_char,
        end_char=end_char,
        dialect_label=dialect_label,
    )


def test_compute_cs_index_ignores_neutral_labels_for_switch_density() -> None:
    predictions = [
        build_prediction("انا", "AR", 0, 3),
        build_prediction("from", "EN", 4, 8),
        build_prediction("IBM", "NE", 9, 12),
        build_prediction("اليوم", "AR", 13, 18),
    ]

    cs_index, switch_count, transition_count = compute_cs_index(predictions)

    assert cs_index == pytest.approx(1.0)
    assert switch_count == 2
    assert transition_count == 2


def test_detect_matrix_language_uses_language_token_majority() -> None:
    predictions = [
        build_prediction("hello", "EN", 0, 5),
        build_prediction("team", "EN", 6, 10),
        build_prediction("مرحبا", "AR", 11, 16),
    ]

    assert detect_matrix_language(predictions) == "EN"


def test_detect_matrix_language_breaks_ties_with_confidence_then_order() -> None:
    predictions = [
        build_prediction("مرحبا", "AR", 0, 5, score=0.60),
        build_prediction("hello", "EN", 6, 11, score=0.55),
        build_prediction("world", "EN", 12, 17, score=0.40),
        build_prediction("اليوم", "AR", 18, 23, score=0.50),
    ]

    assert detect_matrix_language(predictions) == "AR"


def test_extract_embedded_language_islands_returns_secondary_language_spans() -> None:
    predictions = [
        build_prediction("انا", "AR", 0, 3),
        build_prediction("love", "EN", 4, 8),
        build_prediction("machine", "EN", 9, 16),
        build_prediction("learning", "EN", 17, 25),
        build_prediction("اليوم", "AR", 26, 31),
        build_prediction("thanks", "EN", 32, 38),
    ]

    islands = extract_embedded_language_islands(predictions, matrix_language="AR")

    assert len(islands) == 2
    assert isinstance(islands[0], EmbeddedLanguageIsland)
    assert islands[0].language == "EN"
    assert islands[0].start_token_index == 1
    assert islands[0].end_token_index == 3
    assert islands[0].text == "love machine learning"
    assert islands[1].text == "thanks"


def test_analyze_code_switch_returns_all_core_metrics() -> None:
    predictions = [
        build_prediction("hello", "EN", 0, 5, dialect_label="Hejazi"),
        build_prediction("يا", "AR", 6, 8, dialect_label="Hejazi"),
        build_prediction("team", "EN", 9, 13, dialect_label="Hejazi"),
        build_prediction("!", "OTHER", 13, 14, dialect_label="Hejazi"),
    ]

    metrics = analyze_code_switch(predictions)

    assert metrics.cs_index == pytest.approx(1.0)
    assert metrics.switch_count == 2
    assert metrics.valid_transition_count == 2
    assert metrics.matrix_language == "EN"
    assert metrics.secondary_language == "AR"
    assert len(metrics.embedded_language_islands) == 1
    assert metrics.embedded_language_islands[0].text == "يا"
    assert metrics.language_token_count == 3
    assert metrics.dialect_label == "Hejazi"


def test_analyze_code_switch_handles_no_language_tokens() -> None:
    predictions = [
        build_prediction("IBM", "NE", 0, 3),
        build_prediction("2026", "OTHER", 4, 8),
    ]

    metrics = analyze_code_switch(predictions, dialect_label="MSA")

    assert metrics.cs_index == 0.0
    assert metrics.matrix_language == "UNKNOWN"
    assert metrics.secondary_language is None
    assert metrics.embedded_language_islands == []
    assert metrics.language_token_count == 0


def test_extract_features_matches_example_output() -> None:
    tokens = ["ماذا", "تفعل", "today", "انا", "coffee"]
    labels = ["AR", "AR", "EN", "AR", "EN"]

    features = extract_features(tokens, labels)

    assert features["switch_points"] == 3
    assert features["CSI"] == pytest.approx(0.6)
    assert features["matrix_language"] == "AR"
    assert len(features["embedded_islands"]) == 2
    assert features["embedded_islands"][0]["text"] == "today"
    assert features["embedded_islands"][1]["text"] == "coffee"


def test_extract_features_handles_empty_input() -> None:
    features = extract_features([], [])

    assert features["switch_points"] == 0
    assert features["CSI"] == 0.0
    assert features["matrix_language"] == "UNKNOWN"
    assert features["embedded_islands"] == []


def test_extract_features_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        extract_features(["hello"], ["EN", "AR"])
