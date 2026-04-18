from app.cs_detection.cs_features import TokenPrediction
from app.dialect.camel_dialect import DialectSignal
from app.tts.tts_router import ResponseTTSRouter


class FakeCodeSwitchDetector:
    def __init__(self, predictions):
        self.predictions = predictions
        self.last_text = None
        self.last_dialect_signal = None

    def predict_tokens(self, text, dialect_signal=None):
        self.last_text = text
        self.last_dialect_signal = dialect_signal
        return type("DetectionResult", (), {"predictions": self.predictions})()


def build_prediction(
    token: str,
    label: str,
    start_char: int,
    end_char: int,
    dialect_label: str = "Gulf",
    score: float = 0.9,
) -> TokenPrediction:
    return TokenPrediction(
        token=token,
        label=label,
        score=score,
        start_char=start_char,
        end_char=end_char,
        dialect_label=dialect_label,
    )


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.9,
        raw_label="RIY" if label == "Gulf" else "JED",
        raw_city=None,
        raw_country=None,
        raw_region=None,
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.8, "Hejazi": 0.1},
        is_fallback=False,
    )


def test_route_response_builds_language_segments_and_routes_voices() -> None:
    text = "ياهلا hello team معاك"
    predictions = [
        build_prediction("ياهلا", "AR", 0, 5, dialect_label="Hejazi"),
        build_prediction("hello", "EN", 6, 11, dialect_label="Hejazi"),
        build_prediction("team", "EN", 12, 16, dialect_label="Hejazi"),
        build_prediction("معاك", "AR", 17, 21, dialect_label="Hejazi"),
    ]
    router = ResponseTTSRouter(
        code_switch_detector=FakeCodeSwitchDetector(predictions),
    )

    result = router.route_response(text, dialect_signal=build_signal("Hejazi"))

    assert result.matrix_language == "AR"
    assert [segment.text for segment in result.segments] == ["ياهلا", "hello team", "معاك"]
    assert [segment.language for segment in result.segments] == ["AR", "EN", "AR"]
    assert [segment.voice.voice_id for segment in result.segments] == [
        "ar_hejazi_default",
        "en_default",
        "ar_hejazi_default",
    ]


def test_route_response_merges_named_entities_into_neighbor_language_segment() -> None:
    text = "hello IBM team"
    predictions = [
        build_prediction("hello", "EN", 0, 5),
        build_prediction("IBM", "NE", 6, 9),
        build_prediction("team", "EN", 10, 14),
    ]
    router = ResponseTTSRouter(
        code_switch_detector=FakeCodeSwitchDetector(predictions),
    )

    result = router.route_response(text, dialect_signal=build_signal("Gulf"))

    assert len(result.segments) == 1
    assert result.segments[0].language == "EN"
    assert result.segments[0].text == "hello IBM team"
    assert result.segments[0].voice.voice_id == "en_default"


def test_route_response_uses_matrix_language_for_other_tokens_when_needed() -> None:
    text = "مرحبا ! hello"
    predictions = [
        build_prediction("مرحبا", "AR", 0, 5, dialect_label="Gulf"),
        build_prediction("!", "OTHER", 6, 7, dialect_label="Gulf"),
        build_prediction("hello", "EN", 8, 13, dialect_label="Gulf"),
    ]
    router = ResponseTTSRouter(
        code_switch_detector=FakeCodeSwitchDetector(predictions),
    )

    result = router.route_response(text, dialect_signal=build_signal("Gulf"))

    assert [segment.text for segment in result.segments] == ["مرحبا !", "hello"]
    assert [segment.language for segment in result.segments] == ["AR", "EN"]
    assert result.segments[0].voice.voice_id == "ar_gulf_default"


def test_route_response_accepts_precomputed_token_predictions() -> None:
    predictions = [
        build_prediction("hello", "EN", 0, 5),
        build_prediction("مرحبا", "AR", 6, 11),
    ]
    router = ResponseTTSRouter()

    result = router.route_response("hello مرحبا", token_predictions=predictions)

    assert len(result.segments) == 2
    assert result.dialect_label == "Gulf"
