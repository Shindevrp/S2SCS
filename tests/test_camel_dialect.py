import pytest

from app.dialect.camel_dialect import CamelDialectIdentifier


class FakePrediction:
    def __init__(self, top: str, scores: dict[str, float]) -> None:
        self.top = top
        self.scores = scores


class FakeDialectModel:
    def __init__(self, prediction: FakePrediction) -> None:
        self.prediction = prediction
        self.last_texts = None
        self.last_output = None

    def predict(self, texts, output="label"):
        self.last_texts = texts
        self.last_output = output
        return [self.prediction]


def test_identify_returns_msa_signal() -> None:
    model = FakeDialectModel(FakePrediction(top="MSA", scores={"MSA": 0.92, "JED": 0.03}))
    identifier = CamelDialectIdentifier(model=model)

    signal = identifier.identify("هذا اختبار باللغة العربية الفصحى")

    assert signal.conditioning_label == "MSA"
    assert signal.is_fallback is False
    assert signal.raw_label == "MSA"
    assert signal.raw_region == "Modern Standard Arabic"


def test_identify_maps_jeddah_to_hejazi() -> None:
    model = FakeDialectModel(FakePrediction(top="JED", scores={"JED": 0.81, "RIY": 0.10}))
    identifier = CamelDialectIdentifier(model=model)

    signal = identifier.identify("ياخي كيف الحال اليوم")

    assert signal.conditioning_label == "Hejazi"
    assert signal.is_fallback is False
    assert signal.raw_city == "Jeddah"
    assert signal.bucket_scores["Hejazi"] == pytest.approx(0.81)


def test_identify_maps_riyadh_to_gulf() -> None:
    model = FakeDialectModel(FakePrediction(top="RIY", scores={"RIY": 0.73, "MSA": 0.15}))
    identifier = CamelDialectIdentifier(model=model)

    signal = identifier.identify("وش الأخبار معك اليوم")

    assert signal.conditioning_label == "Gulf"
    assert signal.is_fallback is False
    assert signal.raw_country == "Saudi Arabia"
    assert signal.bucket_scores["Gulf"] == pytest.approx(0.73)


def test_identify_extracts_only_arabic_from_code_switched_text() -> None:
    model = FakeDialectModel(FakePrediction(top="RIY", scores={"RIY": 0.88}))
    identifier = CamelDialectIdentifier(model=model)

    signal = identifier.identify("hello يا حبيبي see you later")

    assert model.last_texts == ["يا حبيبي"]
    assert model.last_output == "label"
    assert signal.normalized_text == "يا حبيبي"


def test_identify_falls_back_to_msa_for_out_of_scope_label() -> None:
    model = FakeDialectModel(FakePrediction(top="CAI", scores={"CAI": 0.79, "MSA": 0.05}))
    identifier = CamelDialectIdentifier(model=model)

    signal = identifier.identify("احنا رايحين بكرة ان شاء الله")

    assert signal.conditioning_label == "MSA"
    assert signal.is_fallback is True
    assert signal.fallback_reason == "out_of_scope_dialect"
    assert signal.raw_label == "CAI"


def test_identify_falls_back_to_msa_for_low_confidence_target_label() -> None:
    model = FakeDialectModel(FakePrediction(top="JED", scores={"JED": 0.22, "MSA": 0.20}))
    identifier = CamelDialectIdentifier(model=model, confidence_threshold=0.40)

    signal = identifier.identify("كيفك يا صاحبي اليوم")

    assert signal.conditioning_label == "MSA"
    assert signal.is_fallback is True
    assert signal.fallback_reason == "low_confidence"


def test_identify_skips_model_when_arabic_content_is_too_small() -> None:
    model = FakeDialectModel(FakePrediction(top="MSA", scores={"MSA": 0.99}))
    identifier = CamelDialectIdentifier(model=model, minimum_arabic_chars=6)

    signal = identifier.identify("hello ok bye")

    assert signal.conditioning_label == "MSA"
    assert signal.is_fallback is True
    assert signal.fallback_reason == "insufficient_arabic_content"
    assert model.last_texts is None
