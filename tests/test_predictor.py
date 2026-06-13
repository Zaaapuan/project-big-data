import pytest

from employee_app.predictor import InputValidationError


VALID_PAYLOAD = {
    "age": 30,
    "years_experience": 7,
    "education_level": 4,
    "department": "Research & Development",
}


def test_prediction_contains_complete_result(predictor):
    result = predictor.predict(VALID_PAYLOAD)

    assert result["category"] in {
        "Emerging Talent",
        "Academic Achiever",
        "Seasoned Veteran",
    }
    assert result["svm"]["category"] == result["category"]
    assert 0 <= result["svm"]["confidence"] <= 1
    assert result["kmeans"]["category"]
    assert isinstance(result["models_agree"], bool)
    assert result["description"]
    assert result["color"] in {"violet", "pink", "ink"}
    assert result["input_summary"]["education_label"] == "Master"
    assert result["process"]["input_validation"]["status"] == "valid"
    assert len(result["process"]["preprocessing"]["numeric_scaling"]) == 3
    assert len(result["process"]["preprocessing"]["transformed_vector"]) == 6
    assert len(result["process"]["kmeans"]["distances"]) == 3
    assert len(result["process"]["svm"]["probabilities"]) == 3
    assert "bukan penilaian performa kerja aktual" in result["disclaimer"]


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"age": 17}, "age"),
        ({"age": 61}, "age"),
        ({"years_experience": -1}, "years_experience"),
        ({"years_experience": 41}, "years_experience"),
        ({"education_level": 0}, "education_level"),
        ({"education_level": 6}, "education_level"),
        ({"department": "Finance"}, "department"),
        ({"age": 20, "years_experience": 10}, "years_experience"),
    ],
)
def test_invalid_input_is_rejected(predictor, changes, field):
    payload = VALID_PAYLOAD | changes

    with pytest.raises(InputValidationError) as error:
        predictor.predict(payload)

    assert field in error.value.errors


def test_missing_payload_is_rejected(predictor):
    with pytest.raises(InputValidationError) as error:
        predictor.predict(None)

    assert "request" in error.value.errors
