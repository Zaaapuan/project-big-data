from employee_app.core.preprocessing import employee_frame
from employee_app.models.svm import (
    build_svm_trace,
    predict_with_probabilities,
)


def test_svm_consistency_exceeds_acceptance_threshold(model_bundle):
    assert model_bundle.metrics["svm_balanced_accuracy"] >= 0.90


def test_svm_prediction_includes_calibrated_probabilities(model_bundle):
    transformed = model_bundle.preprocessor.transform(
        employee_frame(30, 7, 4, "Research & Development")
    )
    selected_cluster, probabilities = predict_with_probabilities(
        model_bundle.svm,
        transformed,
    )
    trace = build_svm_trace(
        probabilities,
        selected_cluster,
        model_bundle.cluster_labels,
    )

    assert set(probabilities) == {0, 1, 2}
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    assert len(trace["probabilities"]) == 3
    assert trace["selected_cluster_id"] == selected_cluster
