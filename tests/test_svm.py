from employee_app.core.preprocessing import employee_frame
from employee_app.models.svm import (
    build_svm_decision_projection,
    build_svm_trace,
    predict_with_probabilities,
    project_svm_new_point,
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


def test_rbf_svm_visualization_contains_boundary_and_support_vectors(model_bundle):
    points = model_bundle.cluster_plot["points"]
    projected = [[point["x"], point["y"]] for point in points]
    cluster_ids = [point["cluster_id"] for point in points]

    visualizer, plot = build_svm_decision_projection(
        projected,
        cluster_ids,
        model_bundle.cluster_labels,
    )
    sample = model_bundle.pca.transform(
        model_bundle.preprocessor.transform(
            employee_frame(30, 7, 4, "Research & Development")
        )
    )
    selected_cluster, _ = predict_with_probabilities(
        model_bundle.svm,
        model_bundle.preprocessor.transform(
            employee_frame(30, 7, 4, "Research & Development")
        ),
    )
    new_point = project_svm_new_point(
        visualizer,
        sample,
        selected_cluster,
        model_bundle.cluster_labels,
    )

    assert plot["kernel"] == "rbf"
    assert len(plot["grid"]["x_values"]) == 100
    assert len(plot["grid"]["y_values"]) == 60
    assert len(plot["grid"]["cluster_ids"]) == 6000
    assert plot["support_vectors"]
    assert new_point["category"] == model_bundle.cluster_labels[selected_cluster]
