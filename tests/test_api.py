from employee_app.api import create_app


def make_client(predictor):
    app = create_app(predictor)
    app.config.update(TESTING=True)
    return app.test_client()


def test_dashboard_and_health_endpoints(predictor):
    client = make_client(predictor)

    dashboard = client.get("/")
    health = client.get("/api/health")
    model_info = client.get("/api/model-info")

    assert dashboard.status_code == 200
    assert b"Simulasi Clustering Profil Karyawan" in dashboard.data
    assert b"Konfigurasi Analisis" not in dashboard.data
    assert b"Pemeriksaan input" not in dashboard.data
    assert b'id="result-section"' in dashboard.data
    assert dashboard.data.count(b'class="wizard-screen') == 6
    assert dashboard.data.count(b'class="stepper-item') == 6
    assert b'id="baseline-cluster-plot"' in dashboard.data
    assert b'id="prediction-cluster-plot"' in dashboard.data
    assert b'id="svm-decision-plot"' in dashboard.data
    assert b'id="plot-lightbox"' in dashboard.data
    assert b"RUMUS JARAK" not in dashboard.data
    assert b"Semakin kecil jaraknya" in dashboard.data
    assert b"/static/css/styles.css" in dashboard.data
    font = client.get("/static/fonts/Rubik-Variable.ttf")
    stylesheet = client.get("/static/css/styles.css")
    assert font.status_code == 200
    assert b"@font-face" in stylesheet.data
    assert b"gradient(" not in stylesheet.data
    assert health.get_json()["status"] == "ready"
    assert model_info.get_json()["dataset_rows"] == 1470
    assert len(model_info.get_json()["categories"]) == 3
    assert len(model_info.get_json()["cluster_plot"]["points"]) == 1470
    assert len(model_info.get_json()["cluster_plot"]["centroids"]) == 3
    assert model_info.get_json()["svm_plot"]["kernel"] == "rbf"
    assert len(model_info.get_json()["svm_plot"]["grid"]["cluster_ids"]) == 6000


def test_predict_endpoint_returns_model_results(predictor):
    client = make_client(predictor)
    response = client.post(
        "/api/predict",
        json={
            "age": 30,
            "years_experience": 7,
            "education_level": 4,
            "department": "Research & Development",
        },
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["svm"]["category"]
    assert body["kmeans"]["category"]
    assert "confidence" in body["svm"]
    assert "preprocessing" in body["process"]
    assert "kmeans" in body["process"]
    assert "svm" in body["process"]
    assert "svm_plot" in body["process"]
    assert "cluster_plot" in body["process"]
    assert body["process"]["cluster_plot"]["new_point"]["category"]
    assert body["process"]["svm_plot"]["new_point"]["category"]
    assert "disclaimer" not in body


def test_predict_endpoint_returns_structured_validation_errors(predictor):
    client = make_client(predictor)
    response = client.post(
        "/api/predict",
        json={
            "age": 20,
            "years_experience": 15,
            "education_level": 9,
            "department": "Unknown",
        },
    )

    body = response.get_json()
    assert response.status_code == 400
    assert body["error"] == "validation_error"
    assert set(body["fields"]) == {
        "years_experience",
        "education_level",
        "department",
    }


def test_predict_endpoint_rejects_non_json_body(predictor):
    client = make_client(predictor)
    response = client.post(
        "/api/predict",
        data="not-json",
        content_type="text/plain",
    )

    assert response.status_code == 400
    assert "request" in response.get_json()["fields"]
