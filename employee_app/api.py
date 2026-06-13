"""Flask routes used by the local desktop dashboard."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from employee_app.config import ROOT_DIR
from employee_app.model_trainer import load_or_train_model
from employee_app.predictor import EmployeeProfilePredictor, InputValidationError


def create_app(
    predictor: EmployeeProfilePredictor | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static"),
    )

    if predictor is None:
        bundle, loaded_from_cache = load_or_train_model()
        predictor = EmployeeProfilePredictor(bundle, loaded_from_cache)

    app.config["PREDICTOR"] = predictor

    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ready",
                "model_loaded": True,
                "pipeline_version": predictor.bundle.pipeline_version,
            }
        )

    @app.get("/api/model-info")
    def model_info():
        return jsonify(predictor.model_info())

    @app.post("/api/predict")
    def predict():
        try:
            result = predictor.predict(request.get_json(silent=True))
        except InputValidationError as error:
            return (
                jsonify(
                    {
                        "error": "validation_error",
                        "message": str(error),
                        "fields": error.errors,
                    }
                ),
                400,
            )
        return jsonify(result)

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "not_found", "message": "Endpoint tidak ditemukan."}), 404

    @app.errorhandler(500)
    def internal_error(_error):
        return (
            jsonify(
                {
                    "error": "internal_error",
                    "message": "Terjadi kesalahan internal pada aplikasi.",
                }
            ),
            500,
        )

    return app
