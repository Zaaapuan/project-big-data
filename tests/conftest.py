import pytest

from employee_app.config import DATASET_PATH
from employee_app.model_trainer import train_model
from employee_app.predictor import EmployeeProfilePredictor


@pytest.fixture(scope="session")
def model_bundle(tmp_path_factory):
    artifact_path = tmp_path_factory.mktemp("models") / "model.joblib"
    return train_model(DATASET_PATH, artifact_path)


@pytest.fixture(scope="session")
def predictor(model_bundle):
    return EmployeeProfilePredictor(model_bundle)

