from employee_app.core.config import DATASET_PATH, DEPARTMENTS, MODEL_FEATURES
from employee_app.core.data_loader import dataset_hash, load_dataset


def test_dataset_has_expected_schema_and_values():
    dataframe = load_dataset(DATASET_PATH)

    assert list(dataframe.columns) == MODEL_FEATURES
    assert len(dataframe) == 1470
    assert set(dataframe["Department"]) == set(DEPARTMENTS)
    assert set(dataframe["Education"]) == {1, 2, 3, 4, 5}
    assert dataframe["Age"].between(18, 60).all()
    assert dataframe["TotalWorkingYears"].between(0, 40).all()


def test_dataset_hash_is_stable():
    assert dataset_hash(DATASET_PATH) == (
        "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"
    )
