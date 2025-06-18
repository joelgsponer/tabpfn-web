import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def sample_prediction_data():
    """Sample data for testing predictions."""
    return {
        "data": [
            {"feature1": 1.0, "feature2": 2.0, "target": "A"},
            {"feature1": 3.0, "feature2": 4.0, "target": "B"},
            {"feature1": 5.0, "feature2": 6.0, "target": "A"},
            {"feature1": 7.0, "feature2": 8.0, "target": "B"},
        ],
        "target_column": "target"
    }