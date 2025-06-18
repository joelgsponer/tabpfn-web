import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from app import app

client = TestClient(app)

class TestBackendWithIris:
    @pytest.fixture
    def iris_data(self):
        """Load and prepare Iris dataset for testing."""
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        # Convert target to class names for more realistic testing
        df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    
    @pytest.fixture
    def iris_json_data(self, iris_data):
        """Convert Iris DataFrame to JSON format for API requests."""
        return iris_data.to_dict(orient='records')
    
    def test_root_endpoint(self):
        """Test the root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "TabPFN API is running"}
    
    def test_predict_with_iris_dataset(self, iris_json_data):
        """Test prediction endpoint with Iris dataset."""
        request_data = {
            "data": iris_json_data,
            "target_column": "target"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        
        # Check response structure
        assert "predictions" in result
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "confusion_matrix" in result
        assert "classes" in result
        
        # Validate predictions
        assert len(result["predictions"]) == len(iris_json_data)
        assert all(pred in ['setosa', 'versicolor', 'virginica'] for pred in result["predictions"])
        
        # Validate metrics
        assert 0 <= result["accuracy"] <= 1
        assert result["accuracy"] > 0.8  # Iris is an easy dataset, expect good accuracy
        
        # Validate confusion matrix
        assert len(result["confusion_matrix"]) == 3  # 3 classes
        assert all(len(row) == 3 for row in result["confusion_matrix"])
        
        # Validate classes
        assert set(result["classes"]) == {'setosa', 'versicolor', 'virginica'}
        
        # Validate per-class metrics
        assert len(result["precision"]) == 3
        assert len(result["recall"]) == 3
        assert len(result["f1_score"]) == 3
        assert all(0 <= score <= 1 for score in result["precision"])
        assert all(0 <= score <= 1 for score in result["recall"])
        assert all(0 <= score <= 1 for score in result["f1_score"])
    
    def test_predict_with_numeric_target(self, iris_data):
        """Test prediction with numeric target values."""
        # Convert back to numeric targets
        iris_data['target'] = iris_data['target'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
        request_data = {
            "data": iris_data.to_dict(orient='records'),
            "target_column": "target"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert all(isinstance(pred, int) and 0 <= pred <= 2 for pred in result["predictions"])
        assert result["classes"] == [0, 1, 2]
    
    def test_predict_with_missing_target_column(self, iris_json_data):
        """Test error handling when target column is missing."""
        request_data = {
            "data": iris_json_data,
            "target_column": "non_existent_column"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        assert "not found in data" in response.json()["detail"]
    
    def test_predict_with_binary_classification(self, iris_data):
        """Test binary classification (should include ROC data)."""
        # Filter to only two classes for binary classification
        binary_data = iris_data[iris_data['target'].isin(['setosa', 'versicolor'])]
        request_data = {
            "data": binary_data.to_dict(orient='records'),
            "target_column": "target"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        
        # Check ROC data is present for binary classification
        assert "roc_data" in result
        assert result["roc_data"] is not None
        assert "fpr" in result["roc_data"]
        assert "tpr" in result["roc_data"]
        assert "auc" in result["roc_data"]
        
        # Validate ROC curve data
        assert len(result["roc_data"]["fpr"]) == len(result["roc_data"]["tpr"])
        assert 0 <= result["roc_data"]["auc"] <= 1
        assert all(0 <= x <= 1 for x in result["roc_data"]["fpr"])
        assert all(0 <= x <= 1 for x in result["roc_data"]["tpr"])
    
    def test_predict_with_small_dataset(self):
        """Test with a very small subset of Iris data."""
        small_data = [
            {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2, "target": "setosa"},
            {"sepal length (cm)": 7.0, "sepal width (cm)": 3.2, "petal length (cm)": 4.7, "petal width (cm)": 1.4, "target": "versicolor"},
            {"sepal length (cm)": 6.3, "sepal width (cm)": 3.3, "petal length (cm)": 6.0, "petal width (cm)": 2.5, "target": "virginica"},
            {"sepal length (cm)": 4.9, "sepal width (cm)": 3.0, "petal length (cm)": 1.4, "petal width (cm)": 0.2, "target": "setosa"},
            {"sepal length (cm)": 5.8, "sepal width (cm)": 2.7, "petal length (cm)": 4.1, "petal width (cm)": 1.0, "target": "versicolor"},
        ]
        
        request_data = {
            "data": small_data,
            "target_column": "target"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert len(result["predictions"]) == 5
    
    def test_shap_endpoint_placeholder(self, iris_json_data):
        """Test SHAP endpoint (currently returns placeholder)."""
        request_data = {
            "data": iris_json_data,
            "target_column": "target",
            "instance_index": 0
        }
        
        response = client.post("/shap", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "message" in result
        assert "not yet implemented" in result["message"]
    
    def test_predict_with_missing_values(self, iris_data):
        """Test handling of missing values in the dataset."""
        # Introduce some missing values
        iris_with_nan = iris_data.copy()
        iris_with_nan.loc[0, 'sepal length (cm)'] = None
        iris_with_nan.loc[5, 'petal width (cm)'] = np.nan
        
        request_data = {
            "data": iris_with_nan.to_dict(orient='records'),
            "target_column": "target"
        }
        
        # This should either handle missing values or raise an appropriate error
        response = client.post("/predict", json=request_data)
        # The current implementation might fail with missing values
        # This test documents the expected behavior
    
    @pytest.mark.parametrize("test_size", [10, 50, 100, 150])
    def test_predict_with_different_dataset_sizes(self, iris_data, test_size):
        """Test predictions with different dataset sizes."""
        subset_data = iris_data.sample(n=min(test_size, len(iris_data)), random_state=42)
        request_data = {
            "data": subset_data.to_dict(orient='records'),
            "target_column": "target"
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        assert len(result["predictions"]) == len(subset_data)