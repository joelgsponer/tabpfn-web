from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TabPFN API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str

class PredictResponse(BaseModel):
    predictions: List[Any]
    accuracy: float
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    confusion_matrix: List[List[int]]
    classes: List[Any]
    roc_data: Optional[Dict[str, Any]] = None

class ShapRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    instance_index: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "TabPFN API is running"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found in data")
        
        # Separate features and target
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
        # For now, we'll use a dummy classifier as TabPFN is not installed
        # In production, replace this with actual TabPFN model
        from sklearn.ensemble import RandomForestClassifier
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle categorical features
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                encoders[col] = le
        
        # Encode target if categorical
        target_encoder = None
        if y_train.dtype == 'object':
            target_encoder = LabelEncoder()
            y_train = target_encoder.fit_transform(y_train)
            y_test = target_encoder.transform(y_test)
            classes = target_encoder.classes_.tolist()
        else:
            classes = sorted(y.unique().tolist())
        
        # Train model (placeholder for TabPFN)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC data for binary classification
        roc_data = None
        if len(classes) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc)
            }
        
        # Prepare full dataset predictions
        X_full = X.copy()
        for col, encoder in encoders.items():
            X_full[col] = encoder.transform(X_full[col].astype(str))
        
        predictions = model.predict(X_full)
        if target_encoder is not None:
            predictions = target_encoder.inverse_transform(predictions)
        
        return PredictResponse(
            predictions=predictions.tolist(),
            accuracy=float(accuracy),
            precision=precision.tolist(),
            recall=recall.tolist(),
            f1_score=f1.tolist(),
            confusion_matrix=cm.tolist(),
            classes=classes,
            roc_data=roc_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shap")
async def compute_shap(request: ShapRequest):
    # TODO: Implement SHAP value computation
    # This will require the trained model from the predict endpoint
    # For now, return a placeholder
    return {
        "message": "SHAP computation not yet implemented",
        "feature_importance": {},
        "shap_values": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)