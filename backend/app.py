from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import logging
import pickle
import base64
from io import BytesIO
import torch

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
    model_config = {"protected_namespaces": ()}
    
    data: List[Dict[str, Any]]
    target_column: str
    column_types: Optional[Dict[str, str]] = {}
    excluded_columns: Optional[List[str]] = []
    model_config_params: Optional[Dict[str, Any]] = {}

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    predictions: List[Any]
    accuracy: float
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    confusion_matrix: List[List[int]]
    classes: List[Any]
    roc_data: Optional[Dict[str, Any]] = None
    model_data: Optional[str] = None  # Base64 encoded model
    model_metadata: Optional[Dict[str, Any]] = None

class ShapRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_column: str
    instance_index: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "TabPFN API is running"}

@app.get("/device-info")
async def get_device_info():
    """Get information about available computing devices"""
    device_info = {
        "detected_device": detect_best_device(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_device_count"] = torch.cuda.device_count()
    
    return device_info

def detect_best_device():
    """Detect the best available device for TabPFN"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Apple Silicon GPU (MPS) available")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected)")
    
    return device

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found in data")
        
        # Apply column type conversions
        logger.info(f"Column types received: {request.column_types}")
        
        for col in df.columns:
            col_type = request.column_types.get(col, 'auto')
            
            # Auto-detect numeric columns if not specified
            if col_type == 'auto':
                sample_values = df[col].dropna().head(10)
                numeric_count = sum(1 for val in sample_values if str(val).replace('.', '').replace('-', '').isdigit())
                if numeric_count / len(sample_values) > 0.8:  # If 80% look numeric
                    col_type = 'numeric'
                    logger.info(f"Auto-detected column '{col}' as numeric")
            
            if col_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Converted column '{col}' to numeric")
            elif col_type == 'integer':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                logger.info(f"Converted column '{col}' to integer")
            elif col_type == 'categorical':
                df[col] = df[col].astype('category')
                logger.info(f"Converted column '{col}' to categorical")
            # 'text' columns stay as-is
        
        # Remove excluded columns
        columns_to_exclude = set(request.excluded_columns + [request.target_column])
        feature_columns = [col for col in df.columns if col not in columns_to_exclude]
        
        logger.info(f"Feature columns after exclusion: {feature_columns}")
        logger.info(f"Excluded columns: {request.excluded_columns}")
        
        # Separate features and target
        X = df[feature_columns]
        y = df[request.target_column]
        
        # Import TabPFN models
        from tabpfn import TabPFNClassifier
        try:
            from tabpfn import TabPFNRegressor
            tabpfn_regressor_available = True
        except ImportError:
            tabpfn_regressor_available = False
            logger.warning("TabPFNRegressor not available, will convert regression to classification")
        
        # Detect best available device
        auto_device = detect_best_device()
        
        # Get model configuration with defaults
        config = {
            'test_size': 0.2,
            'random_state': 42,
            # TabPFN specific parameters
            'N_ensemble_configurations': 16,  # Number of ensemble configurations
            'device': auto_device,  # Use best available device
            'max_samples': 1000,  # Maximum samples for TabPFN (can be increased but may cause memory issues)
            **request.model_config_params
        }
        
        # Override device if user specifically requested one
        if 'device' in request.model_config_params and request.model_config_params['device'] != 'auto':
            config['device'] = request.model_config_params['device']
            logger.info(f"User override: using device {config['device']}")
        else:
            config['device'] = auto_device  # Make sure we use the detected device
            logger.info(f"Auto-detected device: {config['device']}")
        
        logger.info(f"Model configuration: {config}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['test_size'], 
            random_state=config['random_state']
        )
        
        # Handle categorical and text features with one-hot encoding
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        categorical_cols = []
        encoders = {}
        
        # Identify categorical columns
        for col in X_train.columns:
            col_type = request.column_types.get(col, 'auto')
            if X_train[col].dtype in ['object', 'category'] or col_type in ['categorical', 'text']:
                categorical_cols.append(col)
        
        logger.info(f"Categorical columns for one-hot encoding: {categorical_cols}")
        
        if categorical_cols:
            # Apply one-hot encoding
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            
            # Fit on training data
            X_train_categorical = ohe.fit_transform(X_train[categorical_cols])
            X_test_categorical = ohe.transform(X_test[categorical_cols])
            
            # Get feature names
            feature_names = ohe.get_feature_names_out(categorical_cols)
            
            # Create DataFrames for encoded categorical features
            X_train_cat_df = pd.DataFrame(X_train_categorical, columns=feature_names, index=X_train.index)
            X_test_cat_df = pd.DataFrame(X_test_categorical, columns=feature_names, index=X_test.index)
            
            # Drop original categorical columns and concatenate encoded ones
            X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_cat_df], axis=1)
            X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_cat_df], axis=1)
            
            encoders['onehot'] = ohe
            logger.info(f"Applied one-hot encoding, resulting in {len(feature_names)} new features")
        
        # Handle any remaining non-numeric columns with label encoding (fallback)
        for col in X_train.columns:
            if X_train[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                encoders[col] = le
                logger.info(f"Applied label encoding to column: {col}")
        
        # Handle target encoding based on column type
        target_encoder = None
        target_type = request.column_types.get(request.target_column, 'auto')
        
        logger.info(f"Target column '{request.target_column}' type: {target_type}, dtype: {y_train.dtype}")
        
        # Check if target should be treated as categorical
        if target_type in ['categorical', 'text']:
            target_encoder = LabelEncoder()
            y_train = target_encoder.fit_transform(y_train.astype(str))
            y_test = target_encoder.transform(y_test.astype(str))
            classes = target_encoder.classes_.tolist()
            logger.info(f"Target encoded as categorical with classes: {classes}")
        elif y_train.dtype in ['object', 'category'] and target_type == 'auto':
            # Only use LabelEncoder for string targets if they appear to be categorical
            unique_values = y.unique()
            if len(unique_values) <= 20:  # Assume categorical if <= 20 unique values
                target_encoder = LabelEncoder()
                y_train = target_encoder.fit_transform(y_train.astype(str))
                y_test = target_encoder.transform(y_test.astype(str))
                classes = target_encoder.classes_.tolist()
                logger.info(f"Auto-detected target as categorical with classes: {classes}")
            else:
                raise HTTPException(status_code=400, detail=f"Target column appears to have too many unique values ({len(unique_values)}) for classification. Please specify the column type.")
        else:
            # Numeric target - determine if classification or regression
            unique_values = y.unique()
            if len(unique_values) <= 10:
                # Treat as classification with numeric labels
                classes = sorted(unique_values.tolist())
                logger.info(f"Numeric target with few unique values ({len(unique_values)}), treating as classification")
            else:
                # Treat as regression
                classes = []
                logger.info(f"Numeric target with {len(unique_values)} unique values, treating as regression")
        
        # Determine if this is classification or regression
        target_type = request.column_types.get(request.target_column, 'auto')
        # Regression if: numeric target AND (no encoder OR empty classes list)
        is_regression = target_type in ['numeric', 'integer'] and target_encoder is None and (len(classes) == 0 or len(y.unique()) > 10)
        is_classification = not is_regression
        
        logger.info(f"Task type: {'Regression' if is_regression else 'Classification'}")
        
        # Initialize variables that might be used in regression
        bin_centers = None
        
        if is_classification:
            # Check TabPFN constraints
            n_samples, n_features = X_train.shape
            n_classes = len(classes)
            
            # TabPFN constraints: configurable max samples, max 100 features, max 10 classes
            max_samples = config['max_samples']
            if n_samples > max_samples:
                logger.warning(f"Dataset has {n_samples} samples, using max {max_samples}. Using subset.")
                # Use stratified sampling to maintain class distribution
                from sklearn.model_selection import train_test_split as tts_subset
                X_train, _, y_train, _ = tts_subset(
                    X_train, y_train, 
                    train_size=max_samples, 
                    stratify=y_train,
                    random_state=config['random_state']
                )
                n_samples = max_samples
                
            if n_features > 100:
                raise HTTPException(
                    status_code=400, 
                    detail=f"TabPFN supports max 100 features, but dataset has {n_features}. Please exclude more columns."
                )
                
            if n_classes > 10:
                raise HTTPException(
                    status_code=400, 
                    detail=f"TabPFN supports max 10 classes, but target has {n_classes} unique values."
                )
            
            logger.info(f"TabPFN training: {n_samples} samples, {n_features} features, {n_classes} classes")
            
            # Train TabPFN classifier
            device_to_use = config['device']
            
            # Try with configured device first, then fallback to CPU if needed
            for attempt in range(2):
                try:
                    if attempt == 0:
                        model = TabPFNClassifier(device=device_to_use)
                        logger.info(f"TabPFNClassifier initialized with device: {device_to_use}")
                    else:
                        # Second attempt with CPU
                        device_to_use = 'cpu'
                        model = TabPFNClassifier(device=device_to_use)
                        logger.info("TabPFNClassifier fallback to CPU")
                    
                    # Try to fit - this is where MPS error might occur
                    model.fit(X_train, y_train)
                    logger.info(f"TabPFNClassifier training successful on {device_to_use}")
                    break  # Success, exit loop
                    
                except Exception as e:
                    if attempt == 0 and ("MPS" in str(e) or "Memory estimation" in str(e)):
                        logger.warning(f"MPS device issue during fit, will retry with CPU: {e}")
                        continue  # Try again with CPU
                    elif attempt == 1:
                        # CPU also failed, raise the error
                        raise e
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC data for binary classification
            roc_data = None
            if len(classes) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc = roc_auc_score(y_test, y_proba)
                    roc_data = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": float(auc)
                    }
                    logger.info(f"ROC AUC: {auc:.4f}")
                except Exception as e:
                    logger.warning(f"Could not compute ROC curve: {e}")
                    pass
        else:
            # Regression task
            n_samples, n_features = X_train.shape
            
            # Check TabPFN constraints for regression
            max_samples = config['max_samples']
            if n_samples > max_samples:
                logger.warning(f"Dataset has {n_samples} samples, using max {max_samples}. Using subset.")
                from sklearn.model_selection import train_test_split as tts_subset
                X_train, _, y_train, _ = tts_subset(
                    X_train, y_train, 
                    train_size=max_samples, 
                    random_state=config['random_state']
                )
                n_samples = max_samples
                
            if n_features > 100:
                raise HTTPException(
                    status_code=400, 
                    detail=f"TabPFN supports max 100 features, but dataset has {n_features}. Please exclude more columns."
                )
            
            logger.info(f"TabPFN regression: {n_samples} samples, {n_features} features")
            
            if tabpfn_regressor_available:
                # Use TabPFNRegressor if available
                device_to_use = config['device']
                
                # Try with configured device first
                for attempt in range(2):
                    try:
                        if attempt == 0:
                            model = TabPFNRegressor(device=device_to_use)
                            logger.info(f"TabPFNRegressor initialized with device: {device_to_use}")
                        else:
                            # Second attempt with CPU
                            device_to_use = 'cpu'
                            model = TabPFNRegressor(device=device_to_use)
                            logger.info("TabPFNRegressor fallback to CPU")
                        
                        # Try to fit - this is where MPS error occurs
                        model.fit(X_train, y_train)
                        logger.info(f"TabPFNRegressor training successful on {device_to_use}")
                        break  # Success, exit loop
                        
                    except Exception as e:
                        if attempt == 0 and ("MPS" in str(e) or "Memory estimation" in str(e)):
                            logger.warning(f"MPS device issue during fit, will retry with CPU: {e}")
                            continue  # Try again with CPU
                        elif attempt == 1:
                            # CPU also failed, raise the error
                            raise e
                
                # Predict with fallback handling
                try:
                    y_pred = model.predict(X_test)
                except Exception as e:
                    if "MPS" in str(e) or "Memory estimation" in str(e):
                        logger.warning(f"MPS device issue during predict, recreating model with CPU: {e}")
                        # Recreate model with CPU and refit
                        model = TabPFNRegressor(device='cpu')
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        logger.info("Prediction successful on CPU after MPS failure")
                    else:
                        raise e
                
                # Calculate regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"Regression metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                # For regression, we'll adapt the response format
                accuracy = r2  # Use R² as "accuracy" for regression
                precision = [mae]  # Use MAE as precision
                recall = [mse]  # Use MSE as recall
                f1 = [r2]  # Use R² as F1
                cm = [[0]]  # No confusion matrix for regression
                classes = []  # No classes for regression
                roc_data = None  # No ROC for regression
                
            else:
                # Convert regression to classification by binning
                logger.info("TabPFNRegressor not available, converting to classification bins")
                
                # Create bins for the target variable
                n_bins = min(10, len(y.unique()))
                y_binned = pd.cut(y, bins=n_bins, labels=False)
                y_train_binned = pd.cut(y_train, bins=n_bins, labels=False)
                y_test_binned = pd.cut(y_test, bins=n_bins, labels=False)
                
                # Use classification approach
                try:
                    model = TabPFNClassifier(device=config['device'])
                    logger.info(f"TabPFNClassifier initialized for binned regression with device: {config['device']}")
                except Exception as e:
                    logger.warning(f"Could not set device, using default: {e}")
                    model = TabPFNClassifier()
                
                model.fit(X_train, y_train_binned)
                y_pred_binned = model.predict(X_test)
                
                # Convert back to continuous values (use bin centers)
                bin_edges = pd.cut(y, bins=n_bins, retbins=True)[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                y_pred = bin_centers[y_pred_binned]
                
                # Calculate regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"Binned regression metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                
                # Adapt response format
                accuracy = r2
                precision = [mae]
                recall = [mse]
                f1 = [r2]
                cm = [[0]]
                classes = []
                roc_data = None
        
        # Prepare full dataset predictions
        X_full = X.copy()
        
        # Apply one-hot encoding if it was used
        if 'onehot' in encoders:
            ohe = encoders['onehot']
            if categorical_cols:
                # Apply one-hot encoding to categorical columns
                X_full_categorical = ohe.transform(X_full[categorical_cols])
                feature_names = ohe.get_feature_names_out(categorical_cols)
                X_full_cat_df = pd.DataFrame(X_full_categorical, columns=feature_names, index=X_full.index)
                
                # Drop original categorical columns and concatenate encoded ones
                X_full = pd.concat([X_full.drop(columns=categorical_cols), X_full_cat_df], axis=1)
        
        # Apply label encoding to remaining columns
        for col, encoder in encoders.items():
            if col != 'onehot' and col in X_full.columns:
                X_full[col] = encoder.transform(X_full[col].astype(str))
        
        # Make predictions on full dataset with MPS fallback handling
        try:
            if is_regression and not tabpfn_regressor_available and bin_centers is not None:
                # For binned regression, need to convert predictions back to continuous
                pred_binned = model.predict(X_full)
                predictions = bin_centers[pred_binned]
            else:
                predictions = model.predict(X_full)
                if target_encoder is not None:
                    predictions = target_encoder.inverse_transform(predictions)
        except Exception as e:
            if "MPS" in str(e) or "Memory estimation" in str(e):
                logger.warning(f"MPS device issue during full prediction, recreating model with CPU: {e}")
                # Need to recreate and refit the model with CPU
                if is_regression and tabpfn_regressor_available:
                    model = TabPFNRegressor(device='cpu')
                    model.fit(X_train, y_train)
                else:  # is_classification
                    model = TabPFNClassifier(device='cpu')
                    model.fit(X_train, y_train)
                
                # Try predictions again
                if is_regression and not tabpfn_regressor_available and bin_centers is not None:
                    pred_binned = model.predict(X_full)
                    predictions = bin_centers[pred_binned]
                else:
                    predictions = model.predict(X_full)
                    if target_encoder is not None:
                        predictions = target_encoder.inverse_transform(predictions)
                
                logger.info("Full dataset prediction successful on CPU after MPS failure")
            else:
                raise e
        
        # Serialize model and metadata for download
        model_package = {
            'model': model,
            'encoders': encoders,
            'target_encoder': target_encoder,
            'feature_columns': X.columns.tolist(),
            'target_column': request.target_column,
            'column_types': request.column_types,
            'classes': classes
        }
        
        # Serialize to base64 for transport
        buffer = BytesIO()
        pickle.dump(model_package, buffer)
        buffer.seek(0)
        model_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        model_metadata = {
            'model_type': 'TabPFNClassifier',
            'feature_count': len(X.columns),
            'target_column': request.target_column,
            'classes': classes,
            'accuracy': float(accuracy),
            'n_ensemble_configurations': config['N_ensemble_configurations'],
            'device': config['device'],
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # Convert predictions to list if it's a numpy array
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = predictions if isinstance(predictions, list) else list(predictions)
        
        return PredictResponse(
            predictions=predictions_list,
            accuracy=float(accuracy),
            precision=precision.tolist() if hasattr(precision, 'tolist') else precision,
            recall=recall.tolist() if hasattr(recall, 'tolist') else recall,
            f1_score=f1.tolist() if hasattr(f1, 'tolist') else f1,
            confusion_matrix=cm.tolist() if hasattr(cm, 'tolist') else cm,
            classes=classes,
            roc_data=roc_data,
            model_data=model_data,
            model_metadata=model_metadata
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