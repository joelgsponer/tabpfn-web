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
import warnings

# Suppress NumPy FutureWarning about random seeding (common in SHAP/TabPFN)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.random.seed.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SHAP related imports
try:
    import shap
    import shapiq
    from tabpfn_extensions.interpretability.shapiq import get_tabpfn_explainer
    SHAP_AVAILABLE = True
    logger.info("SHAP libraries loaded successfully")
except ImportError as e:
    logger.warning(f"SHAP libraries not available: {e}")
    SHAP_AVAILABLE = False

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
    model_config = {"protected_namespaces": ()}
    
    model_data: str  # Base64 encoded model package
    data: List[Dict[str, Any]]  # Full dataset for background context
    instance_index: Optional[int] = None  # Which instance to explain (default: first)
    
class ShapResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    shap_values: Dict[str, float]  # Feature name to SHAP value mapping
    baseline_value: float
    predicted_value: float
    feature_names: List[str]
    instance_values: Dict[str, Any]  # Actual feature values for the explained instance
    interpretation: str  # Text interpretation of the results

class ShapOverviewRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_data: str  # Base64 encoded model package
    data: List[Dict[str, Any]]  # Full dataset for background context
    num_instances: Optional[int] = 20  # Number of instances to compute SHAP for
    
class ShapOverviewResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    feature_names: List[str]
    shap_values_matrix: List[List[float]]  # Each row is SHAP values for one instance
    feature_values_matrix: List[List[Any]]  # Each row is feature values for one instance
    baseline_value: float
    mean_abs_shap: Dict[str, float]  # Mean absolute SHAP value per feature
    summary_plot_data: Dict[str, Any]  # Data for creating summary plots

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
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Apple Silicon GPU (MPS) available for TabPFN and SHAP computation")
    else:
        device = 'cpu'
        logger.info("Using CPU (no GPU detected) - SHAP computation may be slower")
    
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
            # Check cardinality of categorical columns to avoid feature explosion
            high_cardinality_cols = []
            for col in categorical_cols:
                unique_count = X_train[col].nunique()
                if unique_count > 20:  # Limit to 20 unique values per categorical column
                    high_cardinality_cols.append(col)
                    logger.warning(f"Column '{col}' has {unique_count} unique values, will use label encoding instead")
            
            # Remove high cardinality columns from one-hot encoding
            cols_for_onehot = [col for col in categorical_cols if col not in high_cardinality_cols]
            logger.info(f"Columns for one-hot encoding (after filtering): {cols_for_onehot}")
            
            # Estimate final feature count
            current_numeric_features = len(X_train.columns) - len(categorical_cols)
            estimated_onehot_features = sum(min(X_train[col].nunique() - 1, 19) for col in cols_for_onehot)  # -1 for drop='first'
            estimated_label_encoded = len(high_cardinality_cols)
            total_estimated_features = current_numeric_features + estimated_onehot_features + estimated_label_encoded
            
            logger.info(f"Feature count estimation: {current_numeric_features} numeric + {estimated_onehot_features} one-hot + {estimated_label_encoded} label = {total_estimated_features} total")
            
            if total_estimated_features > 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset would result in {total_estimated_features} features after encoding, but TabPFN supports max 100 features. Please exclude more columns or reduce categorical feature cardinality."
                )
            
            # Apply one-hot encoding to low-cardinality columns
            if cols_for_onehot:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                
                # Fit on training data
                X_train_categorical = ohe.fit_transform(X_train[cols_for_onehot])
                X_test_categorical = ohe.transform(X_test[cols_for_onehot])
                
                # Get feature names
                feature_names = ohe.get_feature_names_out(cols_for_onehot)
                
                # Create DataFrames for encoded categorical features
                X_train_cat_df = pd.DataFrame(X_train_categorical, columns=feature_names, index=X_train.index)
                X_test_cat_df = pd.DataFrame(X_test_categorical, columns=feature_names, index=X_test.index)
                
                # Drop original one-hot encoded columns and concatenate encoded ones
                X_train = pd.concat([X_train.drop(columns=cols_for_onehot), X_train_cat_df], axis=1)
                X_test = pd.concat([X_test.drop(columns=cols_for_onehot), X_test_cat_df], axis=1)
                
                encoders['onehot'] = ohe
                encoders['onehot_columns'] = cols_for_onehot  # Store which columns were one-hot encoded
                logger.info(f"Applied one-hot encoding, resulting in {len(feature_names)} new features")
            
            # Apply label encoding to high-cardinality columns
            for col in high_cardinality_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                encoders[col] = le
                logger.info(f"Applied label encoding to high-cardinality column: {col}")
        
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
                        # Configure model parameters based on device
                        model_params = {
                            'device': device_to_use,
                            'random_state': config['random_state'],
                            'N_ensemble_configurations': config['N_ensemble_configurations']
                        }
                        
                        # Add additional parameters from demo script if available
                        if 'n_estimators' in config:
                            model_params['n_estimators'] = config['n_estimators']
                        if 'n_jobs' in config:
                            model_params['n_jobs'] = config['n_jobs']
                        
                        # Add MPS-specific optimizations
                        if device_to_use == 'mps':
                            model_params.update({
                                'memory_saving_mode': True,
                                'ignore_pretraining_limits': True
                            })
                            logger.info("Using MPS-specific optimizations: memory_saving_mode=True, ignore_pretraining_limits=True")
                        
                        model = TabPFNClassifier(**model_params)
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
            
            # Make predictions with MPS fallback
            try:
                y_pred = model.predict(X_test)
            except Exception as e:
                if "MPS" in str(e) or "Memory estimation" in str(e):
                    logger.warning(f"MPS device issue during predict, recreating model with CPU: {e}")
                    # Recreate model with CPU and refit
                    model = TabPFNClassifier(device='cpu')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    device_to_use = 'cpu'  # Update device tracking
                    logger.info("Prediction successful on CPU after MPS failure")
                else:
                    raise e
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC data for binary classification with MPS fallback
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
                    if "MPS" in str(e) or "Memory estimation" in str(e):
                        logger.warning(f"MPS device issue during predict_proba, using CPU model: {e}")
                        # Model should already be CPU from above fallback
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            auc = roc_auc_score(y_test, y_proba)
                            roc_data = {
                                "fpr": fpr.tolist(),
                                "tpr": tpr.tolist(),
                                "auc": float(auc)
                            }
                            logger.info(f"ROC AUC (CPU): {auc:.4f}")
                        except Exception as e2:
                            logger.warning(f"Could not compute ROC curve even with CPU: {e2}")
                    else:
                        logger.warning(f"Could not compute ROC curve: {e}")
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
                            # Configure model parameters based on device
                            model_params = {
                                'device': device_to_use,
                                'random_state': config['random_state']
                            }
                            
                            # Add additional parameters from demo script if available
                            if 'n_estimators' in config:
                                model_params['n_estimators'] = config['n_estimators']
                            if 'n_jobs' in config:
                                model_params['n_jobs'] = config['n_jobs']
                            
                            # Add MPS-specific optimizations
                            if device_to_use == 'mps':
                                model_params.update({
                                    'memory_saving_mode': True,
                                    'ignore_pretraining_limits': True
                                })
                                logger.info("Using MPS-specific optimizations for regression: memory_saving_mode=True, ignore_pretraining_limits=True")
                            
                            model = TabPFNRegressor(**model_params)
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
            onehot_columns = encoders.get('onehot_columns', [])
            if onehot_columns:
                cols_to_encode = [col for col in onehot_columns if col in X_full.columns]
                if cols_to_encode:
                    # Apply one-hot encoding to the same columns as training
                    X_full_categorical = ohe.transform(X_full[cols_to_encode])
                    feature_names = ohe.get_feature_names_out(cols_to_encode)
                    X_full_cat_df = pd.DataFrame(X_full_categorical, columns=feature_names, index=X_full.index)
                    
                    # Drop original categorical columns and concatenate encoded ones
                    X_full = pd.concat([X_full.drop(columns=cols_to_encode), X_full_cat_df], axis=1)
        
        # Apply label encoding to remaining columns  
        for col, encoder in encoders.items():
            if col not in ['onehot', 'onehot_columns'] and col in X_full.columns:
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

@app.post("/shap", response_model=ShapResponse)
async def compute_shap(request: ShapRequest):
    try:
        if not SHAP_AVAILABLE:
            raise HTTPException(
                status_code=501, 
                detail="SHAP libraries not available. Please install: pip install shap shapiq tabpfn-extensions"
            )
        
        logger.info("Starting SHAP computation...")
        
        # Decode model package
        try:
            model_bytes = base64.b64decode(request.model_data)
            buffer = BytesIO(model_bytes)
            model_package = pickle.load(buffer)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode model data: {str(e)}")
        
        # Extract components from model package
        model = model_package['model']
        encoders = model_package['encoders']
        target_encoder = model_package.get('target_encoder')
        feature_columns = model_package['feature_columns']
        target_column = model_package['target_column']
        column_types = model_package.get('column_types', {})
        classes = model_package.get('classes', [])
        
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Apply same preprocessing as training
        logger.info("Applying preprocessing for SHAP...")
        
        # Apply column type conversions
        for col in df.columns:
            col_type = column_types.get(col, 'auto')
            if col_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == 'integer':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif col_type == 'categorical':
                df[col] = df[col].astype('category')
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column] if target_column in df.columns else None
        
        # Apply same encoding as training
        # First, identify categorical columns using the SAME logic as training
        categorical_cols = []
        for col in X.columns:
            col_type = column_types.get(col, 'auto')
            if X[col].dtype in ['object', 'category'] or col_type in ['categorical', 'text']:
                categorical_cols.append(col)
        
        logger.info(f"SHAP: Categorical columns detected: {categorical_cols}")
        logger.info(f"SHAP: OneHot encoder available: {'onehot' in encoders}")
        
        # Apply one-hot encoding if it was used during training
        if 'onehot' in encoders:
            ohe = encoders['onehot']
            # Get the columns that were one-hot encoded during training
            onehot_columns = encoders.get('onehot_columns', [])
            logger.info(f"SHAP: Columns that were one-hot encoded during training: {onehot_columns}")
            
            # Only apply to columns that exist in current data and were one-hot encoded during training
            cols_to_encode = [col for col in onehot_columns if col in X.columns]
            logger.info(f"SHAP: Columns to one-hot encode: {cols_to_encode}")
            
            if cols_to_encode:
                try:
                    # Apply one-hot encoding to the same columns as training
                    X_categorical = ohe.transform(X[cols_to_encode])
                    feature_names = ohe.get_feature_names_out(cols_to_encode)
                    X_cat_df = pd.DataFrame(X_categorical, columns=feature_names, index=X.index)
                    
                    # Drop original categorical columns and concatenate encoded ones
                    X = pd.concat([X.drop(columns=cols_to_encode), X_cat_df], axis=1)
                    logger.info(f"SHAP: Applied one-hot encoding, resulting features: {X.columns.tolist()}")
                except Exception as e:
                    logger.warning(f"SHAP: One-hot encoding failed: {e}")
                    # Continue without one-hot encoding
        
        # Apply label encoding to remaining columns
        for col, encoder in encoders.items():
            if col not in ['onehot', 'onehot_columns'] and col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))
        
        # Add debugging for feature matrix
        logger.info(f"SHAP: Final feature matrix shape: {X.shape}")
        logger.info(f"SHAP: Feature columns: {X.columns.tolist()}")
        logger.info(f"SHAP: Feature matrix sample:\n{X.head(3)}")
        
        # Check for constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.warning(f"SHAP: Found constant features: {constant_features}")
            # If ALL features are constant, this is a serious issue
            if len(constant_features) == len(X.columns):
                raise HTTPException(
                    status_code=400,
                    detail=f"All features are constant after preprocessing. This usually indicates a data preprocessing issue. Constant features: {constant_features}"
                )
        
        # Select instance to explain
        instance_index = request.instance_index or 0
        if instance_index >= len(X):
            raise HTTPException(
                status_code=400, 
                detail=f"Instance index {instance_index} out of range. Dataset has {len(X)} samples."
            )
        
        X_instance = X.iloc[instance_index:instance_index+1]
        
        # Use a larger background set and ensure it has variability
        background_size = min(100, len(X))
        # Try to get a diverse background by sampling if dataset is large enough
        if len(X) > background_size:
            # Sample background data to ensure diversity
            background_indices = np.random.choice(len(X), size=background_size, replace=False)
            X_background = X.iloc[background_indices]
        else:
            X_background = X.iloc[:background_size]
        
        logger.info(f"SHAP: Instance to explain shape: {X_instance.shape}")
        logger.info(f"SHAP: Background data shape: {X_background.shape}")
        
        logger.info(f"Explaining instance {instance_index} with background of {len(X_background)} samples")
        
        # Get model prediction for the instance
        try:
            if hasattr(model, 'predict_proba'):
                # Classification
                prediction = model.predict_proba(X_instance)[0]
                if len(classes) == 2:
                    predicted_value = float(prediction[1])  # Probability of positive class
                    class_index = 1
                else:
                    predicted_value = float(np.max(prediction))
                    class_index = int(np.argmax(prediction))
            else:
                # Regression
                prediction = model.predict(X_instance)
                predicted_value = float(prediction[0])
                class_index = None
        except Exception as e:
            logger.warning(f"Prediction failed, using fallback: {e}")
            predicted_value = 0.0
            class_index = None
        
        # Apply target encoding if it was used during training
        y_background = None
        if y is not None and target_encoder is not None:
            y_background = target_encoder.transform(y[:len(X_background)].astype(str))
        elif y is not None:
            y_background = y[:len(X_background)].values
        
        # Get feature names for SHAP computation
        feature_names = X.columns.tolist()
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        try:
            explainer = get_tabpfn_explainer(
                model=model,
                data=X_background.values,
                labels=y_background if y_background is not None else np.zeros(len(X_background)),
                index="SV",  # Standard Shapley Values - most reliable option
                class_index=class_index if class_index is not None else None
            )
            
            # Compute SHAP values
            budget = min(1000, 2**len(feature_names))  # Adaptive budget based on number of features
            # Detect current device for logging
            current_device = detect_best_device()
            logger.info(f"Computing SHAP values for instance {instance_index} with budget {budget} on device: {current_device}")
            
            import time
            start_time = time.time()
            shap_result = explainer.explain(X_instance.values, budget=budget)
            computation_time = time.time() - start_time
            logger.info(f"SHAP computation completed in {computation_time:.2f}s")
            
            # Extract SHAP values and create feature mapping
            shap_values_dict = {}
            
            if hasattr(shap_result, 'dict_values'):
                # shapiq format - dict_values contains tuples as keys
                baseline_value = float(shap_result.dict_values.get((), 0.0))  # Empty tuple is baseline
                
                # Extract individual feature contributions
                for key, shap_val in shap_result.dict_values.items():
                    if isinstance(key, tuple) and len(key) == 1:  # Single feature index
                        feature_idx = key[0]
                        if feature_idx < len(feature_names):
                            shap_values_dict[feature_names[feature_idx]] = float(shap_val)
                
                # If no baseline found in dict_values, try the baseline_value attribute
                if baseline_value == 0.0 and hasattr(shap_result, 'baseline_value'):
                    baseline_value = float(shap_result.baseline_value)
            else:
                # Fallback format
                baseline_value = float(getattr(shap_result, 'baseline_value', 0.0))
                if hasattr(shap_result, 'values'):
                    for i, feature_name in enumerate(feature_names):
                        if i < len(shap_result.values):
                            shap_values_dict[feature_name] = float(shap_result.values[i])
            
            # Get instance values for interpretation
            instance_values = {}
            original_instance = df.iloc[instance_index]
            for col in feature_columns:
                instance_values[col] = original_instance[col]
            
            # Create interpretation text
            top_features = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            interpretation_parts = []
            
            if len(classes) == 2:  # Binary classification
                interpretation_parts.append(f"Predicted probability: {predicted_value:.3f}")
            elif len(classes) > 2:  # Multi-class classification
                interpretation_parts.append(f"Predicted class: {classes[class_index] if class_index < len(classes) else class_index}")
            else:  # Regression
                interpretation_parts.append(f"Predicted value: {predicted_value:.3f}")
            
            interpretation_parts.append(f"Baseline: {baseline_value:.3f}")
            interpretation_parts.append("Top contributing features:")
            
            for feature, value in top_features:
                direction = "increases" if value > 0 else "decreases"
                interpretation_parts.append(f"- {feature}: {value:+.3f} ({direction} prediction)")
            
            interpretation = "\n".join(interpretation_parts)
            
            logger.info("SHAP computation completed successfully")
            
            return ShapResponse(
                shap_values=shap_values_dict,
                baseline_value=baseline_value,
                predicted_value=predicted_value,
                feature_names=feature_names,
                instance_values=instance_values,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            # Fallback to simple feature importance if SHAP fails
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                shap_values_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            else:
                shap_values_dict = {name: 0.0 for name in feature_names}
            
            return ShapResponse(
                shap_values=shap_values_dict,
                baseline_value=0.0,
                predicted_value=predicted_value,
                feature_names=feature_names,
                instance_values={col: df.iloc[instance_index][col] for col in feature_columns},
                interpretation=f"SHAP computation failed: {str(e)}. Showing fallback feature importance."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shap-overview", response_model=ShapOverviewResponse)
async def compute_shap_overview(request: ShapOverviewRequest):
    try:
        if not SHAP_AVAILABLE:
            raise HTTPException(
                status_code=501, 
                detail="SHAP libraries not available. Please install: pip install shap shapiq tabpfn-extensions"
            )
        
        logger.info("Starting SHAP overview computation...")
        
        # Decode model package
        try:
            model_bytes = base64.b64decode(request.model_data)
            buffer = BytesIO(model_bytes)
            model_package = pickle.load(buffer)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode model data: {str(e)}")
        
        # Extract components from model package
        model = model_package['model']
        encoders = model_package['encoders']
        target_encoder = model_package.get('target_encoder')
        feature_columns = model_package['feature_columns']
        target_column = model_package['target_column']
        column_types = model_package.get('column_types', {})
        classes = model_package.get('classes', [])
        
        # Convert request data to DataFrame and preprocess (same as individual SHAP)
        df = pd.DataFrame(request.data)
        
        # Apply same preprocessing as training
        logger.info("Applying preprocessing for SHAP overview...")
        
        # Apply column type conversions
        for col in df.columns:
            col_type = column_types.get(col, 'auto')
            if col_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == 'integer':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif col_type == 'categorical':
                df[col] = df[col].astype('category')
        
        # Prepare features
        X = df[feature_columns]
        y = df[target_column] if target_column in df.columns else None
        
        # Apply same encoding as training (reuse the preprocessing logic)
        categorical_cols = []
        for col in X.columns:
            col_type = column_types.get(col, 'auto')
            if X[col].dtype in ['object', 'category'] or col_type in ['categorical', 'text']:
                categorical_cols.append(col)
        
        # Apply one-hot encoding if it was used during training
        if 'onehot' in encoders:
            ohe = encoders['onehot']
            onehot_columns = encoders.get('onehot_columns', [])
            if onehot_columns:
                cols_to_encode = [col for col in onehot_columns if col in X.columns]
                if cols_to_encode:
                    try:
                        X_categorical = ohe.transform(X[cols_to_encode])
                        feature_names = ohe.get_feature_names_out(cols_to_encode)
                        X_cat_df = pd.DataFrame(X_categorical, columns=feature_names, index=X.index)
                        X = pd.concat([X.drop(columns=cols_to_encode), X_cat_df], axis=1)
                    except Exception as e:
                        logger.warning(f"SHAP Overview: One-hot encoding failed: {e}")
        
        # Apply label encoding to remaining columns
        for col, encoder in encoders.items():
            if col not in ['onehot', 'onehot_columns'] and col in X.columns:
                X[col] = encoder.transform(X[col].astype(str))
        
        # Get feature names for SHAP computation
        feature_names = X.columns.tolist()
        
        # Select instances to compute SHAP for
        num_instances = min(request.num_instances, len(X))  # Use requested number of instances, up to dataset size
        instance_indices = np.random.choice(len(X), size=num_instances, replace=False)
        X_instances = X.iloc[instance_indices]
        X_background = X.iloc[:min(100, len(X))]  # Background for SHAP
        
        logger.info(f"Computing SHAP for {num_instances} instances with {len(X_background)} background samples")
        
        # Apply target encoding if needed
        y_background = None
        if y is not None and target_encoder is not None:
            y_background = target_encoder.transform(y[:len(X_background)].astype(str))
        elif y is not None:
            y_background = y[:len(X_background)].values
        
        # Determine class index for classification
        class_index = None
        if len(classes) == 2:
            class_index = 1  # Explain positive class for binary classification
        elif len(classes) > 2:
            class_index = None  # Let SHAP handle multiclass
        
        # Create SHAP explainer
        try:
            explainer = get_tabpfn_explainer(
                model=model,
                data=X_background.values,
                labels=y_background if y_background is not None else np.zeros(len(X_background)),
                index="SV",  # Standard Shapley Values - most reliable option
                class_index=class_index
            )
            
            # Compute SHAP values for multiple instances
            budget = min(500, 2**len(feature_names))  # Smaller budget for multiple instances
            logger.info(f"Computing SHAP values with budget {budget} for {num_instances} instances...")
            # Detect current device for logging
            current_device = detect_best_device()
            logger.info(f"Using SV explainer on device: {current_device}")
            
            shap_results = []
            import time
            start_time = time.time()
            
            for i, instance_idx in enumerate(instance_indices):
                try:
                    iter_start = time.time()
                    X_single = X.iloc[instance_idx:instance_idx+1]
                    shap_result = explainer.explain(X_single.values, budget=budget)
                    shap_results.append(shap_result)
                    
                    iter_time = time.time() - iter_start
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    eta = avg_time * (num_instances - i - 1)
                    
                    logger.info(f"SV explainer: {i+1}it [{elapsed:.0f}s<{eta:.0f}s, {avg_time:.2f}s/it] Instance {instance_idx}")
                        
                except Exception as e:
                    logger.warning(f"Failed to compute SHAP for instance {instance_idx}: {e}")
                    continue
            
            if not shap_results:
                raise Exception("Failed to compute SHAP values for any instances")
            
            # Parse SHAP results into matrices
            shap_values_matrix = []
            feature_values_matrix = []
            baseline_values = []
            
            for i, shap_result in enumerate(shap_results):
                instance_shap = [0.0] * len(feature_names)
                
                if hasattr(shap_result, 'dict_values'):
                    baseline_values.append(float(shap_result.dict_values.get((), 0.0)))
                    
                    # Extract individual feature contributions
                    for key, shap_val in shap_result.dict_values.items():
                        if isinstance(key, tuple) and len(key) == 1:
                            feature_idx = key[0]
                            if feature_idx < len(feature_names):
                                instance_shap[feature_idx] = float(shap_val)
                
                shap_values_matrix.append(instance_shap)
                
                # Get feature values for this instance
                instance_idx = instance_indices[i]
                feature_values = []
                for feature in feature_names:
                    if feature in X.columns:
                        val = X.iloc[instance_idx][feature]
                        feature_values.append(float(val) if isinstance(val, (int, float)) else str(val))
                    else:
                        feature_values.append(0.0)
                feature_values_matrix.append(feature_values)
            
            # Calculate mean absolute SHAP values for feature importance ranking
            shap_array = np.array(shap_values_matrix)
            mean_abs_shap = {
                feature: float(np.mean(np.abs(shap_array[:, i])))
                for i, feature in enumerate(feature_names)
            }
            
            # Create summary plot data
            summary_plot_data = {
                "feature_importance_order": sorted(mean_abs_shap.keys(), key=lambda x: mean_abs_shap[x], reverse=True),
                "shap_range": {
                    feature: {
                        "min": float(np.min(shap_array[:, i])),
                        "max": float(np.max(shap_array[:, i])),
                        "mean": float(np.mean(shap_array[:, i])),
                        "std": float(np.std(shap_array[:, i]))
                    }
                    for i, feature in enumerate(feature_names)
                }
            }
            
            baseline_value = float(np.mean(baseline_values)) if baseline_values else 0.0
            
            logger.info(f"SHAP overview computation completed for {len(shap_values_matrix)} instances")
            
            return ShapOverviewResponse(
                feature_names=feature_names,
                shap_values_matrix=shap_values_matrix,
                feature_values_matrix=feature_values_matrix,
                baseline_value=baseline_value,
                mean_abs_shap=mean_abs_shap,
                summary_plot_data=summary_plot_data
            )
            
        except Exception as e:
            logger.error(f"SHAP overview computation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"SHAP overview computation failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP overview endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)