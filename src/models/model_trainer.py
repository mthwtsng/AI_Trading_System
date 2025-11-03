import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_models(X_train, y_train):
    logger = logging.getLogger('TradingBot')
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    logger.info(f"Original features: {X_train.shape[1]}, After PCA: {X_pca.shape[1]}")
    
    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train individual models
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_pca, y_train)
    
    # Create ensemble model
    ensemble_models = [(name, model) for name, model in models.items()]
    models["Ensemble"] = VotingClassifier(estimators=ensemble_models, voting='soft')
    models["Ensemble"].fit(X_pca, y_train)
    
    logger.info("All models trained successfully")
    return models, scaler, pca
