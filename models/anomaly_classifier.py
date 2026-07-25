import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class AnomalyClassifier:
    def __init__(self):
        self.model = XGBClassifier(**config.XGB_PARAMS)
        
    def fit(self, features_df, labels):
        # Handle both Series and array labels
        if hasattr(labels, 'map'):
            y = labels.map(config.LABEL_TO_IDX).values
        else:
            y = np.array([config.LABEL_TO_IDX.get(l, 0) for l in labels])
        X = features_df.values if hasattr(features_df, 'values') else features_df
        
        class_counts = pd.Series(y).value_counts()
        
        try:
            if class_counts.min() > 5:
                smote = SMOTE(random_state=config.RANDOM_SEED)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            else:
                X_resampled, y_resampled = X, y
        except Exception:
            X_resampled, y_resampled = X, y
            
        resampled_counts = pd.Series(y_resampled).value_counts()
        total_samples = len(y_resampled)
        n_classes = len(resampled_counts)
        
        weights_dict = {cls: total_samples / (n_classes * count) for cls, count in resampled_counts.items()}
        sample_weights = np.array([weights_dict[lbl] for lbl in y_resampled])
        
        self.model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
        
    def predict(self, features_df):
        X = features_df.values if hasattr(features_df, 'values') else features_df
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)
        
        pred_labels = [config.IDX_TO_LABEL[p] for p in preds]
        confidence_scores = []
        for i, p in enumerate(preds):
            confidence_scores.append(probas[i][p])
            
        return np.array(pred_labels), np.array(confidence_scores)
        
    def predict_proba(self, features_df):
        X = features_df.values if hasattr(features_df, 'values') else features_df
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importances = []
        for idx in indices:
            feature_importances.append({
                "feature": config.FEATURE_NAMES[idx],
                "importance": float(importances[idx])
            })
            
        return feature_importances
        
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path.with_suffix('.joblib'))
        
    def load(self, path):
        path = Path(path)
        self.model = joblib.load(path.with_suffix('.joblib'))
