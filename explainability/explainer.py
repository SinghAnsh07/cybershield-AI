import os
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class AnomalyExplainer:
    def __init__(self, classifier_model):
        self.classifier_model = classifier_model
        # Get the underlying booster if it's an XGBClassifier
        self.booster = getattr(classifier_model, "get_booster", lambda: classifier_model)()
        try:
            self.explainer = shap.TreeExplainer(self.booster)
        except Exception as e:
            self.explainer = None
            print(f"Warning: Could not initialize SHAP explainer: {e}")

    def explain(self, features_df, feature_names=None):
        if feature_names is None:
            feature_names = features_df.columns.tolist() if isinstance(features_df, pd.DataFrame) else config.FEATURE_NAMES
            
        if not isinstance(features_df, pd.DataFrame):
            features_df = pd.DataFrame(features_df, columns=feature_names)

        results = []
        try:
            shap_values = self.explainer.shap_values(features_df)
            if isinstance(shap_values, list):
                # multi-class: take the class with max prob for explanation
                preds = self.classifier_model.predict(features_df)
                shap_values_to_use = []
                for i, pred_idx in enumerate(preds):
                    shap_values_to_use.append(shap_values[pred_idx][i])
                shap_values_to_use = np.array(shap_values_to_use)
            elif len(shap_values.shape) == 3:
                preds = self.classifier_model.predict(features_df)
                shap_values_to_use = []
                for i, pred_idx in enumerate(preds):
                    shap_values_to_use.append(shap_values[i, :, pred_idx])
                shap_values_to_use = np.array(shap_values_to_use)
            else:
                shap_values_to_use = shap_values
                
            for i in range(len(features_df)):
                row_shap = shap_values_to_use[i]
                row_vals = features_df.iloc[i]
                
                shap_dict = {name: val for name, val in zip(feature_names, row_shap)}
                
                # Top K features
                sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                top_features = [f[0] for f in sorted_features[:config.TOP_K_FEATURES]]
                
                # Reason string
                reasons = []
                for f_name in top_features:
                    f_val = row_vals[f_name]
                    s_val = shap_dict[f_name]
                    reasons.append(self.generate_reason_string(f_name, f_val, s_val))
                reason_string = "Flagged due to: " + "; ".join(reasons)
                
                results.append({
                    "shap_values": shap_dict,
                    "top_features": top_features,
                    "reason_string": reason_string
                })
        except Exception as e:
            # Fallback to feature importance from model
            try:
                importances = self.booster.get_score(importance_type='weight')
                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                top_features = [f[0] for f in sorted_imp[:config.TOP_K_FEATURES]]
            except:
                top_features = feature_names[:config.TOP_K_FEATURES]
                
            for i in range(len(features_df)):
                row_vals = features_df.iloc[i]
                reasons = []
                shap_dict = {}
                for f_name in top_features:
                    f_val = row_vals.get(f_name, 0)
                    shap_dict[f_name] = 0.0 # fallback dummy
                    reasons.append(self.generate_reason_string(f_name, f_val, 0.0))
                
                reason_string = "Flagged due to: " + "; ".join(reasons)
                results.append({
                    "shap_values": shap_dict,
                    "top_features": top_features,
                    "reason_string": reason_string
                })
                
        return results

    def generate_reason_string(self, feature_name, feature_value, shap_value):
        if feature_name in config.FEATURE_REASON_MAP:
            template = config.FEATURE_REASON_MAP[feature_name]
            desc = "short" if feature_value < 10 else "long"
            try:
                # Formatting may need different keys
                return template.format(val=feature_value, desc=desc)
            except:
                # Fallback replacement if format string vars don't match
                s = template.replace("{val:.0f}", str(int(feature_value)))
                s = s.replace("{val:.1f}", f"{feature_value:.1f}")
                s = s.replace("{desc}", desc)
                return s
        else:
            return f"elevated {feature_name} ({feature_value})"

    def get_alert_explanation(self, feature_vector, feature_names):
        df = pd.DataFrame([feature_vector], columns=feature_names)
        explanation = self.explain(df, feature_names)[0]
        
        try:
            preds = self.classifier_model.predict(df)[0]
            probas = self.classifier_model.predict_proba(df)[0]
            conf = float(np.max(probas))
        except:
            preds = 0
            conf = 1.0
            
        attack_type = config.IDX_TO_LABEL.get(preds, "unknown")
        
        return {
            "risk_factors": explanation["shap_values"],
            "attack_type": attack_type,
            "confidence": conf,
            "reason_string": explanation["reason_string"]
        }

    def plot_explanation(self, feature_vector, feature_names, save_path=None):
        df = pd.DataFrame([feature_vector], columns=feature_names)
        if not self.explainer:
            print("Explainer not initialized. Cannot plot.")
            return
        
        shap_values = self.explainer(df)
        try:
            preds = self.classifier_model.predict(df)[0]
            if isinstance(shap_values.values, list):
                sv = shap_values[preds]
            elif len(shap_values.shape) == 3:
                sv = shap_values[:, :, preds]
            else:
                sv = shap_values
                
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(sv[0], show=False)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print(f"Could not generate plot: {e}")
