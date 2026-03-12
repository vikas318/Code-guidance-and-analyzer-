import os
import joblib
import json
import pandas as pd
from AST import parse_code, CodeAnalyzer
from dataset_preprocessor import extract_feature_vector, get_feature_names
from complexity_analyzer import TimeComplexityAnalyzer

class ComplexityPredictor:
    def __init__(self, model_path="best_model.pkl", scaler_path="scaler.pkl", mapping_path="label_mapping.json"):
        self.model = None
        self.scaler = None
        self.label_mapping = {}
        self.is_ready = False
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(mapping_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                with open(mapping_path, "r") as f:
                    self.label_mapping = json.load(f)
                self.is_ready = True
                print("[Predictor] Successfully loaded Machine Learning model.")
            except Exception as e:
                print(f"[Predictor] Warning: Failed to load ML artifacts. Error: {e}")
        else:
            print("[Predictor] Warning: ML artifacts not found. Falling back to Static Heuristic Analysis.")

    def predict(self, code):
        tree, err = parse_code(code)
        if err:
            return {
                "status": "error", 
                "message": f"Syntax Error: {err}",
                "complexity": "Unknown"
            }
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        features = analyzer.get_features()
        if not self.is_ready:
            static_engine = TimeComplexityAnalyzer(features)
            static_result = static_engine.estimate()
            return {
                "status": "success",
                "complexity": static_result["predicted_complexity"],
                "method": "static_fallback",
                "details": "ML model not found. Used mathematical heuristic analyzer."
            }
        try:
            vector = extract_feature_vector(features)
            df = pd.DataFrame([vector], columns=get_feature_names())
            df_scaled = self.scaler.transform(df)
            prediction_idx = self.model.predict(df_scaled)[0]
            predicted_label = self.label_mapping.get(str(prediction_idx), "Unknown")
            return {
                "status": "success",
                "complexity": predicted_label,
                "method": "machine_learning",
                "details": "Prediction generated using the trained Random Forest / XGBoost model."
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Prediction failed: {str(e)}",
                "complexity": "Unknown"
            }

if __name__ == "__main__":
    predictor = ComplexityPredictor()
    test_code = """
def find_max(arr):
    max_val = arr[0]
    for i in arr:
        if i > max_val:
            max_val = i
    return max_val
"""
    result = predictor.predict(test_code)
    print("\nTest Result:")
    print(json.dumps(result, indent=4))
