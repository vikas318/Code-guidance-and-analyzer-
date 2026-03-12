import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from AST import parse_code, CodeAnalyzer
from complexity_analyzer import TimeComplexityAnalyzer

DATASET_FILE = "python_data.jsonl"
FEATURE_OUTPUT = "features.csv"
LABEL_OUTPUT = "labels.csv"
LABEL_MAP_OUTPUT = "label_mapping.json"
FEATURE_MAP_OUTPUT = "feature_mapping.json"
SCALER_OUTPUT = "scaler.pkl"
LABEL_MAPPING = {
    "constant": 0,
    "logn": 0,
    "linear": 1,
    "nlogn": 2,
    "quadratic": 3,
    "cubic": 3,
    "np": 3
}
INVERSE_LABEL_MAPPING = {
    0: "Sublinear (O(1) / O(log n))",
    1: "Linear (O(n))",
    2: "Near-linear (O(n log n))",
    3: "Polynomial+ (O(n^2) and above)"
}

def map_complexity(label):
    return LABEL_MAPPING.get(label, None)

def extract_feature_vector(features):
    struct = features["structural_features"]
    rec = features["recursion_features"]
    logf = features["log_features"]
    node_hist = features["node_histogram"]
    data_structs = features["data_structures"]
    loop_depth_x_input = struct["max_loop_depth"] * struct["input_dependent_loops"]
    quadratic_signal = struct["range_quadratic"] + struct["dependent_nested_loops"]
    log_signal = int(logf["has_division_by_two"]) + int(logf["has_multiplicative_update"])
    recursion_signal = int(rec["is_binary_recursion"]) + int(rec["has_divide_and_conquer_pattern"])
    effective_input_degree = struct["max_loop_depth"] * struct["input_dependent_loops"]
    static_analyzer = TimeComplexityAnalyzer(features)
    static_scores = static_analyzer.compute_scores()
    vector = [
        struct["num_loops"],
        struct["max_loop_depth"],
        struct["input_dependent_loops"],
        struct["constant_loops"],
        struct["nested_loop_pairs"],
        struct["estimated_polynomial_degree"],
        struct["cyclomatic_complexity"],
        struct["avg_loop_body_size"],
        struct["num_function_calls"],
        struct["range_linear"],
        struct["range_quadratic"],
        struct["range_constant_pattern"],
        struct["dependent_nested_loops"],
        rec["has_recursion"],
        rec["is_binary_recursion"],
        rec["has_divide_and_conquer_pattern"],
        logf["has_division_by_two"],
        logf["has_multiplicative_update"],
        logf["uses_sort"],
        logf["uses_heapq"],
        logf["uses_bisect"],
        data_structs["uses_list"],
        data_structs["uses_dict"],
        data_structs["uses_set"],
        node_hist["BinOp"],
        node_hist["Compare"],
        node_hist["Return"],
        loop_depth_x_input,
        quadratic_signal,
        log_signal,
        recursion_signal,
        effective_input_degree,
        static_scores["static_sublinear_score"],
        static_scores["static_linear_score"],
        static_scores["static_nlogn_score"],
        static_scores["static_polynomial_score"],
    ]
    return vector

def get_feature_names():
    return [
        "num_loops", "max_loop_depth", "input_dependent_loops", "constant_loops",
        "nested_loop_pairs", "estimated_polynomial_degree", "cyclomatic_complexity",
        "avg_loop_body_size", "num_function_calls", "range_linear", "range_quadratic",
        "range_constant_pattern", "dependent_nested_loops", "has_recursion",
        "is_binary_recursion", "has_divide_and_conquer_pattern", "has_division_by_two",
        "has_multiplicative_update", "uses_sort", "uses_heapq", "uses_bisect",
        "uses_list", "uses_dict", "uses_set", "binop_count", "compare_count",
        "return_count", "loop_depth_x_input", "quadratic_signal", "log_signal",
        "recursion_signal", "effective_input_degree", "static_sublinear_score",
        "static_linear_score", "static_nlogn_score", "static_polynomial_score"
    ]

def preprocess_dataset():
    X = []
    y = []
    total = 0
    skipped = 0
    print(f"Reading {DATASET_FILE}... This might take a moment depending on file size.")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if total % 500 == 0:
                print(f"Processed {total} lines...")
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            code = data.get("src", "")
            original_label = data.get("complexity", "unknown")
            mapped_label = map_complexity(original_label)
            if mapped_label is None:
                skipped += 1
                continue
            tree, err = parse_code(code)
            if err:
                skipped += 1
                continue
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            features = analyzer.get_features()
            vector = extract_feature_vector(features)
            X.append(vector)
            y.append(mapped_label)
    if not X:
        print("No valid samples found. Please check your JSONL format.")
        return
    feature_names = get_feature_names()
    df_X = pd.DataFrame(X, columns=feature_names)
    scaler = StandardScaler()
    df_X_scaled = pd.DataFrame(
        scaler.fit_transform(df_X),
        columns=feature_names
    )
    joblib.dump(scaler, SCALER_OUTPUT)
    df_X_scaled.to_csv(FEATURE_OUTPUT, index=False)
    df_y = pd.DataFrame({"label": y})
    df_y.to_csv(LABEL_OUTPUT, index=False)
    with open(LABEL_MAP_OUTPUT, "w") as f:
        json.dump(INVERSE_LABEL_MAPPING, f, indent=4)
    with open(FEATURE_MAP_OUTPUT, "w") as f:
        json.dump({i: name for i, name in enumerate(feature_names)}, f, indent=4)
    print("\n" + "="*30)
    print("      DATASET SUMMARY      ")
    print("="*30)
    print(f"Total lines read: {total}")
    print(f"Skipped (Parse/Label error): {skipped}")
    print(f"Valid samples processed: {len(X)}")
    print(f"Number of ML Features: {len(feature_names)}")
    print("\nOutputs Generated:")
    print(f"- {FEATURE_OUTPUT}")
    print(f"- {LABEL_OUTPUT}")
    print(f"- {SCALER_OUTPUT}")
    print(f"- {LABEL_MAP_OUTPUT}")

if __name__ == "__main__":
    preprocess_dataset()
