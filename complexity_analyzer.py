class TimeComplexityAnalyzer:
    def __init__(self, features):
        self.struct = features.get("structural_features", {})
        self.rec = features.get("recursion_features", {})
        self.logf = features.get("log_features", {})
        self.data_structs = features.get("data_structures", {})

    def compute_scores(self):
        input_loops = self.struct.get("input_dependent_loops", 0)
        max_depth = self.struct.get("max_loop_depth", 0)
        nested_pairs = self.struct.get("nested_loop_pairs", 0)
        quadratic_range = self.struct.get("range_quadratic", 0)
        dependent_nested = self.struct.get("dependent_nested_loops", 0)
        has_rec = self.rec.get("has_recursion", 0)
        binary_rec = self.rec.get("is_binary_recursion", 0)
        divide_conquer = self.rec.get("has_divide_and_conquer_pattern", 0)
        has_log = self.logf.get("has_division_by_two", 0) or self.logf.get("has_multiplicative_update", 0)
        uses_sort = self.logf.get("uses_sort", 0)
        uses_heapq = self.logf.get("uses_heapq", 0)
        uses_bisect = self.logf.get("uses_bisect", 0)
        sublinear_score = 0.0
        linear_score = 0.0
        nlogn_score = 0.0
        polynomial_score = 0.0
        if input_loops == 0 and not has_rec:
            sublinear_score += 0.8
        if uses_bisect:
            sublinear_score += 0.7
        if has_log and input_loops == 0:
            sublinear_score += 0.5
        if input_loops == 1:
            linear_score += 0.6
        if max_depth == 1:
            linear_score += 0.3
        if has_rec and not binary_rec and not divide_conquer:
            linear_score += 0.4
        if uses_sort:
            nlogn_score += 0.9
        if uses_heapq and input_loops >= 1:
            nlogn_score += 0.7
        if input_loops == 1 and has_log:
            nlogn_score += 0.5
        if divide_conquer:
            nlogn_score += 0.6
        if max_depth >= 2:
            polynomial_score += 0.5 + (0.1 * max_depth)
        if dependent_nested > 0 or quadratic_range > 0:
            polynomial_score += 0.6
        if binary_rec:
            polynomial_score += 0.8 
        if polynomial_score >= 0.5:
            nlogn_score *= 0.2
            linear_score *= 0.1
            sublinear_score *= 0.0
        elif nlogn_score >= 0.5:
            linear_score *= 0.2
            sublinear_score *= 0.0
        elif linear_score >= 0.5:
            sublinear_score *= 0.1
        return {
            "static_sublinear_score": min(sublinear_score, 1.0),
            "static_linear_score": min(linear_score, 1.0),
            "static_nlogn_score": min(nlogn_score, 1.0),
            "static_polynomial_score": min(polynomial_score, 1.0),
        }
    
    def estimate(self):
        scores = self.compute_scores()
        complexity_order = {
            "static_polynomial_score": (4, "O(n^k) or O(2^n)"),
            "static_nlogn_score": (3, "O(n log n)"),
            "static_linear_score": (2, "O(n)"),
            "static_sublinear_score": (1, "O(1) or O(log n)")
        }
        predicted_key = max(scores.keys(), key=lambda k: (scores[k], complexity_order[k][0]))
        return {
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "predicted_complexity": complexity_order[predicted_key][1]
        }
