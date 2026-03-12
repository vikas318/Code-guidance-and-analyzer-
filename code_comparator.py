import ollama
import warnings
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")
from complexity_predictor import ComplexityPredictor
STABLE_MODEL = 'qwen2.5-coder:0.5b'

class CodeComparator:
    def __init__(self):
        print("[Comparator] Booting up engines...")
        self.predictor = ComplexityPredictor()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.is_ready = True

    def calculate_similarity(self, code_a, code_b):
        try:
            vector_a = self.encoder.encode(code_a)
            vector_b = self.encoder.encode(code_b)            
            similarity = util.cos_sim(vector_a, vector_b).item()
            percentage = max(0.0, min(100.0, similarity * 100))
            return round(percentage, 1)
        except Exception as e:
            print(f"[Comparator] Similarity Error: {e}")
            return 0.0

    def get_logical_differences(self, code_a, code_b, comp_a, comp_b):
        rank_map = {
            "Sublinear (O(1) / O(log n))": 1,
            "Linear (O(n))": 2,
            "Near-linear (O(n log n))": 3,
            "Polynomial+ (O(n^2) and above)": 4,
            "Unknown": 99
        }
        rank_a = rank_map.get(comp_a, 99)
        rank_b = rank_map.get(comp_b, 99)
        if rank_a < rank_b:
            efficiency_statement = f"Code A is mathematically more efficient for large inputs ({comp_a} vs {comp_b})."
        elif rank_b < rank_a:
            efficiency_statement = f"Code B is mathematically more efficient for large inputs ({comp_b} vs {comp_a})."
        else:
            efficiency_statement = f"Both codes have similar asymptotic efficiency ({comp_a})."
        prompt = f"""
        Code A:
        ```python
        {code_a}
        ```
        1. Summarize what Code A does.
        Code B:
        ```python
        {code_b}
        ```
        2. Summarize what Code B does.
        """ 
        try:
            response = ollama.chat(
                model=STABLE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_gpu": 0, "temperature": 0.1}
            )
            ai_desc = response['message']['content'].strip()
        except Exception as e:
            ai_desc = f"[AI Engine Offline] Error: {e}"
        final_output = f"1. **Mechanics**: {ai_desc}\n"
        final_output += f"2. **Complexity**: Code A is {comp_a} | Code B is {comp_b}\n"
        final_output += f"3. **Efficiency**: {efficiency_statement}"
        return final_output

    def compare(self, code_a, code_b):
        print("\n[Comparator] Calculating mathematical similarity...")
        match_percentage = self.calculate_similarity(code_a, code_b)
        print("[Comparator] Predicting time complexities using ML model...")
        res_a = self.predictor.predict(code_a)
        res_b = self.predictor.predict(code_b)
        comp_a = res_a.get("complexity", "Unknown")
        comp_b = res_b.get("complexity", "Unknown")
        print("[Comparator] Generating logic explanation...")
        ai_feedback = self.get_logical_differences(code_a, code_b, comp_a, comp_b)
        return {
            "status": "success",
            "similarity_score": f"{match_percentage}%",
            "complexity_code_a": comp_a,
            "complexity_code_b": comp_b,
            "ai_feedback": ai_feedback
        }

if __name__ == "__main__":
    comparator = CodeComparator()
    
    code_linear = """
def find_item(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
"""

    code_binary = """
def search(nums, val):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == val:
            return mid
        elif nums[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    print("\n--- Testing Comparison (Linear vs Binary Search) ---")
    result = comparator.compare(code_linear, code_binary)
    print(f"\nLOGICAL MATCH: {result['similarity_score']}")
    print(f"AI ANALYSIS:\n{result['ai_feedback']}\n")
