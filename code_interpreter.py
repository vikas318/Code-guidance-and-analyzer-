import os
import tempfile
import subprocess
import ollama
from complexity_predictor import ComplexityPredictor

STABLE_MODEL = 'qwen2.5-coder:0.5b'

class CodeInterpreter:
    def __init__(self):
        self.predictor = ComplexityPredictor()

    def run_safe_compilation(self, code, timeout_seconds=3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            if result.returncode == 0:
                return True, result.stdout, None
            else:
                return False, result.stderr, "Runtime/Syntax Error"
                
        except subprocess.TimeoutExpired:
            return False, f"Code execution exceeded {timeout_seconds} seconds.", "Infinite Loop"
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def evaluate_dsa(self, user_code, test_cases):
        results = []
        all_passed = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(user_code)
            temp_file_path = temp_file.name

        try:
            for idx, tc in enumerate(test_cases):
                tc_input = tc.get("input", "")
                expected_output = tc.get("expected_output", "").strip() 
                try:
                    proc = subprocess.run(
                        ['python', temp_file_path],
                        input=tc_input,
                        capture_output=True,
                        text=True,
                        timeout=3
                    )
                    if proc.returncode != 0:
                        human_error = self.translate_error_to_english(user_code, proc.stderr, "Runtime Error")
                        results.append({
                            "case": idx + 1, 
                            "status": "Error", 
                            "details": human_error
                        })
                        all_passed = False
                        break
                    actual_output = proc.stdout.strip()
                    if actual_output == expected_output:
                        results.append({"case": idx + 1, "status": "Passed"})
                    else:
                        results.append({
                            "case": idx + 1, 
                            "status": "Failed", 
                            "expected": expected_output, 
                            "actual": actual_output,
                            "input": tc_input
                        })
                        all_passed = False      
                except subprocess.TimeoutExpired:
                    results.append({"case": idx + 1, "status": "Timeout", "details": "Infinite loop detected during execution."})
                    all_passed = False
                    break
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        complexity = "N/A"
        if all_passed:
            comp_res = self.predictor.predict(user_code)
            complexity = comp_res.get("complexity", "Unknown")   
        return {
            "success": all_passed,
            "test_results": results,
            "predicted_complexity": complexity
        }
    
    def ask_ai_for_explanation(self, prompt):
        try:
            response = ollama.chat(
                model=STABLE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_gpu": 0}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"[AI Engine Offline] Error: {e}"

    def translate_error_to_english(self, code, raw_error, error_type):
        prompt = f"""
        You are a friendly Python tutor. The student's code threw this error:
        Error Type: {error_type}
        Raw Error: {raw_error}
        
        Student Code:
        ```python
        {code}
        ```
        
        In exactly 2 short bullet points, explain what went wrong in plain English 
        and how they can fix it. Do not use heavy jargon. Do not show the raw error code.
        """
        return self.ask_ai_for_explanation(prompt)

    def explain_code_logic(self, code):
        prompt = f"""
        Analyze this Python code. In 2 short bullet points, explain its core algorithm 
        and what it accomplishes. Be extremely concise.
        
        Code:
        ```python
        {code}
        ```
        """
        return self.ask_ai_for_explanation(prompt)

    def analyze(self, code):
        is_success, output, error_type = self.run_safe_compilation(code)
        if not is_success:
            human_readable_error = self.translate_error_to_english(code, output, error_type)
            return {
                "status": "error",
                "error_type": error_type,
                "ai_feedback": human_readable_error,
                "complexity": "N/A (Code failed to run)"
            }
        complexity_result = self.predictor.predict(code)
        predicted_complexity = complexity_result.get("complexity", "Unknown")
        logic_summary = self.explain_code_logic(code)
        return {
            "status": "success",
            "execution_output": output,
            "complexity": predicted_complexity,
            "ai_feedback": logic_summary
        }
    
    def analyze_static(self, code):
        complexity_result = self.predictor.predict(code)
        predicted_complexity = complexity_result.get("complexity", "Unknown")
        logic_summary = self.explain_code_logic(code)
        return {
            "status": "success",
            "complexity": predicted_complexity,
            "ai_feedback": logic_summary
        }

if __name__ == "__main__":
    interpreter = CodeInterpreter()
    bad_code = """
def increment_x():
    x = 0
    while True:
        x += 1

increment_x()
print(x)
"""
    print("\n--- Testing Bad Code (Infinite Loop) ---")
    bad_result = interpreter.analyze(bad_code)
    for key, value in bad_result.items():
        print(f"{key.upper()}:\n{value}\n")
    good_code = """
def find_max(arr):
    if not arr: return None
    highest = arr[0]
    for num in arr:
        if num > highest:
            highest = num
    return highest

print(find_max([1, 5, 3, 9, 2]))
"""
    print("\n--- Testing Good Code (Find Max) ---")
    good_result = interpreter.analyze(good_code)
    for key, value in good_result.items():
        print(f"{key.upper()}:\n{value}\n")
