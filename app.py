import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from code_interpreter import CodeInterpreter
from code_comparator import CodeComparator
from smart_search import SmartSearchEngine
app = Flask(__name__)
CORS(app)
print("\n" + "="*50)
print("  INITIALIZING CODE GUIDANCE & ANALYZER BACKEND  ")
print("="*50)
interpreter = CodeInterpreter()
comparator = CodeComparator()
search_engine = SmartSearchEngine()
DSA_DIR = "dsa_database"
PLAYGROUND_DIR = "user_playground"
current_playground = PLAYGROUND_DIR
os.makedirs(DSA_DIR, exist_ok=True)
os.makedirs(PLAYGROUND_DIR, exist_ok=True)
search_engine.index_directory(DSA_DIR, "dsa")
search_engine.index_directory(PLAYGROUND_DIR, "sandbox")
print("\n[Server] All engines operational. Starting Flask API...\n")

@app.route('/api/compile', methods=['POST'])
def compile_and_analyze():
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({"status": "error", "message": "No code provided."}), 400
    user_code = data['code']
    result = interpreter.analyze(user_code)
    return jsonify(result)

@app.route('/api/compare', methods=['POST'])
def compare_codes():
    data = request.get_json()
    if not data or 'code_a' not in data or 'code_b' not in data:
        return jsonify({"status": "error", "message": "Both code_a and code_b are required."}), 400
    code_a = data['code_a']
    code_b = data['code_b']
    result = comparator.compare(code_a, code_b)
    return jsonify(result)

@app.route('/api/search', methods=['POST'])
def smart_search():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "Search query required."}), 400
    query = data['query']
    results = search_engine.search(query, threshold=25.0, top_k=5)

    return jsonify({"status": "success", "query": query, "results": results})

@app.route('/api/set_playground', methods=['POST'])
def set_playground():
    global current_playground
    data = request.get_json()
    new_path = data.get('directory_path')
    
    if not new_path or not os.path.exists(new_path):
        return jsonify({"status": "error", "message": "Directory does not exist."}), 404

    current_playground = new_path
    search_engine.index_directory(new_path, "sandbox")
    return jsonify({"status": "success", "message": f"Playground updated to {new_path}"})

@app.route('/api/files', methods=['GET'])
def list_files():
    global current_playground
    if not os.path.exists(current_playground):
        return jsonify({"files": []})
    
    files = [f for f in os.listdir(current_playground) if f.endswith('.py')]
    return jsonify({"files": files})

@app.route('/api/files/read', methods=['POST'])
def read_file():
    global current_playground
    filename = request.get_json().get('filename')
    filepath = os.path.join(current_playground, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return jsonify({"status": "success", "content": f.read()})
    return jsonify({"status": "error", "message": "File not found."}), 404

@app.route('/api/files/save', methods=['POST'])
def save_file():
    global current_playground
    data = request.get_json()
    filename = data.get('filename')
    content = data.get('content')    
    if not filename.endswith('.py'):
        filename += '.py'        
    filepath = os.path.join(current_playground, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        search_engine.index_directory(current_playground, "sandbox")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dsa/problems', methods=['GET'])
def list_dsa_problems():
    problems = []
    if os.path.exists(DSA_DIR):
        for f in os.listdir(DSA_DIR):
            if f.endswith('.json'):
                filepath = os.path.join(DSA_DIR, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        problems.append({
                            "id": data.get("id", f), 
                            "title": data.get("title", f.replace('.json', '')),
                            "filename": f
                        })
                except Exception as e:
                    print(f"Error loading {f}: {e}")
    return jsonify({"problems": problems})

@app.route('/api/dsa/problem/<filename>', methods=['GET'])
def get_dsa_problem(filename):
    filepath = os.path.join(DSA_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return jsonify({"status": "success", "data": json.load(file)})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Problem not found."}), 404

@app.route('/api/dsa/evaluate', methods=['POST'])
def evaluate_dsa_solution():
    req_data = request.get_json()
    user_code = req_data.get('code')
    test_cases = req_data.get('test_cases', [])
    if not user_code or not test_cases:
        return jsonify({"status": "error", "message": "Code and test cases required."}), 400
    result = interpreter.evaluate_dsa(user_code, test_cases)
    return jsonify(result)

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({"status": "error", "message": "No code provided."}), 400
    user_code = data['code']
    result = interpreter.analyze_static(user_code)
    return jsonify(result)
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
