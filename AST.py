import ast
import builtins
import json

BUILTINS = set(dir(builtins)) | {"__name__", "__main__"}

def parse_code(code: str):
    try:
        tree = ast.parse(code)
        return tree, None
    except SyntaxError as e:
        return None, str(e)

class CodeAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.num_loops = 0
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.num_conditions = 0
        self.num_function_calls = 0
        self.cyclomatic_complexity = 1
        self.input_dependent_loops = 0
        self.constant_loops = 0
        self.nested_loop_pairs = 0
        self.loop_stack = []
        self.loop_body_sizes = []
        self.has_break = False
        self.has_continue = False
        self.functions = {}
        self.current_function = "<global>"
        self.recursive_call_count = 0
        self.has_recursion = False
        self.is_binary_recursion = False
        self.has_divide_and_conquer_pattern = False
        self.has_division_by_two = False
        self.has_multiplicative_update = False
        self.uses_sort = False
        self.uses_heapq = False
        self.uses_bisect = False
        self.uses_list = False
        self.uses_dict = False
        self.uses_set = False
        self.node_counts = {
            "Assign": 0, "BinOp": 0, "Compare": 0, 
            "Return": 0, "ListComp": 0, "DictComp": 0
        }
        self.scope = {"<global>": {"defined": set(), "used": set()}}
        self.dependent_nested_loops = 0
        self.range_linear = 0
        self.range_quadratic = 0
        self.range_outer_var = 0
        self.range_constant_pattern = 0
        self.estimated_polynomial_degree = 0
        
    def is_dependent_on_outer(self, node):
        if not self.loop_stack or not isinstance(node.iter, ast.Call):
            return False
        outer_loop = self.loop_stack[-1]
        func_id = getattr(node.iter.func, "id", "")        
        if func_id == "range" and node.iter.args:
            arg = node.iter.args[0]
            if isinstance(arg, ast.Name) and isinstance(outer_loop, ast.For):
                outer_target = getattr(outer_loop.target, "id", "")
                if arg.id == outer_target:
                    return True
        return False

    def is_input_dependent(self, node):
        if isinstance(node.iter, ast.Name):
            return True
        if isinstance(node.iter, ast.Call):
            func_id = getattr(node.iter.func, "id", "")
            if func_id == "range" and node.iter.args:
                arg = node.iter.args[0]
                if isinstance(arg, ast.Name):
                    return True
                if isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "len":
                    return True
        return False

    def generic_visit(self, node):
        name = type(node).__name__
        if name in self.node_counts:
            self.node_counts[name] += 1
        super().generic_visit(node)

    def visit_If(self, node):
        self.num_conditions += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.num_conditions += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_List(self, node):
        self.uses_list = True
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        self.uses_list = True
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.uses_dict = True
        self.generic_visit(node)

    def visit_Set(self, node):
        self.uses_set = True
        self.generic_visit(node)

    def visit_For(self, node):
        self.num_loops += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.cyclomatic_complexity += 1
        if self.loop_depth > 1:
            self.nested_loop_pairs += 1
            self.estimated_polynomial_degree += 1
            if self.is_dependent_on_outer(node):
                self.dependent_nested_loops += 1
        if isinstance(node.iter, ast.Call) and getattr(node.iter.func, "id", "") == "range":
            if node.iter.args:
                arg = node.iter.args[0]
                if isinstance(arg, ast.Constant):
                    self.range_constant_pattern += 1
                elif isinstance(arg, ast.Name):
                    self.range_linear += 1
                elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
                    self.range_quadratic += 1
        self.loop_stack.append(node)
        self.loop_body_sizes.append(len(node.body))
        if self.is_input_dependent(node):
            self.input_dependent_loops += 1
        else:
            self.constant_loops += 1

        self.generic_visit(node)
        self.loop_stack.pop()
        self.loop_depth -= 1

    def visit_While(self, node):
        self.num_loops += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.cyclomatic_complexity += 1
        if self.loop_depth > 1:
            self.nested_loop_pairs += 1
        self.loop_stack.append(node)
        self.loop_body_sizes.append(len(node.body))
        self.input_dependent_loops += 1
        self.generic_visit(node)
        self.loop_stack.pop()
        self.loop_depth -= 1

    def visit_Break(self, node):
        self.has_break = True
        self.generic_visit(node)

    def visit_Continue(self, node):
        self.has_continue = True
        self.generic_visit(node)

    def visit_Call(self, node):
        self.num_function_calls += 1
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == self.current_function:
                self.has_recursion = True
                self.recursive_call_count += 1
            if func_name == "sorted":
                self.uses_sort = True
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "sort":
                self.uses_sort = True
            elif node.func.attr in ["heappush", "heappop", "heapify"]:
                self.uses_heapq = True
            elif node.func.attr in ["bisect", "insort", "bisect_left", "bisect_right"]:
                self.uses_bisect = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions[node.name] = {
            "recursive_calls": 0,
            "parameters": [arg.arg for arg in node.args.args]
        }
        self.scope[node.name] = {"defined": set(), "used": set()}
        prev = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        if self.recursive_call_count >= 2:
            self.is_binary_recursion = True
        self.current_function = prev

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Mult) and isinstance(node.value, ast.Constant) and node.value.value == 2:
            self.has_multiplicative_update = True
        if isinstance(node.op, ast.FloorDiv) and isinstance(node.value, ast.Constant) and node.value.value == 2:
            self.has_division_by_two = True
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value == 2:
            self.has_division_by_two = True
        if isinstance(node.op, ast.Mult) and isinstance(node.right, ast.Constant) and node.right.value == 2:
            self.has_multiplicative_update = True
        
        self.node_counts["BinOp"] += 1
        self.generic_visit(node)

    def get_features(self):
        avg_body = sum(self.loop_body_sizes) / len(self.loop_body_sizes) if self.loop_body_sizes else 0
        return {
            "structural_features": {
                "num_loops": self.num_loops,
                "max_loop_depth": self.max_loop_depth,
                "num_conditions": self.num_conditions,
                "num_function_calls": self.num_function_calls,
                "cyclomatic_complexity": self.cyclomatic_complexity,
                "input_dependent_loops": self.input_dependent_loops,
                "constant_loops": self.constant_loops,
                "nested_loop_pairs": self.nested_loop_pairs,
                "avg_loop_body_size": avg_body,
                "has_break": int(self.has_break),
                "has_continue": int(self.has_continue),
                "dependent_nested_loops": self.dependent_nested_loops,
                "range_linear": self.range_linear,
                "range_quadratic": self.range_quadratic,
                "range_constant_pattern": self.range_constant_pattern,
                "estimated_polynomial_degree": self.estimated_polynomial_degree
            },
            "recursion_features": {
                "has_recursion": int(self.has_recursion),
                "recursive_call_count": self.recursive_call_count,
                "is_binary_recursion": int(self.is_binary_recursion),
                "has_divide_and_conquer_pattern": int(self.has_division_by_two and self.is_binary_recursion)
            },
            "log_features": {
                "has_division_by_two": int(self.has_division_by_two),
                "has_multiplicative_update": int(self.has_multiplicative_update),
                "uses_sort": int(self.uses_sort),
                "uses_heapq": int(self.uses_heapq),
                "uses_bisect": int(self.uses_bisect)
            },
            "data_structures": {
                "uses_list": int(self.uses_list),
                "uses_dict": int(self.uses_dict),
                "uses_set": int(self.uses_set)
            },
            "node_histogram": self.node_counts
        }

def flatten_features(nested_dict, parent_key=""):
    items = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_features(v, new_key))
        else:
            items[new_key] = v
    return items

def create_feature_label_mapping(flat_features, output_path="feature_label_mapping.json"):
    sorted_features = sorted(flat_features.keys())
    mapping = {idx: name for idx, name in enumerate(sorted_features)}
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Feature label mapping saved to {output_path}")
    return mapping

if __name__ == "__main__":
    sample_code = """
def example(arr):
    n = len(arr)
    if n == 0:
        return
    for i in range(n):
        for j in range(n):
            print(arr[i] + arr[j])
"""
    tree, err = parse_code(sample_code)
    if err:
        print("Syntax Error:", err)
    else:
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        features = flatten_features(analyzer.get_features())
        print(json.dumps(features, indent=4))
