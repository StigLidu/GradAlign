import re
import ast
import math
import argparse
import sys
from collections import Counter

def extract_problem_data(problem_str):
    """
    Extracts the target number and the list of available numbers from the problem statement.
    Expected format: "Given the numbers [15, 20, 36, 3], find ... reach the target 45."
    """
    try:
        # Extract numbers list
        nums_match = re.search(r"Given the numbers \[(.*?)\],", problem_str)
        if not nums_match:
             # Try without brackets just in case
            nums_match = re.search(r"Given the numbers (.*?), find", problem_str)
        
        if not nums_match:
            return None, None, "Could not find numbers in problem string"
            
        nums_str = nums_match.group(1)
        nums = [int(n.strip()) for n in nums_str.split(',')]
        
        # Extract target
        target_match = re.search(r"reach the target (\d+)", problem_str)
        if not target_match:
            return None, None, "Could not find target in problem string"
            
        target = int(target_match.group(1))
        
        return nums, target, None
    except Exception as e:
        return None, None, f"Error parsing problem data: {str(e)}"

def extract_boxed_content(response_str):
    """
    Extracts the content inside the last \boxed{...}.
    Handles nested braces if they exist (though unlikely in this simple format).
    """
    # Simple regex for non-nested boxed
    matches = re.findall(r"\\boxed\{(.*?)\}", response_str)
    if not matches:
        return None
    # Return the last one as the final answer
    return matches[-1]

def safe_evaluate(expression):
    """
    Safely evaluates a mathematical expression using AST.
    Returns the numerical result and a list of numbers used.
    """
    try:
        # Sanitize: allow only digits, operators, parens, dot, whitespace
        if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
            return None, None, "Expression contains invalid characters"

        # Parse with AST
        tree = ast.parse(expression, mode='eval')
        
        used_nums = []
        
        class NumVisitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    used_nums.append(node.value)
                self.generic_visit(node)
            # Python < 3.8 uses Num
            def visit_Num(self, node):
                used_nums.append(node.n)
                self.generic_visit(node)
                
        NumVisitor().visit(tree)
        
        # Evaluate
        # Since we sanitized the string, eval is relatively safe, 
        # but we can also compile the AST.
        code = compile(tree, filename="<string>", mode="eval")
        result = eval(code)
        
        return result, used_nums, None
    except Exception as e:
        return None, None, f"Error evaluating expression: {str(e)}"

def judge_countdown(problem_str, response_str):
    """
    Judges the student's response against the problem statement.
    Returns (is_correct, reason)
    """
    # 1. Parse problem
    nums, target, error = extract_problem_data(problem_str)
    if error:
        return False, error
    
    # 2. Extract boxed answer
    boxed_content = extract_boxed_content(response_str)
    if not boxed_content:
        return False, "No \\boxed{} content found"
    
    # 3. Parse equation inside boxed
    # Format expected: Expression = Target (e.g. 34-(43+39)/41=32)
    # We allow the student to just write the expression if it equals the target,
    # OR write Expr = Target.
    
    if "=" in boxed_content:
        parts = boxed_content.split('=')
        if len(parts) != 2:
            return False, "Boxed content has multiple '=' signs"
        lhs, rhs = parts[0].strip(), parts[1].strip()
        
        # Determine which side is the expression and which is the target value
        # Usually LHS is expression, RHS is value.
        # But we check both possibilities or enforce one.
        # If RHS is just a number equal to target, then LHS is expression.
        
        try:
            rhs_val = float(rhs)
            if abs(rhs_val - target) > 1e-6:
                 return False, f"Equation result {rhs} does not match target {target}"
            expression = lhs
        except ValueError:
            # Maybe LHS is the number? (Unlikely but possible: 32 = ...)
            try:
                lhs_val = float(lhs)
                if abs(lhs_val - target) > 1e-6:
                    return False, f"Equation result {lhs} does not match target {target}"
                expression = rhs
            except ValueError:
                return False, "Neither side of the equation is a valid number matching the target"
    else:
        # No equals sign, assume the whole content is the expression
        expression = boxed_content

    # 4. Validate expression
    result, used_nums, error = safe_evaluate(expression)
    if error:
        return False, error
        
    # 5. Check arithmetic correctness
    if abs(result - target) > 1e-6:
        return False, f"Expression evaluates to {result}, expected {target}"
        
    # 6. Check used numbers
    # We need to match used_nums with nums exactly (multiset equality)
    # Note: student might use 15.0 instead of 15.
    
    # Normalize used_nums to integers if they are effectively integers
    cleaned_used_nums = []
    for x in used_nums:
        if abs(x - round(x)) < 1e-6:
            cleaned_used_nums.append(int(round(x)))
        else:
            # If they used a non-integer literal that wasn't in input (e.g. 1.5), fail
            # unless the input had 1.5 (which is not the case for standard countdown)
            cleaned_used_nums.append(x)
            
    # Normalize input nums (they are ints from parsing)
    # Check counts
    nums_counter = Counter(nums)
    used_counter = Counter(cleaned_used_nums)
    
    if nums_counter != used_counter:
        missing = nums_counter - used_counter
        extra = used_counter - nums_counter
        reason = "Numbers usage mismatch."
        if missing:
            reason += f" Missing: {list(missing.elements())}."
        if extra:
            reason += f" Extra/Unexpected: {list(extra.elements())}."
        return False, reason

    return True, "Correct"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge a countdown game answer.")
    parser.add_argument("--problem", type=str, help="The problem string")
    parser.add_argument("--answer", type=str, help="The student's answer string")
    parser.add_argument("--test", action="store_true", help="Run internal tests")
    
    args = parser.parse_args()
    
    if args.test:
        # Test cases
        test_problem = "Given the numbers [15, 20, 36, 3], find a sequence of operations with only +, -, *, / to reach the target 45."
        
        # Correct
        assert judge_countdown(test_problem, r"Solution: \boxed{36 + 20 - (15 - 3) / 1.0 - 1 + 1 = 45}")[0] == False # Uses extra numbers
        assert judge_countdown(test_problem, r"Solution: \boxed{36 + 20 - 15 + 3 + 1 = 45}")[0] == False # 1 is not in list
        
        p2 = "Given the numbers [1, 2, 3], find a sequence of operations with only +, -, *, / to reach the target 6."
        assert judge_countdown(p2, r"\boxed{1+2+3=6}") == (True, "Correct")
        print(judge_countdown(p2, r"\boxed{1+2-3+3+3=6}"))
        assert judge_countdown(p2, r"\boxed{1*2*3=6}") == (True, "Correct")
        assert judge_countdown(p2, r"\boxed{1+2=3}") == (False, "Equation result 3 does not match target 6")
        assert judge_countdown(p2, r"\boxed{1+2+3}") == (True, "Correct") # relaxed rule
        assert judge_countdown(p2, r"\boxed{3+3=6}")[0] == False # Missing 1, 2. Extra 3.
        
        print("All tests passed!")
    elif args.problem and args.answer:
        is_correct, reason = judge_countdown(args.problem, args.answer)
        print(f"Correct: {is_correct}")
        print(f"Reason: {reason}")
        sys.exit(0 if is_correct else 1)
    else:
        parser.print_help()
