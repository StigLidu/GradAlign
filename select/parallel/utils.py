import re
from typing import Optional
# from sympy.parsing.latex import parse_latex
# import sympy as sp

def create_prompt(problem: str, tokenizer) -> str:
    """
    Create a prompt for the model to solve the math problem using the model's chat template
    """
    
    # Create messages in the format expected by the chat template
    messages = [
        {
            "role": "user", 
            "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}"
        }
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt

def extract_answer(response: str) -> Optional[str]:
    """
    Extract the answer from the model's response, handling generic expressions
    """
    # Look for the last \boxed{} pattern and match it properly
    # This regex finds \boxed{ and matches content including nested braces
    import re
    
    # Find all \boxed{...} patterns, handling nested braces
    pattern = r'\\boxed\{'
    matches = []
    
    start_pos = 0
    while True:
        match = re.search(pattern, response[start_pos:])
        if not match:
            break
        
        # Found \boxed{, now find the matching closing brace
        start_idx = start_pos + match.start() + len('\\boxed{')
        brace_count = 1
        end_idx = start_idx
        
        while end_idx < len(response) and brace_count > 0:
            char = response[end_idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            # Found matching closing brace
            content = response[start_idx:end_idx-1].strip()
            matches.append(content)
        
        start_pos = start_pos + match.end()
    
    if matches:
        # Take the last match and clean it up
        answer = matches[-1].strip()
        return answer if answer else None
    
    return None

def compare_answers(model_answer: str, true_answer: str) -> bool:
    """
    Compare answers using symbolic mathematics with SymPy
    """
    if model_answer is None or true_answer is None:
        return False
    
    try:
        # 1. Strip delimiters
        model_tex = str(model_answer).strip('$ ')
        true_tex = str(true_answer).strip('$ ')
        
        # Handle empty answers
        if not model_tex or not true_tex:
            return model_tex == true_tex
        
        # 2. Parse with SymPy
        # try:
        #     model_expr = parse_latex(model_tex)
        #     true_expr = parse_latex(true_tex)
            
        #     # 3. Symbolic check
        #     if sp.simplify(model_expr - true_expr) == 0:
        #         return True
        #     else:
        #         return False
                
        # except Exception as parse_error:
        #     print(f"    ⚠️  LaTeX parsing failed: {parse_error}")
            # Fallback to string comparison for non-LaTeX answers
        return normalize_answer_fallback(model_answer) == normalize_answer_fallback(true_answer)
            
    except Exception as e:
        print(f"    ⚠️  Answer comparison failed: {e}")
        # Ultimate fallback to string comparison
        return normalize_answer_fallback(model_answer) == normalize_answer_fallback(true_answer)

def normalize_answer_fallback(answer: str) -> str:
    """
    Fallback normalization for string comparison when symbolic comparison fails
    """
    if answer is None:
        return ""
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', str(answer).strip())
    
    # Remove trailing periods
    normalized = normalized.rstrip('.')
    
    # Remove common delimiters
    normalized = normalized.strip('$ ')
    
    return normalized

if __name__ == "__main__":
    print(extract_answer("\\boxed{123}"))
    print(compare_answers("True", "True"))