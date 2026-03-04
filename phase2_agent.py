from typing import Optional
from config import client, MODEL_NAME
from data_types import TurnAttempt, TaskTrajectory
from sandbox import CodeSandbox

# ==========================================
# Phase 2 & 3: Planner-Solver-Traceback Loop
# ==========================================

def generate_plan(problem: str, api_docs: str, previous_error: Optional[str] = None) -> str:
    """The Planner: Generates a step-by-step natural language plan."""
    prompt = f"""Problem: {problem}\n\nAPI Docs:\n{api_docs}\n\n"""
    if previous_error:
        prompt += f"\nPREVIOUS EXECUTION FAILED WITH TRACEBACK:\n{previous_error}\n\nAnalyze the error and write a revised step-by-step plan to solve the problem."
    else:
        prompt += "\nWrite a detailed, step-by-step natural language plan to solve this problem using the library."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_code(problem: str, plan: str, api_docs: str) -> str:
    """The Solver: Translates the plan into Python code."""
    prompt = f"""Problem: {problem}\n\nAPI Docs:\n{api_docs}\n\nPlan:\n{plan}\n\n
Translate the plan above into a complete Python script. 
CRUCIAL: You MUST include basic execution calls (e.g., print() or simple assert statements with dummy inputs) at the bottom of the script so the logic actually runs and triggers the runtime.
Output ONLY the python code inside ```python ``` blocks."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return CodeSandbox.extract_code_blocks(response.choices[0].message.content)

def run_react_loop(problem: str, api_docs: str, max_turns: int = 3) -> TaskTrajectory:
    """Executes Phase 2 and Phase 3 (Automated Quality Filtering)."""
    print(f"\n--- Running Task: {problem[:50]}... ---")
    sandbox = CodeSandbox()
    trajectory = TaskTrajectory(problem=problem, api_context=api_docs, attempts=[], is_successful=False)
    
    current_traceback = None
    
    for turn in range(max_turns):
        print(f"  Turn {turn + 1}/{max_turns}...")
        
        plan = generate_plan(problem, api_docs, current_traceback)
        code = generate_code(problem, plan, api_docs)
        
        # Sandbox Execution acts as the Objective Judge
        success, stdout, stderr = sandbox.execute(code)
        
        attempt = TurnAttempt(
            turn_number=turn+1,
            plan=plan,
            code=code,
            traceback=stderr if not success else None,
            success=success
        )
        trajectory.attempts.append(attempt)
        
        if success:
            print(f"  ✅ Code passed execution filter!")
            trajectory.is_successful = True
            break
        else:
            print(f"  ❌ Execution failed. Extracting traceback for next turn.")
            current_traceback = stderr[-1500:] # Prevent context overflow
            
    return trajectory