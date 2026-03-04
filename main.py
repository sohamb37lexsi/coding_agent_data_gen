import os
import json
from dataclasses import asdict

from phase1_generator import generate_problems
from phase2_agent import run_react_loop
from phase4_compiler import compile_datasets
from api_docs import ALIGNTUNE_API_DOCS

# ==========================================
# Main Execution Flow
# ==========================================

# Dummy mock for aligntune documentation (replace with your actual API map context)
# MOCK_API_DOCS = """
# aligntune.Trainer: A class to initialize an RL trainer.
# Args:
#    model_name (str): The HF model name.
#    vram_gb (int): Max VRAM constraint.
# Methods:
#    train(dataset): starts training.
# """

# MOCK_API_DOCS = ALIGNTUNE_API_DOCS

def main():
    print("Initializing AlignTune Data Synthesis Pipeline...\n")
    
    # Run Phase 1
    generated_problems = generate_problems(ALIGNTUNE_API_DOCS, num_problems=2)
    
    # Run Phase 2 & 3
    all_trajectories = []
    for problem in generated_problems:
        traj = run_react_loop(problem, ALIGNTUNE_API_DOCS, max_turns=3)
        all_trajectories.append(traj)
        
    # Run Phase 4
    compile_datasets(all_trajectories, output_dir="output")
    
    # Optional: Save full multi-turn trajectory for Iterative GRPO
    os.makedirs("output", exist_ok=True)
    trajectory_path = os.path.join("output", "trajectory_sft.json")
    with open(trajectory_path, "w") as f:
        json.dump([asdict(t) for t in all_trajectories], f, indent=2)
    print(f"Exported multi-turn trace records to {trajectory_path}")

if __name__ == "__main__":
    main()