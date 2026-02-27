import os
import json
from typing import List
from data_types import TaskTrajectory

# ==========================================
# Phase 4: Constructing AUTOIF Datasets
# ==========================================

def compile_datasets(trajectories: List[TaskTrajectory], output_dir: str = "output"):
    """Parses successful trajectories into SFT and DPO training formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    sft_data = []
    dpo_data = []
    
    for traj in trajectories:
        if not traj.is_successful:
            continue # We only compile data from paths that eventually resolved
            
        final_success = traj.attempts[-1]
        
        # 1. SFT Format: (Prompt -> Plan + Code)
        sft_prompt = f"Problem: {traj.problem}\n\nAPI Docs:\n{traj.api_context}"
        sft_response = f"<plan>\n{final_success.plan}\n</plan>\n\n```python\n{final_success.code}\n```"
        
        sft_data.append({
            "messages": [
                {"role": "user", "content": sft_prompt},
                {"role": "assistant", "content": sft_response}
            ]
        })
        
        # 2. Offline DPO Data: Contrasts a successful turn with the prior failed turn
        if len(traj.attempts) > 1:
            failed_attempt = traj.attempts[-2] 
            rejected_response = f"<plan>\n{failed_attempt.plan}\n</plan>\n\n```python\n{failed_attempt.code}\n```"
            
            dpo_data.append({
                "prompt": sft_prompt,
                "chosen": [{"role": "assistant", "content": sft_response}],
                "rejected": [{"role": "assistant", "content": rejected_response}]
            })

    # Export to JSONL format
    with open(os.path.join(output_dir, "sft_data.jsonl"), "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(output_dir, "dpo_data.jsonl"), "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"\n--- Phase 4: Dataset Construction Complete ---")
    print(f"Exported {len(sft_data)} SFT examples to {output_dir}/sft_data.jsonl")
    print(f"Exported {len(dpo_data)} DPO examples to {output_dir}/dpo_data.jsonl")