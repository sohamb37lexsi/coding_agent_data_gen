# AlignTune Synthesizer 🚀

An execution-grounded data generation pipeline that synthesizes, verifies, and curates **SFT**, **DPO**, and **GRPO** training datasets for coding agents — using runtime truth instead of LLM-based verification.

Built specifically for the [AlignTune](https://aligntune.lexsi.ai/) ML fine-tuning library.

---

## Core Idea

Traditional data synthesis pipelines rely on the LLM to write unit tests to verify its own code. This is circular — the model hallucinates the tests as much as it hallucinates the solutions.

**Our approach:** The LLM generates code. The Python interpreter judges it. If it runs cleanly (`returncode == 0`), it's a good sample. If it crashes, the literal `stderr` traceback is fed back to the LLM for self-correction.

```
LLM (Generator) → Code → Sandbox (Judge) → Pass/Fail
                                 ↑              │
                                 └── Traceback ──┘
```

---

## Architecture

The pipeline has 4 phases:

### Phase 1 — Problem Generation (`phase1_generator.py`)
The LLM reads AlignTune's API docs and generates diverse coding problems — ranging from basic SFT trainer setup to multi-algorithm workflows. Problems are constrained to use tiny models (`sshleifer/tiny-gpt2`) and `max_steps=2` so sandbox execution stays fast.

### Phase 2 — Planner-Solver Loop (`phase2_agent.py`)
For each problem, a multi-turn ReAct loop runs:

1. **Planner** — LLM reads the problem + API docs and writes a step-by-step natural language plan.
2. **Solver** — LLM translates the plan into a complete Python script with execution calls (`print()`, `assert`).
3. **Sandbox** — Script executes in an isolated venv with `aligntune` pre-installed.

### Phase 3 — Traceback Feedback (inside Phase 2)
If the script crashes, the `stderr` traceback is captured and injected back into the LLM prompt: *"Execution failed with: [...] Revise your plan."* The model gets N turns (default 3) to produce a passing script.

### Phase 4 — Dataset Construction (`phase4_compiler.py`)
Successful trajectories are compiled into training-ready formats:

| Output | Format | Description |
|--------|--------|-------------|
| `sft_data.jsonl` | `{messages: [{role, content}]}` | Prompt → Successful Plan + Code |
| `dpo_data.jsonl` | `{prompt, chosen, rejected}` | Successful turn vs. prior failed turn |
| `trajectory_sft.json` | Full trajectory objects | Multi-turn history for GRPO training |

---

## Project Structure

```
aligntune-synthesizer/
├── config.py               # LLM client setup (OpenAI-compatible)
├── api_docs.py             # AlignTune API reference string
├── data_types.py           # Dataclasses: TurnAttempt, TaskTrajectory
├── sandbox.py              # Isolated venv + code sanitizer + executor
├── phase1_generator.py     # Problem generation with fallbacks
├── phase2_agent.py         # Planner-Solver ReAct loop
├── phase4_compiler.py      # SFT/DPO/GRPO dataset formatting
├── main.py                 # Orchestrator
└── output/                 # Generated datasets
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A GPU with CUDA (for sandbox execution of AlignTune scripts)
- An LLM backend (local vLLM, Ollama, or cloud API)

### 2. Install pipeline dependencies

```bash
pip install openai
```

### 3. Set up the sandbox environment

The sandbox creates an isolated venv with `aligntune` and its dependencies. Run this once:

```bash
python sandbox.py --setup
```

Verify it works:

```bash
python sandbox.py --test
```

You should see:

```
Test passed: True
stdout: aligntune import OK
```

### 4. Pre-download tiny models (optional but recommended)

Avoids burning sandbox timeout on first-run downloads:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ['sshleifer/tiny-gpt2', 'hf-internal-testing/tiny-random-LlamaForCausalLM']:
    AutoModelForCausalLM.from_pretrained(m)
    AutoTokenizer.from_pretrained(m)
print('Models cached.')
"
```

### 5. Configure the LLM backend

**Local vLLM (recommended)**

**One-time setup:**

```bash
conda create -n qwen_vllm python=3.12 -y
conda activate qwen_vllm
conda install -c nvidia cuda-toolkit -y
pip install vllm openai psutil "transformers<5.0.0" accelerate safetensors
```

**Terminal 1 — Start the server:**

```bash
conda activate qwen_vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.5
```

Wait for: `INFO: Uvicorn running on http://0.0.0.0:8000` — then leave this terminal open.

> `--gpu-memory-utilization 0.5` caps vLLM at 50% VRAM, leaving room for sandbox training. Default is 90%.

**Terminal 2 — Run the pipeline:**

```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
python main.py
```

Stop the server with `Ctrl+C` in Terminal 1 when done.

---

## Running the Pipeline

### Basic run

```bash
python main.py
```

This will:
1. Generate 2 coding problems (configurable in `main.py`)
2. Run the Planner-Solver loop (3 turns max per problem)
3. Compile successful trajectories into `output/`

### GPU configuration

```bash
# Use default GPU (inherits host CUDA)
python main.py

# Force CPU-only execution
export SANDBOX_CUDA_DEVICES=""
python main.py

# Use a specific GPU
export SANDBOX_CUDA_DEVICES="0"
python main.py
```

### Customizing the run

Edit `main.py`:

```python
# Number of problems to generate
generated_problems = generate_problems(ALIGNTUNE_API_DOCS, num_problems=10)

# Max self-correction turns per problem
traj = run_react_loop(problem, ALIGNTUNE_API_DOCS, max_turns=5)
```

---

## Output

After a successful run, check `output/`:

```
output/
├── sft_data.jsonl          # Instruction-tuning pairs (HF trl compatible)
├── dpo_data.jsonl          # Preference pairs (prompt/chosen/rejected)
└── trajectory_sft.json     # Full multi-turn agent trajectories
```

**`sft_data.jsonl`** — each line:
```json
{
  "messages": [
    {"role": "user", "content": "Problem: ...\n\nAPI Docs:\n..."},
    {"role": "assistant", "content": "<plan>\n...\n</plan>\n\n```python\n...\n```"}
  ]
}
```

**`dpo_data.jsonl`** — each line:
```json
{
  "prompt": "Problem: ...\n\nAPI Docs:\n...",
  "chosen": [{"role": "assistant", "content": "...successful code..."}],
  "rejected": [{"role": "assistant", "content": "...failed code..."}]
}
```

---

## How the Sandbox Works

The sandbox (`sandbox.py`) has three layers:

1. **Isolated venv** — a dedicated Python environment with `aligntune`, `torch`, `transformers`, `datasets`, and `accelerate` pre-installed. Created once via `python sandbox.py --setup`.

2. **Code sanitizer** — strips common LLM artifacts before execution:
   - Jupyter `!pip install` and `%magic` commands
   - Runtime `subprocess.run(pip install ...)` calls
   - Validates syntax via `ast.parse`

3. **Subprocess executor** — runs the cleaned script using the venv's Python binary with a 120s timeout, capturing stdout/stderr.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: aligntune` | Sandbox venv not set up | Run `python sandbox.py --setup` |
| `Execution timed out` | Model download or slow GPU | Pre-cache models (step 4 above), increase timeout in `sandbox.py` |
| `KeyError: slice` in Phase 2 | LLM returned dicts instead of strings | Already handled — `phase1_generator.py` normalizes output |
| All trajectories fail | LLM too small / docs too sparse | Use 14B+ model, verify `api_docs.py` is imported correctly |
| 0 SFT/DPO examples exported | No trajectory succeeded | Check sandbox logs, increase `max_turns`, or use a stronger LLM |

---

## Recommended Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Baseline | Fast |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 14B | Good | Moderate |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 32B | Best | Slow |
| `gpt-4o-mini` (cloud) | — | Good | Fast |

The 14B variant is the sweet spot — noticeably better than 7B at following the API doc constraints, while still fast enough for iterative runs.

---

## License

MIT
