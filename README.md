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
├── generation_constraints.py # Model pool, param validation, prompt hardening
├── quality_scorer.py       # Multi-dimensional scoring (syntax/api/exec/outcome)
├── quality_pool.py         # Passing solution store + few-shot retrieval
├── data_types.py           # Dataclasses: TurnAttempt, TaskTrajectory
├── sandbox.py              # Isolated venv + code sanitizer + executor
├── phase1_generator.py     # Multi-strategy synthesis (ToolACE, OSS-Instruct, Evol-Instruct)
├── phase2_agent.py         # Planner-Solver with Reflexion + quality pool + scoring
├── phase4_compiler.py      # Quality-ranked SFT/DPO dataset construction
├── rag_indexer.py          # Codebase parser + ChromaDB indexer
├── rag_retriever.py        # Semantic search + bug report DB
├── main.py                 # Orchestrator
├── .chroma_db/             # ChromaDB storage (auto-created)
├── repos/                  # Cloned library repos for indexing
│   ├── aligntune/
│   ├── trl/
│   └── unsloth/
└── output/                 # Generated datasets + quality pool + failure log
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

### Basic run (static docs, no RAG)

```bash
python main.py
```

### RAG-enhanced run

RAG mode retrieves relevant API docs and library source code dynamically, and logs unresolved errors to a bug report database for developer review.

**1. Install RAG dependencies:**

```bash
pip install chromadb sentence-transformers
```

**2. Clone the library repos and index them:**

```bash
# Clone repos (one-time)
git clone https://github.com/Lexsi-Labs/aligntune.git ./repos/aligntune
git clone https://github.com/huggingface/trl.git ./repos/trl
git clone https://github.com/unslothai/unsloth.git ./repos/unsloth

# Index all codebases + API docs into ChromaDB
python rag_indexer.py \
    --aligntune-path ./repos/aligntune \
    --trl-path ./repos/trl \
    --unsloth-path ./repos/unsloth \
    --api-docs \
    --stats
```

This creates a `.chroma_db/` folder in your project directory. No Docker, no PostgreSQL, no permissions needed.

**3. Run the pipeline with RAG:**

```bash
python main.py --use-rag --num-problems 5 --max-turns 5
```

**4. Review unresolved errors (developer feedback):**

```bash
# Print to terminal
python main.py --show-bugs

# Export to JSON for issue tracking / CI
python main.py --export-bugs output/bug_report.json
```

This prints all errors the pipeline couldn't self-correct, grouped by severity, with the LLM's diagnosis of whether the bug is in user code or the library itself.

---

## Generation Safety Features

### Multi-Strategy Problem Synthesis (Phase 1)
Three strategies ensure diverse problem coverage:

| Strategy | Source | What it produces |
|---|---|---|
| **api_doc** (40%) | API function signatures + docs | "Use `create_rl_trainer` with DPO on hh-rlhf" — user-perspective tasks |
| **source_code** (30%) | Actual library source via RAG | "Exercise the fallback branch in `setup_model()`" — developer/tester tasks |
| **mutation** (30%) | Existing passing solutions from quality pool | "Add eval step to this SFT script" — evolved harder variants |

### Quality Pool Feedback Loop
Passing solutions are stored in ChromaDB and injected as few-shot examples into future Planner and Solver prompts. The pool grows across pipeline runs — each run benefits from all prior successes.

```bash
# Check pool stats
python main.py --pool-stats
```

### Multi-Dimensional Scoring
Every attempt is scored on 4 dimensions (not just pass/fail):

| Dimension | Weight | Signal |
|---|---|---|
| Syntax | 0.10 | `ast.parse` succeeds |
| API compliance | 0.20 | Code imports/calls aligntune |
| Execution | 0.35 | Sandbox `returncode == 0` |
| Outcome | 0.35 | Output matches expected patterns (loss, "SUCCESS") |

DPO pairs are ranked by composite score, so a syntax-error response is ranked lower than an import-error response, which is lower than a runtime crash.

### Reflexion Repair
After each failed turn, the LLM writes a verbal reflection: "I tried X, it failed because Y, next time I should Z." Reflections accumulate across turns and are injected into every subsequent prompt, preventing the model from repeating the same mistake.

### Solver Sees Previous Code
On retry turns, the Solver receives its own failed code alongside the traceback, enabling targeted fixes instead of full rewrites.

### Model Rotation
When the sandbox hits an architecture-specific error (e.g., "Target modules not found", unsloth compiled cache bugs), the pipeline automatically rotates to the next approved model instead of retrying the same one.

Approved model pool (defined in `generation_constraints.py`):

| Model | Arch | Backends | Notes |
|-------|------|----------|-------|
| `sshleifer/tiny-gpt2` | GPT-2 | trl only | Uses c_attn/c_proj, not q_proj/v_proj |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | Llama | trl, unsloth | Standard attention modules |
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2 | trl, unsloth | Requires trust_remote_code=True |

To add models, edit `MODEL_POOL` in `generation_constraints.py`.

### Parameter Validation
Generated code is checked against hard constraints *before* sandbox execution:
- Only approved models and datasets
- `max_steps` capped at 5
- No quantization for tiny models
- No explicit LoRA target modules (always `"all-linear"`)
- No `pip install` or `subprocess` calls

### Prompt Hardening
Both the Planner and Solver receive system prompts with explicit forbidden actions and required patterns. The constraints block is injected into every generation call, not just the API docs.

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

Edit `main.py` args or call directly:

```bash
# More problems, more retries, with RAG
python main.py --num-problems 10 --max-turns 5 --use-rag

# Static mode, fewer problems
python main.py --num-problems 3 --max-turns 3

# Export bug reports after a run
python main.py --export-bugs output/bugs.json

# Check quality pool growth
python main.py --pool-stats
```

To modify the approved model pool, parameter constraints, or synthesis strategy quotas, edit `generation_constraints.py` and `phase1_generator.py`.

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
| `psycopg2.OperationalError` | PostgreSQL not running | Replaced with ChromaDB — no DB server needed |
| RAG retrieval returns junk | Repos not indexed | Run `python rag_indexer.py --stats` to verify |

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