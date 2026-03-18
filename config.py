import os
from openai import OpenAI

# ==========================================
# Configuration & Setup
# ==========================================
# Local vLLM server running Qwen2.5-Coder-14B-Instruct
# Context window: 32,768 tokens
# Start server:
#   conda activate qwen_vllm
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-Coder-14B-Instruct \
#       --host 0.0.0.0 --port 8000 \
#       --gpu-memory-utilization 0.5

API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-14B-Instruct")

# Instantiate the client globally
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)