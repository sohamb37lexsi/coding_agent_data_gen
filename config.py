import os
from openai import OpenAI

# ==========================================
# Configuration & Setup
# ==========================================
# Point this to your local vLLM/Ollama instance running Qwen2.5-Coder 
# or use a cloud provider like OpenAI / Anthropic.

API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") # E.g., Qwen/Qwen2.5-Coder-7B-Instruct

# Instantiate the client globally for the modules to share
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)