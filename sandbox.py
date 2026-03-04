import os
import re
import ast
import sys
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Tuple, List, Optional

# ==========================================
# Sandbox Environment with Dependency Management
# ==========================================

SANDBOX_VENV_DIR = os.getenv("SANDBOX_VENV_DIR", ".sandbox_venv")

# Core packages to install in the sandbox venv
SANDBOX_DEPENDENCIES = [
    "aligntune",       # Target library (https://pypi.org/project/aligntune/)
    "torch",           # Required by aligntune
    "transformers",    # HF ecosystem
    "datasets",        # HF datasets
    "accelerate",      # Training utilities
]


class SandboxEnvironment:
    """Manages an isolated Python venv with pre-installed dependencies."""

    def __init__(self, venv_dir: str = SANDBOX_VENV_DIR, dependencies: Optional[List[str]] = None):
        self.venv_dir = Path(venv_dir).resolve()
        self.dependencies = dependencies or SANDBOX_DEPENDENCIES
        self._python_path: Optional[str] = None

    @property
    def python_path(self) -> str:
        if self._python_path is None:
            if sys.platform == "win32":
                self._python_path = str(self.venv_dir / "Scripts" / "python.exe")
            else:
                self._python_path = str(self.venv_dir / "bin" / "python")
        return self._python_path

    @property
    def is_ready(self) -> bool:
        return Path(self.python_path).exists()

    def setup(self, force_rebuild: bool = False) -> None:
        """Creates the venv and installs dependencies. Idempotent unless force_rebuild=True."""
        if self.is_ready and not force_rebuild:
            print(f"[Sandbox] Venv already exists at {self.venv_dir}")
            return

        print(f"[Sandbox] Creating isolated venv at {self.venv_dir}...")
        venv.create(str(self.venv_dir), with_pip=True, clear=force_rebuild)

        # Upgrade pip first
        self._run_pip(["install", "--upgrade", "pip"])

        # Install all dependencies
        print(f"[Sandbox] Installing dependencies: {', '.join(self.dependencies)}")
        self._run_pip(["install"] + self.dependencies)
        print("[Sandbox] Environment ready.")

    def _run_pip(self, args: List[str]) -> None:
        result = subprocess.run(
            [self.python_path, "-m", "pip"] + args,
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip failed: {result.stderr[:500]}")


class CodeSanitizer:
    """Pre-execution static checks to reject obviously broken code before wasting sandbox time."""

    # Patterns that indicate notebook/shell syntax leaking into generated code
    FORBIDDEN_PATTERNS = [
        (r"^\s*!",          "Shell escape (!) detected — not valid Python"),
        (r"^\s*%",          "IPython magic (%) detected — not valid Python"),
        (r"subprocess\.run\(.+pip\s+install", "Runtime pip install detected — deps are pre-installed"),
    ]

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"

    @classmethod
    def check_forbidden_patterns(cls, code: str) -> Tuple[bool, str]:
        for line_num, line in enumerate(code.splitlines(), 1):
            for pattern, msg in cls.FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    return False, f"Line {line_num}: {msg}"
        return True, ""

    @classmethod
    def sanitize(cls, code: str) -> Tuple[bool, str, str]:
        """
        Returns (is_valid, cleaned_code, error_message).
        Strips common junk; rejects if fundamentally broken.
        """
        # Strip notebook shell lines (! prefix) rather than hard-failing — 
        # the model often prepends these but the rest of the code is fine
        lines = code.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("!") or stripped.startswith("%"):
                continue  # silently drop notebook artifacts
            # Strip inline subprocess pip installs
            if re.search(r"subprocess\.run\(.+pip\s+install", line):
                continue
            cleaned.append(line)

        cleaned_code = "\n".join(cleaned)

        # Check syntax on cleaned version
        valid, err = cls.check_syntax(cleaned_code)
        if not valid:
            return False, code, err

        return True, cleaned_code, ""


class CodeSandbox:
    """Executes code in an isolated venv subprocess with pre-installed dependencies."""

    def __init__(self, timeout: int = 120, environment: Optional[SandboxEnvironment] = None):
        self.timeout = timeout
        self.env = environment or SandboxEnvironment()

        if not self.env.is_ready:
            self.env.setup()

    def execute(self, code: str) -> Tuple[bool, str, str]:
        """
        Sanitizes, then executes code in the sandbox venv.
        Returns (success_bool, stdout, stderr).
        """
        # Pre-execution sanitization
        is_valid, cleaned_code, sanitize_err = CodeSanitizer.sanitize(code)
        if not is_valid:
            return False, "", f"[Pre-execution Filter] {sanitize_err}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(cleaned_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [self.env.python_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    # GPU passthrough: inherits host CUDA_VISIBLE_DEVICES by default.
                    # Override via SANDBOX_CUDA_DEVICES env var:
                    #   export SANDBOX_CUDA_DEVICES=""      -> force CPU only
                    #   export SANDBOX_CUDA_DEVICES="0"     -> use GPU 0
                    #   (unset)                             -> inherit from host
                    **({"CUDA_VISIBLE_DEVICES": os.environ["SANDBOX_CUDA_DEVICES"]}
                       if "SANDBOX_CUDA_DEVICES" in os.environ else {}),
                    # Cache HF models in a shared location to avoid re-downloads
                    "HF_HOME": os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                    "TRANSFORMERS_VERBOSITY": "error",  # Suppress noisy HF logs
                }
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Execution timed out after {self.timeout}s."
        except Exception as e:
            return False, "", f"Sandbox error: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def extract_code_blocks(text: str) -> str:
        """Extracts python code from markdown formatting returned by the LLM."""
        if "```python" in text:
            parts = text.split("```python")
            return parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            return parts[1].strip()
        return text.strip()


# ==========================================
# CLI: Setup the sandbox environment ahead of time
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage the code sandbox environment")
    parser.add_argument("--setup", action="store_true", help="Create/rebuild the sandbox venv")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if venv exists")
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    env = SandboxEnvironment()

    if args.setup:
        env.setup(force_rebuild=args.force)

    if args.test:
        if not env.is_ready:
            env.setup()
        sandbox = CodeSandbox(environment=env)
        test_code = "from aligntune import Trainer\nprint('aligntune import OK')"
        ok, stdout, stderr = sandbox.execute(test_code)
        print(f"Test passed: {ok}")
        print(f"stdout: {stdout}")
        if stderr:
            print(f"stderr: {stderr}")