from dataclasses import dataclass
from typing import List, Optional

# ==========================================
# Core Data Models
# ==========================================

@dataclass
class TurnAttempt:
    """Represents a single step in the ReAct loop."""
    turn_number: int
    plan: str
    code: str
    traceback: Optional[str] = None
    success: bool = False

@dataclass
class TaskTrajectory:
    """Represents the full attempt history for a single problem."""
    problem: str
    api_context: str
    attempts: List[TurnAttempt]
    is_successful: bool