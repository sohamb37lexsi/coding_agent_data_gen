from dataclasses import dataclass, field
from typing import List, Optional, Dict

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
    # Quality scoring (populated by quality_scorer)
    composite_score: float = 0.0
    failure_category: str = ""
    reflection: str = ""  # Reflexion verbal summary for this turn

@dataclass
class TaskTrajectory:
    """Represents the full attempt history for a single problem."""
    problem: str
    api_context: str
    attempts: List[TurnAttempt]
    is_successful: bool
    # Best score achieved across all attempts
    best_score: float = 0.0
    synthesis_strategy: str = ""  # "api_doc", "source_code", "mutation"