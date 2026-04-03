from openenv_core import Action, Observation, State
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal


class ExperimentRecord(BaseModel):
    exp_id: str
    model_name: str
    learning_rate: float
    epochs: int
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    notes: str
    status: str = "pending"


class MLTriageAction(Action):
    action_type: Literal[
        "investigate", "discard", "suggest", "summarize", "compare", "diagnose"
    ]
    exp_id: Optional[str] = None
    suggestion: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    comparison: Optional[Dict[str, Any]] = None
    diagnosis: Optional[Dict[str, Any]] = None
    serialized_state: Optional[Dict[str, Any]] = None


class MLTriageObservation(Observation):
    experiments: List[ExperimentRecord]
    current_step: int
    max_steps: int
    task_id: int
    task_description: str
    feedback: str
    done: bool
    serialized_state: Optional[Dict[str, Any]] = None


class MLTriageState(State):
    task_id: int
    current_step: int
    max_steps: int
    episode_history: List[Dict]
    done: bool
    total_reward: float
    experiments: List[ExperimentRecord]
    task_description: str
    feedback: str
