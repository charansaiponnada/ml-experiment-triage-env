from pydantic import BaseModel, Field
from typing import List, Optional, Dict


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


class Action(BaseModel):
    action_type: str
    exp_id: Optional[str] = None
    suggestion: Optional[Dict] = None
    summary: Optional[str] = None


class Observation(BaseModel):
    experiments: List[ExperimentRecord]
    current_step: int
    max_steps: int
    task_description: str
    feedback: str


class Reward(BaseModel):
    value: float
    reason: str
