from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.env import MLExperimentEnv
from app.models import Action, Observation, Reward, ExperimentRecord
from app.tasks import TASKS

app = FastAPI(title="ML Experiment Triage Environment")
env = MLExperimentEnv()


class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    action: Action


class TaskDescription(BaseModel):
    task_id: int
    name: str
    description: str
    difficulty: str
    max_steps: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    if request.task_id not in [t.task_id for t in TASKS]:
        raise HTTPException(
            status_code=400, detail=f"Invalid task_id. Must be 1, 2, or 3."
        )
    return env.reset(request.task_id)


@app.post("/step")
def step(request: StepRequest):
    obs, reward, done, info = env.step(request.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks", response_model=List[TaskDescription])
def tasks():
    return [
        TaskDescription(
            task_id=t.task_id,
            name=t.name,
            description=t.description,
            difficulty=t.difficulty,
            max_steps=t.max_steps,
        )
        for t in TASKS
    ]
