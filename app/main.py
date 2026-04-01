from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.env import MLExperimentEnv
from app.models import Action, Observation
from app.tasks import TASKS
import os

app = FastAPI(title="ML Experiment Triage Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=TEMPLATES_DIR), name="static")

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


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


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
