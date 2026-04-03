from dataclasses import dataclass
from typing import Callable, List, Dict, Any
from app.models import ExperimentRecord, Action, Reward


@dataclass
class Task:
    task_id: int
    name: str
    description: str
    difficulty: str
    max_steps: int
    grader: Callable


def grade_task_1(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    investigated_exp_ids = set()
    for step in episode_history:
        if step.get("action", {}).get("action_type") == "investigate":
            exp_id = step.get("action", {}).get("exp_id")
            if exp_id:
                investigated_exp_ids.add(exp_id)

    if action.action_type == "summarize" and action.summary:
        if "exp_004" in action.summary:
            return 1.0
        return 0.0

    if "exp_004" in investigated_exp_ids:
        return 0.5

    return 0.0


def grade_task_2(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    overfitting_exp_ids = {"exp_002", "exp_006", "exp_009"}

    discarded = set()
    for step in episode_history:
        if step.get("action", {}).get("action_type") == "discard":
            exp_id = step.get("action", {}).get("exp_id")
            if exp_id:
                discarded.add(exp_id)

    if action.action_type == "discard" and action.exp_id:
        discarded.add(action.exp_id)

    correct_discards = len(discarded & overfitting_exp_ids)
    wrong_discards = len(discarded - overfitting_exp_ids)

    score = (correct_discards / 3.0) - (wrong_discards * 0.1)

    return max(0.0, min(1.0, score))


def grade_task_3(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    if action.action_type != "suggest" or not action.suggestion:
        return 0.0

    ground_truth = {"learning_rate": 0.001, "epochs": 50, "model_name": "resnet50"}

    suggestion = action.suggestion
    correct = 0

    if suggestion.get("learning_rate") == ground_truth["learning_rate"]:
        correct += 1
    if suggestion.get("epochs") == ground_truth["epochs"]:
        correct += 1
    if suggestion.get("model") == ground_truth["model_name"]:
        correct += 1
    if suggestion.get("model_name") == ground_truth["model_name"]:
        correct += 1

    if correct == 3:
        return 1.0
    elif correct == 2:
        return 0.6
    elif correct == 1:
        return 0.3
    else:
        return 0.0


TASK_1 = Task(
    task_id=1,
    name="find_best_experiment",
    description="You are given 8 ML experiments. Use investigate() to explore runs and summarize() with the best exp_id.",
    difficulty="easy",
    max_steps=10,
    grader=grade_task_1,
)

TASK_2 = Task(
    task_id=2,
    name="identify_overfitting",
    description="Find and discard all overfitting experiments. An experiment is overfitting if train_acc > 0.97 and val_acc < 0.75.",
    difficulty="medium",
    max_steps=15,
    grader=grade_task_2,
)

TASK_3 = Task(
    task_id=3,
    name="suggest_next_experiment",
    description="Analyze incomplete experiment results and suggest the next hyperparameter configuration to try.",
    difficulty="hard",
    max_steps=20,
    grader=grade_task_3,
)

TASKS = [TASK_1, TASK_2, TASK_3]


def get_task(task_id: int) -> Task:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task {task_id} not found")
