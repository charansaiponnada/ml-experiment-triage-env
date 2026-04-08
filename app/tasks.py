from dataclasses import dataclass
from typing import Callable, List, Dict, Any
from app.models import ExperimentRecord, Action, Reward

EPSILON = 1e-9


def _clamp_strict(value: float) -> float:
    return max(EPSILON, min(1.0 - EPSILON, value))


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
        if "exp_004" in action.summary.lower():
            return _clamp_strict(0.95)
        return _clamp_strict(0.3)

    if "exp_004" in investigated_exp_ids:
        investigated_count = len(investigated_exp_ids)
        return _clamp_strict(0.2 + (investigated_count * 0.1))

    return _clamp_strict(0.1)


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

    score = (correct_discards / 3.0) * 0.8 + (len(discarded) / 3.0) * 0.2
    score = score - (wrong_discards * 0.05)

    return _clamp_strict(max(0.1, score))


def grade_task_3(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    if action.action_type != "suggest" or not action.suggestion:
        return _clamp_strict(0.2)

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

    if correct == 4:
        return _clamp_strict(0.95)
    elif correct == 3:
        return _clamp_strict(0.7)
    elif correct == 2:
        return _clamp_strict(0.5)
    elif correct == 1:
        return _clamp_strict(0.3)
    else:
        return _clamp_strict(0.2)


def grade_task_4(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    if action.action_type != "compare" or not action.comparison:
        return _clamp_strict(0.2)

    comparison = action.comparison
    analysis = str(comparison.get("analysis", "")).lower()

    score = 0.2
    if "exp_004" in analysis:
        score += 0.3
    if "validation" in analysis or "val_acc" in analysis:
        score += 0.15
    if "generalization" in analysis:
        score += 0.15
    if "tradeoff" in analysis or "trade-off" in analysis:
        score += 0.2

    return _clamp_strict(min(0.95, score))


def grade_task_5(
    experiments: List[ExperimentRecord], action: Action, episode_history: List[Dict]
) -> float:
    if action.action_type != "diagnose" or not action.diagnosis:
        return _clamp_strict(0.2)

    diagnosis = action.diagnosis
    exp_id = str(diagnosis.get("exp_id", "")).lower()
    reason = str(diagnosis.get("reason", "")).lower()
    fix = str(diagnosis.get("fix", "")).lower()

    score = 0.2

    if exp_id == "exp_005":
        if "learning" in reason or "lr" in reason:
            score += 0.25
        if "high" in reason:
            score += 0.1
        if "reduce" in fix or "lower" in fix:
            score += 0.15
    elif exp_id == "exp_008":
        if "memory" in reason or "oom" in reason or "gpu" in reason:
            score += 0.25
        if "batch" in reason or "batch" in fix:
            score += 0.15
    elif exp_id == "exp_003":
        if "plateau" in reason or "schedule" in reason:
            score += 0.25
        if "lr" in fix or "decay" in fix:
            score += 0.15

    return _clamp_strict(min(0.95, score))


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

TASK_4 = Task(
    task_id=4,
    name="compare_experiments",
    description="Compare two experiments and analyze their performance trade-offs.",
    difficulty="medium",
    max_steps=12,
    grader=grade_task_4,
)

TASK_5 = Task(
    task_id=5,
    name="debug_failed_run",
    description="Diagnose a failed experiment run and suggest fixes.",
    difficulty="hard",
    max_steps=15,
    grader=grade_task_5,
)

TASKS = [TASK_1, TASK_2, TASK_3, TASK_4, TASK_5]


def get_task(task_id: int) -> Task:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task {task_id} not found")
