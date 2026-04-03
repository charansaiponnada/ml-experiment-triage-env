from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from openenv_core import Environment
from pydantic import BaseModel
import json


@dataclass
class Task:
    task_id: int
    name: str
    description: str
    difficulty: str
    max_steps: int
    grader: Any


def grade_task_1(
    experiments: List, episode_history: List[Dict], summary: Optional[str]
) -> float:
    investigated_exp_ids = set()
    for step in episode_history:
        if step.get("action", {}).get("action_type") == "investigate":
            exp_id = step.get("action", {}).get("exp_id")
            if exp_id:
                investigated_exp_ids.add(exp_id)

    if summary and "exp_004" in summary:
        return 1.0

    if "exp_004" in investigated_exp_ids:
        return 0.5

    return 0.0


def grade_task_2(
    experiments: List, episode_history: List[Dict], exp_id: Optional[str]
) -> float:
    overfitting_exp_ids = {"exp_002", "exp_006", "exp_009"}

    discarded = set()
    for step in episode_history:
        if step.get("action", {}).get("action_type") == "discard":
            eid = step.get("action", {}).get("exp_id")
            if eid:
                discarded.add(eid)

    if exp_id:
        discarded.add(exp_id)

    correct_discards = len(discarded & overfitting_exp_ids)
    wrong_discards = len(discarded - overfitting_exp_ids)

    score = (correct_discards / 3.0) - (wrong_discards * 0.1)

    return max(0.0, min(1.0, score))


def grade_task_3(suggestion: Optional[Dict]) -> float:
    if not suggestion:
        return 0.0

    ground_truth = {"learning_rate": 0.001, "epochs": 50, "model_name": "resnet50"}

    correct = 0

    if suggestion.get("learning_rate") == ground_truth["learning_rate"]:
        correct += 1
    if suggestion.get("epochs") == ground_truth["epochs"]:
        correct += 1
    if suggestion.get("model") == ground_truth["model_name"]:
        correct += 1
    if suggestion.get("model_name") == ground_truth["model_name"]:
        correct += 1

    if correct >= 3:
        return 1.0
    elif correct == 2:
        return 0.6
    elif correct == 1:
        return 0.3
    else:
        return 0.0


def grade_task_4(comparison: Optional[Dict]) -> float:
    if not comparison:
        return 0.0

    analysis = comparison.get("analysis", "").lower()

    score = 0.0

    if "exp_004" in analysis:
        score += 0.4
    if "validation" in analysis or "val_acc" in analysis:
        score += 0.2
    if "generalization" in analysis:
        score += 0.2
    if "tradeoff" in analysis or "trade-off" in analysis:
        score += 0.2

    return min(1.0, score)


def grade_task_5(diagnosis: Optional[Dict]) -> float:
    if not diagnosis:
        return 0.0

    exp_id = diagnosis.get("exp_id", "")
    reason = diagnosis.get("reason", "").lower()
    fix = diagnosis.get("fix", "").lower()

    score = 0.0

    if exp_id == "exp_005":
        if "learning rate" in reason or "lr" in reason:
            score += 0.35
        if "high" in reason or "too high" in reason:
            score += 0.1
        if "reduce" in fix or "lower" in fix:
            score += 0.1
    elif exp_id == "exp_008":
        if "memory" in reason or "oom" in reason or "gpu" in reason:
            score += 0.35
        if "batch" in reason:
            score += 0.1
    elif exp_id == "exp_003":
        if "plateau" in reason or "schedule" in reason:
            score += 0.35
        if "lr" in fix or "decay" in fix:
            score += 0.1

    return min(1.0, score)


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
    description="Compare two experiments and explain the tradeoffs between them. Identify which is better for production use.",
    difficulty="medium",
    max_steps=12,
    grader=grade_task_4,
)

TASK_5 = Task(
    task_id=5,
    name="debug_failed_run",
    description="Diagnose why certain experiments failed or underperformed. Identify the root cause and suggest a fix.",
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


def generate_experiments(task_id: int):
    from models import ExperimentRecord, MLTriageObservation, MLTriageState

    if task_id == 1:
        return [
            ExperimentRecord(
                exp_id="exp_001",
                model_name="resnet18",
                learning_rate=0.01,
                epochs=20,
                train_acc=0.89,
                val_acc=0.71,
                train_loss=0.33,
                val_loss=0.81,
                notes="baseline model",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_002",
                model_name="resnet18",
                learning_rate=0.005,
                epochs=30,
                train_acc=0.92,
                val_acc=0.78,
                train_loss=0.24,
                val_loss=0.65,
                notes="increased epochs",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_003",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.94,
                val_acc=0.85,
                train_loss=0.18,
                val_loss=0.42,
                notes="lower lr, more epochs",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_004",
                model_name="resnet34",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.96,
                val_acc=0.94,
                train_loss=0.12,
                val_loss=0.18,
                notes="best configuration",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_005",
                model_name="resnet18",
                learning_rate=0.02,
                epochs=15,
                train_acc=0.85,
                val_acc=0.69,
                train_loss=0.42,
                val_loss=0.92,
                notes="higher lr led to instability",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_006",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=40,
                train_acc=0.91,
                val_acc=0.82,
                train_loss=0.26,
                val_loss=0.51,
                notes="good but not best",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_007",
                model_name="resnet18",
                learning_rate=0.0005,
                epochs=60,
                train_acc=0.93,
                val_acc=0.88,
                train_loss=0.21,
                val_loss=0.35,
                notes="very slow training but converges well",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_008",
                model_name="resnet18",
                learning_rate=0.003,
                epochs=25,
                train_acc=0.90,
                val_acc=0.76,
                train_loss=0.29,
                val_loss=0.68,
                notes="moderate performance",
                status="pending",
            ),
        ]

    elif task_id == 2:
        return [
            ExperimentRecord(
                exp_id="exp_001",
                model_name="resnet18",
                learning_rate=0.01,
                epochs=20,
                train_acc=0.91,
                val_acc=0.75,
                train_loss=0.27,
                val_loss=0.72,
                notes="baseline",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_002",
                model_name="resnet34",
                learning_rate=0.01,
                epochs=100,
                train_acc=0.99,
                val_acc=0.68,
                train_loss=0.03,
                val_loss=1.25,
                notes="overtrained - clear overfitting",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_003",
                model_name="resnet18",
                learning_rate=0.005,
                epochs=30,
                train_acc=0.93,
                val_acc=0.80,
                train_loss=0.21,
                val_loss=0.58,
                notes="good generalization",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_004",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.94,
                val_acc=0.86,
                train_loss=0.18,
                val_loss=0.40,
                notes="well regularized",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_005",
                model_name="resnet18",
                learning_rate=0.01,
                epochs=15,
                train_acc=0.88,
                val_acc=0.73,
                train_loss=0.35,
                val_loss=0.78,
                notes="underfit slightly",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_006",
                model_name="resnet50",
                learning_rate=0.01,
                epochs=80,
                train_acc=0.98,
                val_acc=0.71,
                train_loss=0.05,
                val_loss=1.05,
                notes="overfitting - high capacity",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_007",
                model_name="resnet18",
                learning_rate=0.002,
                epochs=40,
                train_acc=0.92,
                val_acc=0.81,
                train_loss=0.24,
                val_loss=0.55,
                notes="balanced",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_008",
                model_name="resnet18",
                learning_rate=0.005,
                epochs=25,
                train_acc=0.90,
                val_acc=0.77,
                train_loss=0.29,
                val_loss=0.66,
                notes="reasonable",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_009",
                model_name="vgg16",
                learning_rate=0.01,
                epochs=60,
                train_acc=0.98,
                val_acc=0.69,
                train_loss=0.04,
                val_loss=1.32,
                notes="VGG overfitting badly",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_010",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=35,
                train_acc=0.89,
                val_acc=0.79,
                train_loss=0.32,
                val_loss=0.62,
                notes="conservative training",
                status="pending",
            ),
        ]

    elif task_id == 3:
        return [
            ExperimentRecord(
                exp_id="exp_001",
                model_name="resnet18",
                learning_rate=0.01,
                epochs=20,
                train_acc=0.88,
                val_acc=0.74,
                train_loss=0.35,
                val_loss=0.75,
                notes="baseline run",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_002",
                model_name="resnet18",
                learning_rate=0.005,
                epochs=30,
                train_acc=0.91,
                val_acc=0.79,
                train_loss=0.27,
                val_loss=0.61,
                notes="improved over baseline",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_003",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=40,
                train_acc=0.93,
                val_acc=0.83,
                train_loss=0.21,
                val_loss=0.48,
                notes="lower lr better",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_004",
                model_name="resnet34",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.95,
                val_acc=0.88,
                train_loss=0.15,
                val_loss=0.35,
                notes="good results so far",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_005",
                model_name="resnet18",
                learning_rate=0.0005,
                epochs=60,
                train_acc=0.94,
                val_acc=0.86,
                train_loss=0.18,
                val_loss=0.40,
                notes="very slow but stable",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_006",
                model_name="resnet50",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.96,
                val_acc=0.91,
                train_loss=0.11,
                val_loss=0.27,
                notes="promising",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_007",
                model_name="resnet18",
                learning_rate=0.002,
                epochs=35,
                train_acc=0.92,
                val_acc=0.80,
                train_loss=0.24,
                val_loss=0.58,
                notes="中等结果",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_008",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=45,
                train_acc=0.0,
                val_acc=0.0,
                train_loss=0.0,
                val_loss=0.0,
                notes="???",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_009",
                model_name="resnet34",
                learning_rate=0.0008,
                epochs=55,
                train_acc=0.95,
                val_acc=0.89,
                train_loss=0.14,
                val_loss=0.32,
                notes="still running",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_010",
                model_name="resnet50",
                learning_rate=0.0012,
                epochs=48,
                train_acc=0.0,
                val_acc=0.0,
                train_loss=0.0,
                val_loss=0.0,
                notes="crashed - OOM",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_011",
                model_name="resnet18",
                learning_rate=0.001,
                epochs=50,
                train_acc=0.94,
                val_acc=0.87,
                train_loss=0.17,
                val_loss=0.38,
                notes="incomplete - stopped early",
                status="pending",
            ),
            ExperimentRecord(
                exp_id="exp_012",
                model_name="resnet18",
                learning_rate=0.0003,
                epochs=70,
                train_acc=0.0,
                val_acc=0.0,
                train_loss=0.0,
                val_loss=0.0,
                notes="noisy - data augmentation issues",
                status="pending",
            ),
        ]

    return []


def serialize_experiment(exp) -> Dict:
    return {
        "exp_id": exp.exp_id,
        "model_name": exp.model_name,
        "learning_rate": exp.learning_rate,
        "epochs": exp.epochs,
        "train_acc": exp.train_acc,
        "val_acc": exp.val_acc,
        "train_loss": exp.train_loss,
        "val_loss": exp.val_loss,
        "notes": exp.notes,
        "status": exp.status,
    }


def deserialize_experiments(exp_data: List[Dict]) -> List:
    from models import ExperimentRecord, MLTriageObservation, MLTriageState

    return [ExperimentRecord(**e) for e in exp_data]


class MLTriageEnvironment(Environment):
    def __init__(self, serialized_state: Optional[Dict[str, Any]] = None):
        if serialized_state:
            self._restore_state(serialized_state)
        else:
            self.task_id = 1
            self.current_task = None
            self.experiments = []
            self.current_step = 0
            self.max_steps = 0
            self.episode_history = []
            self.task_description = ""
            self.feedback = ""
            self.done = False
            self.total_reward = 0.0

    def _get_state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "episode_history": self.episode_history,
            "done": self.done,
            "total_reward": self.total_reward,
            "experiments": [serialize_experiment(e) for e in self.experiments],
            "task_description": self.task_description,
            "feedback": self.feedback,
            "current_task_name": self.current_task.name if self.current_task else None,
        }

    def _restore_state(self, state: Dict[str, Any]):
        self.task_id = state.get("task_id", 1)
        self.current_step = state.get("current_step", 0)
        self.max_steps = state.get("max_steps", 0)
        self.episode_history = state.get("episode_history", [])
        self.done = state.get("done", False)
        self.total_reward = state.get("total_reward", 0.0)
        self.experiments = deserialize_experiments(state.get("experiments", []))
        self.task_description = state.get("task_description", "")
        self.feedback = state.get("feedback", "")
        task_name = state.get("current_task_name")
        if task_name:
            self.current_task = get_task(self.task_id)
        else:
            self.current_task = None

    def reset(
        self,
        task_id: int = 1,
        serialized_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if serialized_state:
            self._restore_state(serialized_state)
            self.task_id = task_id or self.task_id
        else:
            self.task_id = task_id
            self.current_task = get_task(task_id)
            self.experiments = generate_experiments(task_id)
            self.current_step = 0
            self.max_steps = self.current_task.max_steps
            self.episode_history = []
            self.task_description = self.current_task.description
            self.feedback = f"Task {task_id}: {self.current_task.name}. {self.current_task.description}"
            self.done = False
            self.total_reward = 0.0

        from models import MLTriageObservation, MLTriageState

        return MLTriageObservation(
            experiments=self.experiments,
            current_step=self.current_step,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=self.task_description,
            feedback=self.feedback,
            serialized_state=self._get_state(),
        )

    def _get_experiment_by_id(self, exp_id: str):
        for exp in self.experiments:
            if exp.exp_id == exp_id:
                return exp
        return None

    def _is_overfitting(self, exp) -> bool:
        return exp.train_acc > 0.97 and exp.val_acc is not None and exp.val_acc < 0.75

    def step(self, action, serialized_state: Optional[Dict[str, Any]] = None, **kwargs):
        # Extract serialized_state from action object if not provided as parameter
        if serialized_state is None and hasattr(action, "serialized_state"):
            serialized_state = action.serialized_state

        if serialized_state:
            self._restore_state(serialized_state)

        reward_value = 0.0
        reward_reason = ""

        if self.done:
            self.feedback = (
                "Episode already complete. Please reset to start a new episode."
            )
            from models import MLTriageObservation

            return MLTriageObservation(
                experiments=self.experiments,
                current_step=self.current_step,
                max_steps=self.max_steps,
                task_id=self.task_id,
                task_description=self.task_description,
                feedback=self.feedback,
                done=self.done,
                serialized_state=self._get_state(),
            )

        valid_action = True

        if action.action_type not in [
            "investigate",
            "discard",
            "suggest",
            "summarize",
            "compare",
            "diagnose",
        ]:
            reward_value = -0.05
            reward_reason = f"Invalid action type: {action.action_type}"
            valid_action = False
        elif action.action_type in ["investigate", "discard"] and not action.exp_id:
            reward_value = -0.05
            reward_reason = f"Missing exp_id for action: {action.action_type}"
            valid_action = False
        elif action.action_type == "compare" and not action.comparison:
            reward_value = -0.05
            reward_reason = "Missing comparison data"
            valid_action = False
        elif action.action_type == "diagnose" and not action.diagnosis:
            reward_value = -0.05
            reward_reason = "Missing diagnosis data"
            valid_action = False

        if valid_action:
            if action.action_type == "investigate":
                exp = self._get_experiment_by_id(action.exp_id)
                if exp:
                    if exp.status == "pending":
                        exp.status = "investigated"
                        reward_value = 0.1
                        reward_reason = f"Successfully investigated {action.exp_id}. Details: model={exp.model_name}, lr={exp.learning_rate}, val_acc={exp.val_acc}"
                    else:
                        reward_value = 0.0
                        reward_reason = (
                            f"Warning: {action.exp_id} was already investigated"
                        )
                else:
                    reward_value = -0.05
                    reward_reason = f"Experiment {action.exp_id} not found"

            elif action.action_type == "discard":
                exp = self._get_experiment_by_id(action.exp_id)
                if exp:
                    is_overfitting = self._is_overfitting(exp)
                    if is_overfitting:
                        exp.status = "discarded"
                        reward_value = 0.15
                        reward_reason = f"Correctly discarded {action.exp_id} - overfitting detected"
                    else:
                        exp.status = "discarded"
                        reward_value = -0.1
                        reward_reason = f"Incorrectly discarded {action.exp_id} - it was not overfitting"
                else:
                    reward_value = -0.05
                    reward_reason = f"Experiment {action.exp_id} not found"

            elif action.action_type == "suggest":
                if action.suggestion:
                    ground_truth = {
                        "learning_rate": 0.001,
                        "epochs": 50,
                        "model_name": "resnet50",
                    }
                    suggestion = action.suggestion
                    correct = 0

                    if suggestion.get("learning_rate") == ground_truth["learning_rate"]:
                        correct += 1
                    if suggestion.get("epochs") == ground_truth["epochs"]:
                        correct += 1
                    if (
                        suggestion.get("model") == ground_truth["model_name"]
                        or suggestion.get("model_name") == ground_truth["model_name"]
                    ):
                        correct += 1

                    if correct == 3:
                        reward_value = 0.5
                        reward_reason = (
                            "Excellent suggestion! Matches ground truth exactly."
                        )
                    elif correct == 2:
                        reward_value = 0.35
                        reward_reason = (
                            "Good suggestion. Partially matches ground truth."
                        )
                    elif correct == 1:
                        reward_value = 0.15
                        reward_reason = "Partial suggestion. Some fields match."
                    else:
                        reward_value = 0.0
                        reward_reason = "Suggestion does not match ground truth well."
                else:
                    reward_value = -0.05
                    reward_reason = "Missing suggestion data"

            elif action.action_type == "compare":
                if action.comparison:
                    reward_value = 0.15
                    exp_a = action.comparison.get("exp_a", "")
                    exp_b = action.comparison.get("exp_b", "")
                    reward_reason = f"Comparing {exp_a} vs {exp_b}. Analysis noted."
                else:
                    reward_value = -0.05
                    reward_reason = "Missing comparison data"

            elif action.action_type == "diagnose":
                if action.diagnosis:
                    reward_value = 0.15
                    exp_id = action.diagnosis.get("exp_id", "")
                    reason = action.diagnosis.get("reason", "")
                    reward_reason = f"Diagnosing {exp_id}: {reason[:50]}"
                else:
                    reward_value = -0.05
                    reward_reason = "Missing diagnosis data"

            elif action.action_type == "summarize":
                if self.task_id == 1:
                    score = self.current_task.grader(
                        self.experiments, self.episode_history, action.summary
                    )
                elif self.task_id == 2:
                    score = self.current_task.grader(
                        self.experiments, self.episode_history, action.exp_id
                    )
                elif self.task_id == 3:
                    score = self.current_task.grader(None, None, action.suggestion)
                elif self.task_id == 4:
                    score = self.current_task.grader(None, None, action.comparison)
                elif self.task_id == 5:
                    score = self.current_task.grader(None, None, action.diagnosis)
                else:
                    score = 0.0

                reward_value = score
                if score >= 1.0:
                    reward_reason = "Perfect! Correctly identified the best experiment."
                elif score >= 0.5:
                    reward_reason = f"Good work! Score: {score}"
                else:
                    reward_reason = f"Task completed with score: {score}"
                self.done = True

        self.current_step += 1
        self.total_reward += reward_value
        self.episode_history.append(
            {
                "step": self.current_step,
                "action": action.model_dump(),
                "reward": reward_value,
                "reason": reward_reason,
            }
        )

        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            reward_value = 0.0
            reward_reason = "Max steps reached. Episode ended."

        if action.action_type != "summarize" and not self.done:
            self.feedback = reward_reason
        elif self.done:
            self.feedback = f"Episode complete. Final score: {reward_value}"

        from models import MLTriageObservation, MLTriageState

        return MLTriageObservation(
            experiments=self.experiments,
            current_step=self.current_step,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=self.task_description,
            feedback=self.feedback,
            done=self.done,
            serialized_state=self._get_state(),
        )

    @property
    def state(self):
        from models import MLTriageState

        return {
            "episode_id": None,
            "step_count": self.current_step,
            "task_id": self.task_id,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "episode_history": self.episode_history,
            "done": self.done,
            "total_reward": self.total_reward,
        }
