from typing import Dict, List, Any, Optional, Tuple
from app.models import ExperimentRecord, Action, Observation, Reward
from app.data import generate_experiments
from app.tasks import get_task, TASKS
import os
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


class MLExperimentEnv:
    def __init__(self):
        self.current_task = None
        self.experiments: List[ExperimentRecord] = []
        self.current_step = 0
        self.max_steps = 0
        self.episode_history: List[Dict] = []
        self.task_description = ""
        self.feedback = ""
        self.done = False

    def reset(self, task_id: int = 1) -> Observation:
        self.current_task = get_task(task_id)
        self.experiments = generate_experiments(task_id)
        self.current_step = 0
        self.max_steps = self.current_task.max_steps
        self.episode_history = []
        self.task_description = self.current_task.description
        self.feedback = (
            f"Task {task_id}: {self.current_task.name}. {self.current_task.description}"
        )
        self.done = False

        return Observation(
            experiments=self.experiments,
            current_step=self.current_step,
            max_steps=self.max_steps,
            task_description=self.task_description,
            feedback=self.feedback,
        )

    def _get_experiment_by_id(self, exp_id: str) -> Optional[ExperimentRecord]:
        for exp in self.experiments:
            if exp.exp_id == exp_id:
                return exp
        return None

    def _is_overfitting(self, exp: ExperimentRecord) -> bool:
        return exp.train_acc > 0.97 and exp.val_acc is not None and exp.val_acc < 0.75

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        reward_value = 0.05
        reward_reason = ""

        if self.done:
            return (
                Observation(
                    experiments=self.experiments,
                    current_step=self.current_step,
                    max_steps=self.max_steps,
                    task_description=self.task_description,
                    feedback=self.feedback,
                ),
                Reward(value=0.05, reason="Episode already done"),
                True,
                {},
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
            reward_value = 0.0
            reward_reason = f"Invalid action type: {action.action_type}"
            valid_action = False
        elif action.action_type in ["investigate", "discard"] and not action.exp_id:
            reward_value = 0.0
            reward_reason = f"Missing exp_id for action: {action.action_type}"
            valid_action = False

        if valid_action:
            if action.action_type == "investigate":
                exp = self._get_experiment_by_id(action.exp_id)
                if exp:
                    if exp.status == "pending":
                        exp.status = "investigated"
                        reward_value = 0.15
                        reward_reason = f"Successfully investigated {action.exp_id}. Details: model={exp.model_name}, lr={exp.learning_rate}, val_acc={exp.val_acc}"
                    else:
                        reward_value = 0.05
                        reward_reason = (
                            f"Warning: {action.exp_id} was already investigated"
                        )
                else:
                    reward_value = 0.0
                    reward_reason = f"Experiment {action.exp_id} not found"

            elif action.action_type == "discard":
                exp = self._get_experiment_by_id(action.exp_id)
                if exp:
                    is_overfitting = self._is_overfitting(exp)
                    if is_overfitting:
                        exp.status = "discarded"
                        reward_value = 0.25
                        reward_reason = f"Correctly discarded {action.exp_id} - overfitting detected (train_acc={exp.train_acc}, val_acc={exp.val_acc})"
                    else:
                        exp.status = "discarded"
                        reward_value = 0.05
                        reward_reason = f"Discarded {action.exp_id}"
                else:
                    reward_value = 0.0
                    reward_reason = f"Experiment {action.exp_id} not found"

            elif action.action_type == "suggest":
                reward_value = 0.3
                reward_reason = f"Suggestion received: {action.suggestion}"

            elif action.action_type == "compare":
                reward_value = 0.3
                reward_reason = f"Comparison received: {action.comparison}"

            elif action.action_type == "diagnose":
                reward_value = 0.3
                reward_reason = f"Diagnosis received: {action.diagnosis}"

            elif action.action_type == "summarize":
                score = self.current_task.grader(
                    self.experiments, action, self.episode_history
                )
                reward_value = score
                if score >= 0.9:
                    reward_reason = "Perfect! Excellent work."
                elif score >= 0.5:
                    reward_reason = f"Good work! Score: {score:.4f}"
                elif score >= 0.1:
                    reward_reason = f"Task completed. Score: {score:.4f}"
                else:
                    reward_reason = f"Task completed with low score: {score:.4f}"
                self.done = True

        self.current_step += 1
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
            reward_value = 0.1
            reward_reason = "Max steps reached. Episode ended."

        if action.action_type != "summarize" and not self.done:
            self.feedback = reward_reason
        elif self.done:
            self.feedback = f"Episode complete. Final score: {reward_value:.4f}"

        return (
            Observation(
                experiments=self.experiments,
                current_step=self.current_step,
                max_steps=self.max_steps,
                task_description=self.task_description,
                feedback=self.feedback,
            ),
            Reward(value=reward_value, reason=reward_reason),
            self.done,
            {"episode_history": self.episode_history},
        )

    def state(self) -> Dict:
        return {
            "task": self.current_task.name if self.current_task else None,
            "experiments": [exp.model_dump() for exp in self.experiments],
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "done": self.done,
            "episode_history": self.episode_history,
        }
