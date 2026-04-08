"""
ML Experiment Triage Inference Script
=====================================

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import os
import json
import requests
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "ml-experiment-triage"
SUCCESS_SCORE_THRESHOLD = 0.1
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
EPSILON = 1e-9

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"id": 1, "name": "find_best_experiment", "max_steps": MAX_STEPS},
    {"id": 2, "name": "identify_overfitting", "max_steps": MAX_STEPS},
    {"id": 3, "name": "suggest_next_experiment", "max_steps": MAX_STEPS},
    {"id": 4, "name": "compare_experiments", "max_steps": MAX_STEPS},
    {"id": 5, "name": "debug_failed_run", "max_steps": MAX_STEPS},
]

SYSTEM_PROMPT = """You are an ML experiment analysis agent.
Analyze the experiment table and take actions.

Respond ONLY with valid JSON (no markdown, no explanation):

{"action_type": "investigate", "exp_id": "exp_001"}
{"action_type": "discard", "exp_id": "exp_001"}
{"action_type": "suggest", "suggestion": {"model_name": "resnet50", "learning_rate": 0.001, "epochs": 50}}
{"action_type": "summarize", "summary": "exp_004"}

Rules:
- Investigate at least 2 experiments before summarizing
- Discard if train_acc > 0.97 AND val_acc < 0.75 (overfitting)
- For Task 3: suggest the best next config based on patterns
- For summarize: provide the exp_id of the single best experiment
- ONLY respond with JSON. Nothing else."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str = None
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_action(obs_text: str, history: list) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({"role": "user", "content": obs_text})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw = resp.choices[0].message.content.strip()
    return json.loads(raw)


def run_task(task: dict) -> tuple:
    task_id = task["id"]
    task_name = task["name"]
    max_steps = task["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(
            f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=60
        )
    except Exception as e:
        error_msg = f"reset_error:{str(e)}"
        log_step(1, "reset()", EPSILON, True, error_msg)
        log_end(False, 0, EPSILON, [])
        return EPSILON

    if resp.status_code != 200:
        error_msg = f"reset_failed:{resp.status_code}"
        log_step(1, "reset()", EPSILON, True, error_msg)
        log_end(False, 0, EPSILON, [])
        return EPSILON

    obs = resp.json().get("observation", {})

    history = []
    rewards = []
    steps = 0
    success = False
    serialized_state = obs.get("serialized_state", {})

    for step in range(1, max_steps + 1):
        if obs.get("done", False):
            break

        exps = obs.get("experiments", [])
        obs_text = f"Task: {obs.get('task_description', '')}\n"
        obs_text += f"Feedback: {obs.get('feedback', '')}\n"
        obs_text += f"Step: {obs.get('current_step', step)}/{obs.get('max_steps', max_steps)}\n\n"
        obs_text += "Experiments:\n"
        for e in exps:
            obs_text += f"  {e['exp_id']}: model={e['model_name']} lr={e['learning_rate']} epochs={e['epochs']} train_acc={e['train_acc']} val_acc={e['val_acc']} status={e['status']}\n"

        action_dict = get_action(obs_text, history)
        action_type = action_dict.get("action_type", "investigate")

        action_str = json.dumps(action_dict)

        if serialized_state:
            action_dict["serialized_state"] = serialized_state

        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step", json={"action": action_dict}, timeout=60
            )
            result = step_resp.json()
        except Exception as e:
            log_step(step, action_str, EPSILON, True, f"request_error:{str(e)}")
            log_end(False, step, EPSILON, rewards)
            return EPSILON

        reward = result.get("reward", {})
        reward_val = (
            float(reward.get("value", 0.0))
            if isinstance(reward, dict)
            else float(reward or 0.0)
        )
        done = result.get("done", False)
        obs = result.get("observation", {})

        serialized_state = obs.get("serialized_state", {})

        rewards.append(reward_val)
        steps = step

        history.append({"obs": obs_text, "action": json.dumps(action_dict)})

        log_step(step=step, action=action_str, reward=reward_val, done=done, error=None)

        if done:
            success = reward_val >= SUCCESS_SCORE_THRESHOLD
            break

    score = min(max(sum(rewards), EPSILON), 1.0 - EPSILON)

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return score


if __name__ == "__main__":
    # Mandatory test call to ensure API connection works
    try:
        test_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Respond with just 'ok'"}],
            max_tokens=10,
        )
        print(f"[LLM_TEST] {test_resp.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"[LLM_TEST_ERROR] {type(e).__name__}: {e}", flush=True)

    scores = []
    for task in TASKS:
        score = run_task(task)
        scores.append(score)

    print(f"\nFinal scores: {scores}", flush=True)
    print(f"Average: {sum(scores) / len(scores):.4f}", flush=True)
