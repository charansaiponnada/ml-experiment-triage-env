import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"id": 1, "name": "find_best_experiment", "max_steps": 10, "max_reward": 1.1},
    {"id": 2, "name": "identify_overfitting", "max_steps": 15, "max_reward": 0.6},
    {"id": 3, "name": "suggest_next_experiment", "max_steps": 20, "max_reward": 0.5},
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


def get_action(obs_text: str, history: list) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({"role": "user", "content": obs_text})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        return {"action_type": "investigate", "exp_id": "exp_001"}


def run_task(task: dict) -> float:
    task_id = task["id"]
    task_name = task["name"]
    max_steps = task["max_steps"]

    print(
        f"[START] task={task_name} env=ml-experiment-triage model={MODEL_NAME}",
        flush=True,
    )

    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    obs = resp.json().get("observation", {})

    history = []
    total_reward = 0.0
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
            obs_text += (
                f"  {e['exp_id']}: model={e['model_name']} "
                f"lr={e['learning_rate']} epochs={e['epochs']} "
                f"train_acc={e['train_acc']} val_acc={e['val_acc']} "
                f"status={e['status']}\n"
            )

        action_dict = get_action(obs_text, history)
        action_type = action_dict.get("action_type", "investigate")

        # Pass serialized_state in action for stateless HTTP
        if serialized_state:
            action_dict["serialized_state"] = serialized_state

        step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action_dict})
        result = step_resp.json()

        reward = result.get("reward", {})
        reward_val = (
            reward.get("value", 0.0) if isinstance(reward, dict) else float(reward)
        )
        done = result.get("done", False)
        obs = result.get("observation", {})

        # Get updated serialized_state for next step
        serialized_state = obs.get("serialized_state", {})

        total_reward += reward_val
        steps = step

        history.append({"obs": obs_text, "action": json.dumps(action_dict)})

        print(
            f"[STEP] step={step} action={action_type} reward={reward_val:.4f} done={done}",
            flush=True,
        )

        if done:
            success = reward_val >= 0.5
            break

    score = min(max(total_reward / task["max_reward"], 0.0), 1.0)
    print(f"[END] success={success} steps={steps} score={score:.4f}", flush=True)
    return score


if __name__ == "__main__":
    scores = []
    for task in TASKS:
        score = run_task(task)
        scores.append(score)
    print(f"\nFinal scores: {scores}", flush=True)
    print(f"Average: {sum(scores) / len(scores):.4f}", flush=True)
