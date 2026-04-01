import os
import json
import logging
import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an ML experiment analysis agent. You will be shown a table 
of ML experiments. Your job is to analyze them and take actions.

Available actions (respond with ONLY valid JSON, no markdown):
{"action_type": "investigate", "exp_id": "exp_001"}
{"action_type": "discard", "exp_id": "exp_001"}  
{"action_type": "suggest", "suggestion": {"model_name": "resnet50", "learning_rate": 0.001, "epochs": 50}}
{"action_type": "summarize", "summary": "exp_004"}

Rules:
- Always investigate at least 2 experiments before summarizing
- Discard experiments where train_acc > 0.97 AND val_acc < 0.75
- For suggest: recommend config based on best performing experiments
- For summarize: provide the exp_id of the best experiment
- Respond ONLY with JSON. No explanation. No markdown."""


def call_api(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def format_observation(obs: dict) -> str:
    lines = [f"Task: {obs.get('task_description', '')}"]
    lines.append(f"Step: {obs.get('current_step', 0)}/{obs.get('max_steps', 0)}")
    lines.append(f"Feedback: {obs.get('feedback', '')}")
    lines.append("\nExperiments:")
    for exp in obs.get("experiments", []):
        val_acc = exp.get("val_acc")
        if val_acc is None:
            val_acc = "N/A"
        lines.append(
            f"  {exp['exp_id']}: {exp['model_name']} lr={exp['learning_rate']} epochs={exp['epochs']} "
            f"train_acc={exp['train_acc']} val_acc={val_acc} status={exp['status']}"
        )
    return "\n".join(lines)


def run_task(task_id: int, task_name: str) -> dict:
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    obs = reset_resp.json()

    print(
        f"[START] task={task_name} env=ml-experiment-triage model={MODEL_NAME}",
        flush=True,
    )

    total_reward = 0.0
    step_num = 0

    while True:
        step_num += 1

        prompt = format_observation(obs)
        prompt += "\n\nWhat action would you like to take? Return JSON with action_type and exp_id (if applicable)."

        try:
            action = call_api(prompt)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            action = {"action_type": "summarize", "summary": "exp_001"}

        step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action})
        result = step_resp.json()

        obs = result["observation"]
        reward = result["reward"]["value"]
        done = result["done"]

        total_reward += reward
        action_type = action.get("action_type", "unknown")

        print(
            f"[STEP] step={step_num} action={action_type} reward={reward:.4f} done={done}",
            flush=True,
        )

        if done:
            break

        if step_num >= 20:
            break

    score = max(0.0, min(1.0, total_reward))
    success = score >= 0.5

    print(f"[END] success={success} steps={step_num} score={score:.4f}", flush=True)

    return {"success": success, "steps": step_num, "score": score}


def main():
    tasks = [
        (1, "find_best_experiment"),
        (2, "identify_overfitting"),
        (3, "suggest_next_experiment"),
    ]

    results = []
    for task_id, task_name in tasks:
        result = run_task(task_id, task_name)
        results.append((task_name, result))

    print("\n=== Summary ===")
    for task_name, result in results:
        print(
            f"{task_name}: score={result['score']:.2f}, steps={result['steps']}, success={result['success']}"
        )


if __name__ == "__main__":
    main()
