---
title: ML Experiment Triage
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "0.0.0"
python_version: "3.11"
app_file: Dockerfile
pinned: false
---

# ML Experiment Triage Environment

An OpenEnv-compliant RL environment where an AI agent triages ML experiment results — identifying best runs, overfitting models, and suggesting next hyperparameter configs.

## Motivation

Machine learning practitioners often run dozens or hundreds of experiments with different hyperparameters. Manually reviewing each experiment is time-consuming. This environment simulates the task of triaging experiment results, where an AI agent must:

1. **Find the best experiment** - Identify which configuration performed best
2. **Identify overfitting** - Detect models that are overfitting to training data
3. **Suggest next experiments** - Recommend hyperparameter configurations to try next

## Action Space

| Action | Parameters | Description |
|--------|------------|-------------|
| `investigate` | `exp_id: str` | Explore details of a specific experiment |
| `discard` | `exp_id: str` | Mark an experiment as not worth keeping |
| `suggest` | `suggestion: dict` | Suggest next hyperparameter configuration |
| `summarize` | `summary: str` | Provide final answer and end episode |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `experiments` | `List[ExperimentRecord]` | List of all experiment records |
| `current_step` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps allowed |
| `task_description` | `str` | Description of current task |
| `feedback` | `str` | Natural language feedback from last action |

### ExperimentRecord Fields

- `exp_id`: Unique experiment identifier
- `model_name`: Model architecture (e.g., "resnet18", "resnet34", "resnet50")
- `learning_rate`: Learning rate used
- `epochs`: Number of training epochs
- `train_acc`: Training accuracy
- `val_acc`: Validation accuracy
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `notes`: Experiment notes
- `status`: One of "pending", "investigated", "discarded"

## Tasks

### Task 1: Find the Best Experiment
- **Difficulty**: Easy
- **Max Steps**: 10
- **Description**: Given 8 ML experiments, use investigate() to explore runs and summarize() with the best exp_id.
- **Expected Score**: 1.0 if agent identifies exp_004 as best

### Task 2: Identify Overfitting Runs
- **Difficulty**: Medium
- **Max Steps**: 15
- **Description**: Find and discard all overfitting experiments. An experiment is overfitting if train_acc > 0.97 and val_acc < 0.75.
- **Expected Score**: 1.0 for correctly discarding all 3 overfitting experiments (exp_002, exp_006, exp_009)

### Task 3: Suggest the Next Experiment
- **Difficulty**: Hard
- **Max Steps**: 20
- **Description**: Analyze incomplete experiment results and suggest the next hyperparameter configuration to try.
- **Expected Score**: 1.0 for exact match with ground truth (lr=0.001, epochs=50, model="resnet50")

## Setup

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn app.main:app --reload --port 7860
```

### Using Docker

```bash
# Build the image
docker build -t ml-experiment-triage .

# Run the container
docker run -p 7860:7860 ml-experiment-triage
```

### With Hugging Face Spaces

This environment can be deployed to Hugging Face Spaces using the included configuration.

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Reset environment with task_id
- `POST /step` - Take an action
- `GET /state` - Get current state
- `GET /tasks` - List all available tasks

## Example Action JSON

### Investigate
```json
{
  "action_type": "investigate",
  "exp_id": "exp_004"
}
```

### Discard
```json
{
  "action_type": "discard",
  "exp_id": "exp_002"
}
```

### Suggest
```json
{
  "action_type": "suggest",
  "suggestion": {
    "learning_rate": 0.001,
    "epochs": 50,
    "model": "resnet50"
  }
}
```

### Summarize
```json
{
  "action_type": "summarize",
  "summary": "exp_004 is the best experiment with val_acc=0.94"
}
```

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| Find Best Experiment | gpt-4o-mini | TBD |
| Identify Overfitting | gpt-4o-mini | TBD |
| Suggest Next Experiment | gpt-4o-mini | TBD |

## Running Inference

```bash
# Set environment variables
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here

# Run inference
python inference.py
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests (when available)
uv run pytest

# Run linter
uv run ruff check .
```
