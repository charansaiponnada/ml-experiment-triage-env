# ML Experiment Triage Environment

## Quick Reference Card

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `API_KEY` | - | API authentication key |
| `HF_TOKEN` | - | HuggingFace token (preferred) |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment URL |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment for new episode |
| `POST` | `/step` | Execute an action |
| `GET` | `/health` | Health check |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action/observation schemas |

### Actions

```python
{"action_type": "investigate", "exp_id": "exp_001"}
{"action_type": "discard", "exp_id": "exp_001"}
{"action_type": "suggest", "suggestion": {"learning_rate": 0.001, "epochs": 50}}
{"action_type": "summarize", "summary": "exp_004"}
{"action_type": "compare", "comparison": {"exp_a": "exp_001", "exp_b": "exp_002", "analysis": "..."}}
{"action_type": "diagnose", "diagnosis": {"exp_id": "exp_005", "reason": "...", "fix": "..."}}
```

### Task IDs

| ID | Name | Difficulty |
|----|------|------------|
| 1 | `find_best_experiment` | Easy |
| 2 | `identify_overfitting` | Medium |
| 3 | `suggest_next_experiment` | Hard |
| 4 | `compare_experiments` | Medium |
| 5 | `debug_failed_run` | Hard |

### Scoring

- All scores must be strictly in range (0, 1)
- Use `_clamp_strict()` function to ensure bounds
- Final score = average of all task scores

### Commands

```bash
# Run locally
python inference.py

# Build Docker
docker build -t ml-triage .

# Run Docker
docker run -p 7860:7860 ml-triage

# Deploy to HF Space
git push hf main
```

### Common Issues

| Error | Solution |
|-------|----------|
| `KeyError: API_BASE_URL` | Set `HF_TOKEN` or `API_KEY` environment variable |
| `404 Not Found` | Check that server is running on correct port |
| `Score out of range` | Use `_clamp_strict()` instead of hardcoded values |
| `JSON parse error` | Ensure LLM response is valid JSON |

### File Locations

- Environment: `server/ml_triage_environment.py`
- Graders: `app/tasks.py` and `server/ml_triage_environment.py`
- Inference: `inference.py`
- Web UI: `app/main.py`
- Config: `openenv.yaml`
