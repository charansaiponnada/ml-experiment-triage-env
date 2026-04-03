from openenv_core import create_app
from envs.ml_triage_env.models import MLTriageAction, MLTriageObservation
from envs.ml_triage_env.server.ml_triage_environment import MLTriageEnvironment


def create_env():
    return MLTriageEnvironment()


app = create_app(
    create_env, MLTriageAction, MLTriageObservation, env_name="ml-experiment-triage"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
