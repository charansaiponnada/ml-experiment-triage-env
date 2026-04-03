from openenv_core import create_app
from models import MLTriageAction, MLTriageObservation
from server.ml_triage_environment import MLTriageEnvironment


def create_env():
    return MLTriageEnvironment()


app = create_app(
    create_env, MLTriageAction, MLTriageObservation, env_name="ml-experiment-triage"
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
