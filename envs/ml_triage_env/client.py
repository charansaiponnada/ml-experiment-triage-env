from openenv_core import EnvClient
from .models import MLTriageAction, MLTriageObservation


class MLTriageEnv(EnvClient):
    action_class = MLTriageAction
    observation_class = MLTriageObservation

    @classmethod
    async def connect(cls, base_url: str):
        return cls(base_url=base_url)
