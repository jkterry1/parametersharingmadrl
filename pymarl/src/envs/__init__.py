from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .sisl_env import SislEnv
from .pettingzoo_env import PettingZooEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sisl"] = partial(env_fn, env=SislEnv)
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZooEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
