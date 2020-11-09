REGISTRY = {}

from .rnn_agent import RNNAgent, MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["mlp"] = MLPAgent
