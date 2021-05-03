# Unit tests for the train() function

from unityagents import UnityEnvironment
from agent_mgr import AgentMgr
from train      import train

def test_default():
    env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")
    mgr = AgentMgr(env)
    print("Haltus")

    train(mgr, env)

test_default()

