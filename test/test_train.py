# Unit tests for the train() function

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import train
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic

def test_default():
    env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

    # define some bogus NNs for now, just to get the thing to execute
    agent_models = AgentModels()
    agent_models.add_actor_critic("GoalieBrain", GoalieActor(336, 1), GoalieCritic(1344, 4))
    agent_models.add_actor_critic("StrikerBrain", StrikerActor(336, 1), StrikerCritic(1344, 4))

    mgr = AgentMgr(env, agent_models, batch_size=3, buffer_prime=4)
    print("Haltus")

    train(mgr, env)

test_default()

