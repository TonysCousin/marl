# Unit tests for the train() function

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import train
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic


#----------------------------------------------------------------------

NAME        = "TRAIN1"
BATCH       = 128
PRIME       = 2000  #num random experiences added to the replay buffer before training begins
SEED        = 0
GOAL        = 0.9   #avg reward needed to be considered a satisfactory solution

#----------------------------------------------------------------------

def train_model():
    env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

    # define NNs
    agent_models = AgentModels()
    agent_models.add_actor_critic("GoalieBrain", GoalieActor(336, 1, fc1_units=512), GoalieCritic(1344, 4, fcs1_units=2048, fc2_units=256))
    agent_models.add_actor_critic("StrikerBrain", StrikerActor(336, 1, fc1_units=512), StrikerCritic(1344, 4, fcs1_units=2048, fc2_units=256))

    mgr = AgentMgr(env, agent_models, batch_size=BATCH, buffer_prime=PRIME, random_seed=SEED)
    #print("Haltus")

    train(mgr, env, run_name=NAME, max_episodes=1000, chkpt_interval=100, training_goal=GOAL)

#----------------------------------------------------------------------

train_model()

