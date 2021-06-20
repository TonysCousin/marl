# Unit tests for the train() function

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import train
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic


#----------------------------------------------------------------------

NAME        = "TRAIN2" #
BATCH       = 128
PRIME       = 2000  #num random experiences added to the replay buffer before training begins
SEED        = 144
GOAL        = 0.9   #avg reward needed to be considered a satisfactory solution
EPISODES    = 4000  #max num episodes per run
USE_NOISE   = True
NOISE_INIT  = 0.5   #initial probability of noise for any given action
NOISE_DECAY = 0.99999 # amount noise probability is reduced after each time step

#----------------------------------------------------------------------

def train_model():
    env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

    # define NNs
    agent_models = AgentModels()
    agent_models.add_actor_critic("GoalieBrain", GoalieActor(336, 1, fc1_units=512), GoalieCritic(1344, 4, fcs1_units=2048, fc2_units=256))
    agent_models.add_actor_critic("StrikerBrain", StrikerActor(336, 1, fc1_units=512), StrikerCritic(1344, 4, fcs1_units=2048, fc2_units=256))

    mgr = AgentMgr(env, agent_models, batch_size=BATCH, buffer_prime=PRIME, random_seed=SEED,
                    use_noise=USE_NOISE, noise_init=NOISE_INIT, noise_decay=NOISE_DECAY)
    #print("Haltus")

    train(mgr, env, run_name=NAME, max_episodes=EPISODES, chkpt_interval=100, training_goal=GOAL)

#----------------------------------------------------------------------

train_model()
