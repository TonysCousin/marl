"""Executes a training session that explores the hyperparameter space over several
    runs with the same model.
"""

import time
import math

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import train
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic
from random_sampler import RandomSampler


#----------------------------------------------------------------------

NAME        = "TRAIN3" #3
NUM_RUNS    = 10
CHKPT_EVERY = 500
PRIME       = 2000  #num random experiences added to the replay buffer before training begins
SEED        = 111   #0, 111, 468, 5555, 23100, 44939
GOAL        = 0.9   #avg reward needed to be considered a satisfactory solution
EPISODES    = 4000  #max num episodes per run
USE_NOISE   = True

#----------------------------------------------------------------------

def train_model(env         : UnityEnvironment,
                name        : str,
                batch       : int,
                prime       : int,
                seed        : int,
                goal        : float,
                episodes    : int,
                bad_step_prob: float,
                use_noise   : bool,
                noise_init  : float,
                noise_decay : float,
                actor_lr    : float,
                critic_lr   : float,
                actor_nn_l1 : int,
                actor_nn_l2 : int,
                critic_nn_l1: int,
                critic_nn_l2: int
               ):

    # define NNs
    agent_models = AgentModels()
    agent_models.add_actor_critic("GoalieBrain", GoalieActor(336, 1, fc1_units=actor_nn_l1, fc2_units=actor_nn_l2), 
                                    GoalieCritic(1344, 4, fcs1_units=critic_nn_l1, fc2_units=critic_nn_l2),
                                    actor_lr, critic_lr)
    agent_models.add_actor_critic("StrikerBrain", StrikerActor(336, 1, fc1_units=actor_nn_l1, fc2_units=actor_nn_l2),
                                    StrikerCritic(1344, 4, fcs1_units=critic_nn_l1, fc2_units=critic_nn_l2),
                                    actor_lr, critic_lr)

    mgr = AgentMgr(env, agent_models, batch_size=batch, buffer_prime=prime, bad_step_prob=bad_step_prob, random_seed=seed,
                    use_noise=use_noise, noise_init=noise_init, noise_decay=noise_decay)
    #print("Haltus")

    train(mgr, env, run_name=name, max_episodes=episodes, chkpt_interval=CHKPT_EVERY, training_goal=goal)

#----------------------------------------------------------------------
#   EXECUTE THE TRAINING SESSION
#----------------------------------------------------------------------

vars = [
        ["discrete",            32, 64, 128],               # BATCH
        ["continuous-float",    0.1,        0.8],           # BAD_STEP_PROB
        ["continuous-float",    0.2,        0.95],          # NOISE_INIT
        ["continuous-float",    -5.0,       -2.0],          # log10 of 1-NOISE_DECAY
        ["continuous-float",    -5.0,       -2.0],          # log10 of actor LR (all agent types)
        ["continuous-float",    -5.0,       -2.0],          # log10 of critic LR (all agent types)
        ["discrete",            512, 748, 1024],            # ACTOR_NN_L1 num nodes
        ["discrete",            4, 8],                      # ACTOR_NN_L2 divisor (from l1)
        ["discrete",            1536, 2048, 3072, 4096],    # CRITIC_NN_L1 num nodes
        ["discrete",            4, 8]                       # CRITIC_NN_L2 divisor (from l1)
       ]
rs = RandomSampler(vars)

# Need to create the Unity env one time; destroying it and creating a new one inside the loop doesn't work
env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

for run in range(NUM_RUNS):

    run_name = "{}-{:02d}".format(NAME, run)

    v = rs.sample()
    BATCH           = v[0]
    BAD_STEP_PROB   = v[1]
    NOISE_INIT      = v[2]
    NOISE_DECAY     = min((1.0 - math.pow(10.0, v[3])), 0.99999)
    ACTOR_LR        = math.pow(10.0, v[4])
    CRITIC_LR       = math.pow(10.0, v[5])
    ACTOR_NN_L1     = v[6]
    ACTOR_NN_L2     = ACTOR_NN_L1 // v[7]
    CRITIC_NN_L1    = v[8]
    CRITIC_NN_L2    = CRITIC_NN_L1 // v[9]

    print("\n///// Beginning run {} with:".format(run_name))
    print("      Batch size     = {:4d}".format(BATCH))
    print("      Bad step prob  = {:.2f}".format(BAD_STEP_PROB))
    print("      Initial noise  = {:.2f}".format(NOISE_INIT))
    print("      Noise decay    = {:.5f}".format(NOISE_DECAY))
    print("      Actor LR       = {:.6f}".format(ACTOR_LR))
    print("      Critic LR      = {:.6f}".format(CRITIC_LR))
    print("      Actor l1 size  = {:d}".format(ACTOR_NN_L1))
    print("      Actor l2 size  = {:d}".format(ACTOR_NN_L2))
    print("      Critic l1 size = {:d}".format(CRITIC_NN_L1))
    print("      Critic l2 size = {:d}".format(CRITIC_NN_L2))

    train_model(env, run_name, BATCH, PRIME, SEED, GOAL, EPISODES, BAD_STEP_PROB, USE_NOISE, NOISE_INIT, NOISE_DECAY,
                ACTOR_LR, CRITIC_LR, ACTOR_NN_L1, ACTOR_NN_L2, CRITIC_NN_L1, CRITIC_NN_L2)

env.close()
