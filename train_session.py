"""Executes a training session that explores the hyperparameter space over several
    runs with the same model.
"""

import math

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import build_and_train_model
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic
from random_sampler import RandomSampler


#----------------------------------------------------------------------

NAME                = "TRAIN11" #next is 
NUM_RUNS            = 4
CHKPT_EVERY         = 100
PRIME               = 2000  #num random experiences added to the replay buffer before training begins
SEED                = 468   #0, 111, 468, 5555, 23100, 44939
GOAL                = 0.8   #avg reward needed to be considered a satisfactory solution
EPISODES            = 2001  #max num episodes per run
INIT_TIME_STEPS     = 120
INCR_TSTEP_EVERY    = 8
FINAL_TIME_STEPS    = 400
USE_NOISE           = True

# Define the ranges of hyperparams that will be explored
vars = [
        ["discrete",            32, 64, 128],               # BATCH
        ["continuous-float",    0.01,        0.2],          # BAD_STEP_PROB
        ["continuous-float",    0.8,         0.95],          # NOISE_INIT
        ["continuous-float",    -5.1,       -4.0],          # log10 of 1-NOISE_DECAY
        ["continuous-float",    -5.0,       -2.0],          # log10 of actor LR (all agent types)
        ["continuous-float",    -5.0,       -2.0],          # log10 of critic LR (all agent types)
        ["discrete",            748, 1024],            # ACTOR_NN_L1 num nodes
        ["discrete",            4, 8],                      # ACTOR_NN_L2 divisor (from l1)
        ["discrete",            1536, 2048, 3072],    # CRITIC_NN_L1 num nodes
        ["discrete",            4, 8]                       # CRITIC_NN_L2 divisor (from l1)
       ]
rs = RandomSampler(vars)

# Need to create the Unity env one time; destroying it and creating a new one inside the loop doesn't work
env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

# Loop through the desired number of training runs and randomly select a set of hyperparams for each run
for run in range(NUM_RUNS):

    run_name = "{}-{:02d}".format(NAME, run)

    v = rs.sample()
    BATCH           = v[0]
    BAD_STEP_PROB   = v[1]
    NOISE_INIT      = v[2]
    NOISE_DECAY     = min((1.0 - math.pow(10.0, v[3])), 0.999995)
    ACTOR_LR        = math.pow(10.0, v[4])
    CRITIC_LR       = min(math.pow(10.0, v[5]), 1.1*ACTOR_LR)
    ACTOR_NN_L1     = v[6]
    ACTOR_NN_L2     = ACTOR_NN_L1 // v[7]
    CRITIC_NN_L1    = v[8]
    CRITIC_NN_L2    = CRITIC_NN_L1 // v[9]

    print("\n///// Beginning run {} with:".format(run_name))
    print("      Batch size     = {:4d}".format(BATCH))
    print("      Bad step prob  = {:.2f}".format(BAD_STEP_PROB))
    print("      Initial noise  = {:.2f}".format(NOISE_INIT))
    print("      Noise decay    = {:.6f}".format(NOISE_DECAY))
    print("      Actor LR       = {:.6f}".format(ACTOR_LR))
    print("      Critic LR      = {:.6f}".format(CRITIC_LR))
    print("      Actor l1 size  = {:d}".format(ACTOR_NN_L1))
    print("      Actor l2 size  = {:d}".format(ACTOR_NN_L2))
    print("      Critic l1 size = {:d}".format(CRITIC_NN_L1))
    print("      Critic l2 size = {:d}".format(CRITIC_NN_L2))

    # Build the model with the selected hyperparams and train it
    build_and_train_model(env, run_name, BATCH, PRIME, SEED, GOAL, 0, EPISODES, CHKPT_EVERY, INIT_TIME_STEPS, 
                            INCR_TSTEP_EVERY, FINAL_TIME_STEPS, BAD_STEP_PROB, USE_NOISE, NOISE_INIT, NOISE_DECAY,
                            ACTOR_LR, CRITIC_LR, ACTOR_NN_L1, ACTOR_NN_L2, CRITIC_NN_L1, CRITIC_NN_L2)

# Close the Unity environment after all training runs are complete
env.close()
