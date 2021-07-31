"""Executes a training session for the Unity soccer game that explores
    the hyperparameter space over several runs with the same model.
"""

# Goalie actions:
#   0 - move forward
#   1 - move backward
#   2 - move right
#   3 - move left
#   4 - do nothing
#
# Striker actions:
#   0 - move forward
#   1 - move backward
#   2 - turn right
#   3 - turn left
#   4 - move left
#   5 - move right
#   6 - do nothing
#
# State vectors include 3 time steps of 112 values each.  The first 112 are two steps old, the next 112 are one step old, and
# the current time step is represented in the final 112 values.  Within each of these 112 values, the structure is in groups
# of 8 values, each representing one of 14 visual ray traces emanating out from the agent.  Empirical evidence indicates the
# following approximate interpretations, which contradicts the explanation from the Unity site:
#   Rays 2, 5, 6, 9, 12, 13 are more or less forward (it seems some may sense different distances rather than the full range);
#       5 and 12 are slightly to the left, while 6 and 13 are slightly to the right
#   Rays 3, 10 are forward-left
#   Rays 1, 8 are forward-right
#   Rays 4, 11 are more or less left side
#   Rays 0, 7 are more or less right side
#   There is a huge blind spot for approx 180 degrees around the rear of the agent.  There is also a small blind spot (at least
#   at some distance) between each set of side rays and the next forward set of rays.
#
# For each of the rays, its 8 values consist of a 7-long one-hot vector to indicate the type of object it sees, and the last
# value indicates the distance to that object, in [0, 1).  For the one-ho vector, the values are exactly 1.0 or 0.0.  If a
# ray sees nothing then all 8 of its values are 0.0.  The one-hot values (elements 0-6) represent the following:
#   0 - ball
#   1 - unused
#   2 - unused
#   3 - side wall (does not distinguish which side)
#   4 - red striker
#   5 - blue striker
#   6 - unused
# It is a shame that there are so many dead values in this state vector. In particular, it would be nice to identify the goals
# and the goalies.  A striker never has any knowledge where it is relative to the goal it is to shoot at, which seems to be
# a huge impediment to learning desirable behavior.
#

import math

from unityagents    import UnityEnvironment
from train          import build_and_train_model
from random_sampler import RandomSampler


#----------------------------------------------------------------------

NAME                = "DISC01" #next is DISC01
NUM_RUNS            = 8
CHKPT_EVERY         = 100
PRIME               = 0     #num random experiences added to the replay buffer before training begins
SEED                = 5555  #0, 111, 468, 5555, 23100, 44939
GOAL                = 0.8   #avg reward needed to be considered a satisfactory solution
EPISODES            = 1001  #max num episodes per run
INIT_TIME_STEPS     = 500   #prefer 500 - 600
INCR_TSTEP_EVERY    = 8
FINAL_TIME_STEPS    = 600
USE_NOISE           = True
USE_COACHING        = False
TRAIN_AGENTS        = {"GoalieBrain": [True, False], "StrikerBrain": [True, False]} #agent 0 = red, agent 1 = blue
USE_POLICY          = {"GoalieBrain": [True, False], "StrikerBrain": [True, False]}

# Define the ranges of hyperparams that will be explored
vars = [
        ["discrete",            16, 64, 256],               # BATCH
        ["discrete",            1.0        ],          # BAD_STEP_PROB
        ["continuous-float",    0.8,         0.95],          # NOISE_INIT
        ["continuous-float",    -5.0,       -3.0],          # log10 of 1-NOISE_DECAY (was -5.1, -4.5)
        ["continuous-float",    -5.0,       -2.7],          # log10 of actor LR (all agent types)
        ["continuous-float",    0.05,       0.5],          # multiplier on actor LR to get critic LR
        ["discrete",            512, 1024],            # ACTOR_NN_L1 num nodes
        ["discrete",            4, 8],                      # ACTOR_NN_L2 divisor (from l1)
        ["discrete",            1536, 2048, 3072],    # CRITIC_NN_L1 num nodes
        ["discrete",            4, 8],                       # CRITIC_NN_L2 divisor (from l1)
        ["discrete",            1],                    # LEARN_EVERY
        ["continuous-float",   -4.0, -2.0]                 # log10 of TAU
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
    CRITIC_LR       = v[5]*ACTOR_LR
    ACTOR_NN_L1     = v[6]
    ACTOR_NN_L2     = ACTOR_NN_L1 // v[7]
    CRITIC_NN_L1    = v[8]
    CRITIC_NN_L2    = CRITIC_NN_L1 // v[9]
    LEARN_EVERY     = v[10]
    TAU             = math.pow(10.0, v[11])

    print("\n///// Beginning run {} with:".format(run_name))
    print("      Batch size     = {:4d}".format(BATCH))
    print("      Bad step prob  = {:.2f}".format(BAD_STEP_PROB))
    print("      Initial noise  = {:.2f}".format(NOISE_INIT))
    print("      Noise decay    = {:.6f}".format(NOISE_DECAY))
    print("      Actor LR       = {:.6f}".format(ACTOR_LR))
    print("      Critic LR      = {:.6f}".format(CRITIC_LR))
    print("      Tgt update rate= {:.5f}".format(TAU))
    print("      Actor l1 size  = {:d}".format(ACTOR_NN_L1))
    print("      Actor l2 size  = {:d}".format(ACTOR_NN_L2))
    print("      Critic l1 size = {:d}".format(CRITIC_NN_L1))
    print("      Critic l2 size = {:d}".format(CRITIC_NN_L2))
    print("      Learn every    = {:d}".format(LEARN_EVERY))
    print("      Agents trained = ", TRAIN_AGENTS)
    print("      Agent policies = ", USE_POLICY)

    # Build the model with the selected hyperparams and train it
    build_and_train_model(env, run_name, TRAIN_AGENTS, USE_POLICY, USE_COACHING, BATCH, PRIME, 
                            LEARN_EVERY, SEED, GOAL, 0, EPISODES, CHKPT_EVERY, INIT_TIME_STEPS, 
                            INCR_TSTEP_EVERY, FINAL_TIME_STEPS, BAD_STEP_PROB, USE_NOISE, NOISE_INIT, NOISE_DECAY,
                            ACTOR_LR, CRITIC_LR, ACTOR_NN_L1, ACTOR_NN_L2, CRITIC_NN_L1, CRITIC_NN_L2, TAU)

# Close the Unity environment after all training runs are complete
env.close()
