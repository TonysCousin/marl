"""Continues a single training run of a model from a checkpoint, but possibly with certain different hyperparms
    than its training began with.  Note that hyperparams that dictate the structure of the NNs
    must not be changed, as they are defined in the checkpoint file.
"""

import math

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import build_and_train_model
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic
from random_sampler import RandomSampler


#----------------------------------------------------------------------

RUN_NAME            = "TRAIN9-00" #full name of the run being continued - this identifies the checkpoint file
START_EPISODE       = 100   #the checkpoint from which training will continue

CHKPT_EVERY         = 100
PRIME               = 2000  #num random experiences added to the replay buffer before training begins
SEED                = 468   #0, 111, 468, 5555, 23100, 44939
GOAL                = 0.9   #avg reward needed to be considered a satisfactory solution
MAX_EPISODES        = 2001  #max num episodes per run
INIT_TIME_STEPS     = 120
INCR_TSTEP_EVERY    = 8
FINAL_TIME_STEPS    = 400
USE_NOISE           = True

BATCH           = 64
BAD_STEP_PROB   = 0.04
NOISE_INIT      = 0.423
NOISE_DECAY     = 0.999949
ACTOR_LR        = 0.000037
CRITIC_LR       = 0.000041
ACTOR_NN_L1     = 748
ACTOR_NN_L2     = 187
CRITIC_NN_L1    = 3072
CRITIC_NN_L2    = 768

# Need to create the Unity env
env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

print("\n///// Continuing run {} from episode {} with:".format(RUN_NAME, START_EPISODE))
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
build_and_train_model(env, RUN_NAME, BATCH, PRIME, SEED, GOAL, START_EPISODE, MAX_EPISODES, CHKPT_EVERY,
                        INIT_TIME_STEPS, INCR_TSTEP_EVERY, FINAL_TIME_STEPS, BAD_STEP_PROB, 
                        USE_NOISE, NOISE_INIT, NOISE_DECAY,
                        ACTOR_LR, CRITIC_LR, ACTOR_NN_L1, ACTOR_NN_L2, CRITIC_NN_L1, CRITIC_NN_L2)

# Close the Unity environment after all training runs are complete
env.close()
