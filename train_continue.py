"""Continues a single training run of a model from a checkpoint, but possibly with certain different hyperparms
    than its training began with.  Note that hyperparams that dictate the structure of the NNs
    must not be changed, as they are defined in the checkpoint file.
"""

from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import build_and_train_model
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic
from random_sampler import RandomSampler


#----------------------------------------------------------------------

RUN_NAME            = "CONT01-01" #full name of the run being continued - this identifies the checkpoint file
START_EPISODE       = 350   #the checkpoint from which training will continue

CHKPT_EVERY         = 100
PRIME               = 0  #num random experiences added to the replay buffer before training begins
SEED                = 468   #0, 111, 468, 5555, 23100, 44939
GOAL                = 0.85   #avg reward needed to be considered a satisfactory solution
MAX_EPISODES        = 1001  #max num episodes per run
INIT_TIME_STEPS     = 200
INCR_TSTEP_EVERY    = 8
FINAL_TIME_STEPS    = 400
USE_NOISE           = True
USE_COACHING        = False

BATCH           = 64
BAD_STEP_PROB   = 0.1
NOISE_INIT      = 0.2
NOISE_DECAY     = 0.999979
ACTOR_LR        = 0.000033
CRITIC_LR       = 0.000007
ACTOR_NN_L1     = 1024      #these values should be overridden by data in the checkpoint 
ACTOR_NN_L2     = 256
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
build_and_train_model(env, RUN_NAME, USE_COACHING, BATCH, PRIME, SEED, GOAL, START_EPISODE, MAX_EPISODES, CHKPT_EVERY,
                        INIT_TIME_STEPS, INCR_TSTEP_EVERY, FINAL_TIME_STEPS, BAD_STEP_PROB, 
                        USE_NOISE, NOISE_INIT, NOISE_DECAY,
                        ACTOR_LR, CRITIC_LR, ACTOR_NN_L1, ACTOR_NN_L2, CRITIC_NN_L1, CRITIC_NN_L2)

# Close the Unity environment after all training runs are complete
env.close()
