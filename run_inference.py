from train_continue import RUN_NAME
from unityagents    import UnityEnvironment
from agent_mgr      import AgentMgr
from agent_models   import AgentModels
from train          import CHECKPOINT_PATH, inference_time_step, build_model
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic

def run_inference():



    print("\n///// run_inference:  UNDER CONSTRUCTION!  NOT YET FUNCTIONAL.\n\n")
    return
    
    # Create the Unity env
    env = UnityEnvironment(file_name="/home/starkj/soccer/Soccer_Linux/Soccer.x86_64")

    # create the model and manager for the agents
    agent_models = build_model(actor_lr, critic_lr, actor_nn_l1, actor_nn_l2, critic_nn_l1, critic_nn_l2)
    mgr = AgentMgr(env, agent_models, buffer_prime=0)

    # load the pre-trained model
    mgr.restore_checkpoint(CHECKPOINT_PATH, RUN_NAME, EPISODE)

    # Close the Unity environment
    env.close()
