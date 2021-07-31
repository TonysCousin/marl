"""Represents a single type of agent in a multi-agent scenario.

    Note that this class assumes all agents of a given type have the same personality. That is, they use the same neural network
    to determine their next actions based on current state.
"""

from numpy.core.numeric import ones
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import unityagents

class AgentType:

    """Creates an agent type that will be used to model all agents of the type.

        NOTE: This class is set up to work with UnityEnvironment that has discrete-action agents, where there is only one
        action variable, and it has integer values in [0, ma) to represet enumerated movements. The UnityEnvironment uses
        the variable name vector_action_space_size, which deceivingly sounds like a vector length, but it, in fact, is the
        max value of the single action variable.  Actions will need to be approached differently for continuous action
        models.
    """

    def __init__(self,
                 device                 : torch.device,                         # the compute device that is to be used for matrix math
                 name                   : str,                                  # the name of the agent type (equivalent to ML-Agents brain name)
                 brain                  : unityagents.brain.BrainParameters,    # the ML-Agents brain object that represents this type of agent
                 state_size             : int,                                  # number of elements in the state vector
                 action_size            : int,                                  # number of possible actions
                 num_agents             : int,                                  # number of agents of this type in the scenario
                 train_me               : List[bool],                           # a bool entry for each agent; is it to be trained?
                 use_policy             : List[bool],                           # a bool entry for each agent; is it to use policy NN for actions?
                                                                                #   False indicates it will always provide random actions
                 actor_nn               : nn.Module,                            # neural network that maps states -> actions for all agents of this type
                 critic_nn              : nn.Module,                            # neural network used to predict Q value for the actor NN
                 actor_lr               : float,                                # learning rate for the actor NN
                 critic_lr              : float,                                # learning rate for the critic NN
                 actor_weight_decay     : float,                                # optimizer weight decay for the actor NN
                 critic_weight_decay    : float                                 # optimizer weight decay for the critic NN
                ):

        # store the elementary info
        self.name = name
        self.type_obj = brain
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.train_me = train_me
        self.use_policy = use_policy

        # create the NNs and move them to the compute device - this is set up to support the MADDPG learning algorithm
        self.actor_policy = actor_nn.to(device)
        self.actor_target = actor_nn.to(device)
        self.actor_opt = optim.Adam(self.actor_policy.parameters(), lr=actor_lr, weight_decay=actor_weight_decay)

        self.critic_policy = critic_nn.to(device)
        self.critic_target = critic_nn.to(device)
        self.critic_opt = optim.Adam(self.critic_policy.parameters(), lr=critic_lr, weight_decay=critic_weight_decay)

    #------------------------------------------------------------------------------

    def set_training(self,
                     agent_id   : int,      # the sequential ID of the agent to be modified
                     on         : bool      # is training to be turned on for this agent?
                    ):
        
        self.train_me[agent_id] = on
    
    #------------------------------------------------------------------------------

    def set_policy_use(self,
                       agent_id     : int,  # the sequential ID of the agent to be modified
                       use          : bool  # is action policy to be used for this agent? False = random actions
                      ):
        
        self.use_policy[agent_id] = use