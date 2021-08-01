"""Represents a single type of agent in a multi-agent scenario.

    Note that this class assumes all agents of a given type have the same personality. That is, they use the same neural network
    to determine their next actions based on current state.
"""

from numpy.core.numeric import ones
import torch
import torch.nn as nn
import torch.optim as optim
import enum
from typing import List
import unityagents


"""Categories of behavior to be applied to each agent."""

class AgentBehavior(enum.Enum):
    Learn           = 0 # always learning from experiences and executing its latest policy
    Policy          = 1 # always following the latest policy for its type, but not learning
    Random          = 2 # always exhibiting random actions (ignores policy)
    Uniform         = 3 # always chooses the same action (must be specified separately)
    AnnealedRandom  = 4 # gradually changes from Uniform to Random (annealing rate specified separately)

#------------------------------------------------------------------------------

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

        # set the defaults for behavior for each agent of this type
        self.behavior = [AgentBehavior.Learn] * self.num_agents
        self.uniform_action_vector = [None] * self.num_agents
        self.anneal_rate = [1.0] * self.num_agents
        self.anneal_mult = [1.0] * self.num_agents

        # create the NNs and move them to the compute device - this is set up to support the MADDPG learning algorithm
        self.actor_policy = actor_nn.to(device)
        self.actor_target = actor_nn.to(device)
        self.actor_opt = optim.Adam(self.actor_policy.parameters(), lr=actor_lr, weight_decay=actor_weight_decay)

        self.critic_policy = critic_nn.to(device)
        self.critic_target = critic_nn.to(device)
        self.critic_opt = optim.Adam(self.critic_policy.parameters(), lr=critic_lr, weight_decay=critic_weight_decay)

    #------------------------------------------------------------------------------

    """Allows override of the default behavior specification for the given agent within this type. Optional params
        uniform_act and anneal_rate only apply to certain kinds of behavior. If AnnealedRandom behavior is chosen,
        then the anneal_rate specifies how quickly the behavior changes from uniform to random: a value of 1.0 will
        keep the behavior uniform forever, while 0.0 will move it to fully random in one time step. Values in
        between act as a rate of decay of the uniformity gradually into more and more randomness."""

    def set_behavior(self,
                     agent_id   : int,              # the sequential ID of the agent to be modified
                     behavior   : AgentBehavior,    # the behavior to be used by this agent
                     uniform_act: [] = None,        # vector of raw action values to be used in a Uniform behavior
                     anneal_rate: float = 1.0       # annealing rate to be used for AnnealedRandom behavior; value must be in [0, 1]
                    ):
        
        self.behavior[agent_id] = behavior
        self.uniform_action_vector[agent_id] = uniform_act
        self.anneal_rate[agent_id] = max(min(anneal_rate, 1.0), 0.0)
        self.anneal_mult[agent_id] = 1.0 #reset the annealing process to begin after this call
