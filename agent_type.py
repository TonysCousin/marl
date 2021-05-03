"""Represents a single type of agent in a multi-agent scenario.

    Note that this class assumes all agents of a given type have the same personality. That is, they use the same neural network
    to determine their next actions based on current state.
"""

import torch
import unityagents

class AgentType:

    """Creates an agent type definition.

        NOTE: This class is set up to work with UnityEnvironment that has discrete-action agents, where there is only one
        action variable, and it has integer values in [0, ma) to represet enumerated movements. The UnityEnvironment uses
        the variable name vector_action_space_size, which deceivingly sounds like a vector length, but it, in fact, is the
        max value of the single action variable.  Actions will need to be approached differently for complex continuous action
        models.
    """

    def __init__(self,
                 name           : str,                                  # the name of the agent type (equivalent to ML-Agents brain name)
                 brain          : unityagents.brain.BrainParameters,    # the ML-Agents brain object that represents this type of agent
                 state_size     : int,                                  # number of elements in the state vector
                 max_action_val : int,                                  # max value of the action (allowable range is [0, max_action_val))
                 num_agents     : int,                                  # number of agents of this type in the scenario
                 actor_nn       : torch.nn.Module = None,               # neural network that maps states -> actions for all agents of this type
                 critic_nn      : torch.nn.Module = None                # neural network used to predict Q value for the actor NN
                ):

        self.name = name
        self.type_obj = brain
        self.state_size = state_size
        self.max_action_val = max_action_val
        self.num_agents = num_agents
        self.actor_nn = actor_nn
        self.critic_nn = critic_nn

    """Stores the neural network to be used for this type of agent."""
    
    def set_nns(self,
                actor_nn             : torch.nn.Module,                      # the neural network that performs the actor predictions
                critic_nn            : torch.nn.Module                       # the neural network that performs the critic predictions
               ):

        self.actor_nn = actor_nn
        self.critic_nn = critic_nn