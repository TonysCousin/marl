"""Represents a single type of agent in a multi-agent scenario.

    Note that this class assumes all agents of a given type have the same personality. That is, they use the same neural network
    to determine their next actions based on current state.
"""

class AgentType:

    """Creates an agent type definition."""

    def __init__(self,
                 name           : string,                               # the name of the agent type (equivalent to ML-Agents brain name)
                 brain          : unityagents.brain.BrainParameters,    # the ML-Agents brain object that represents this type of agent
                 state_size     : int,                                  # number of elements in the state vector
                 action_size    : int,                                  # number of elements in the action vector
                 num_agents     : int,                                  # number of agents of this type in the scenario
                 actor_nn       : torch.Module = None,                  # neural network that maps states -> actions for all agents of this type
                 critic_nn      : torch.Module = None                   # neural network used to predict Q value for the actor NN
                ):

        self.name = name
        self.type_obj = brain
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.actor_nn = actor_nn
        self.critic_nn = critic_nn

    """Stores the neural network to be used for this type of agent."""
    
    def set_nns(self,
                actor_nn             : torch.Module,                         # the neural network that performs the actor predictions
                critic_nn            : torch.Module                          # the neural network that performs the critic predictions
               ):

        self.actor_nn = actor_nn
        self.critic_nn = critic_nn