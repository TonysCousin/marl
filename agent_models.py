"""A container for the neural networks (NNs) that embody the behavioral rules for all of the agents 
    in this environment.  This class provides a reusable interface for specific
    projects to get their agent models in place.
"""

import torch


class AgentModels:

    def __init__(self):
        self.models = {}    # dict of tuples, where each tuple contains an actor network, a critic network, and other info for one agent type
    

#------------------------------------------------------------------------------

    """Adds a pair of NNs and optimizer info that can be used to fully define a given type of agent."""

    def add_actor_critic(self, 
                         type_name              : str,          # name of the agent type - this MUST match names used in the environment model
                         actor_model            : nn.Module,    # the NN used for the actor in each agent of this type
                         critic_model           : nn.Module,    # the NN used for the critic in each agent of this type
                         actor_lr               : float = 0.001,# the learning rate its actor's optimizer will use
                         critic_lr              : float = 0.001,# the learning rate its critic's optimizer will use
                         actor_weight_decay     : float = 0.0,  # weight decay rate its actor's optimizer will use
                         critic_weight_decay    : float = 0.0   # weight decay rate its critic's optimizer will use
                        ):
        
        self.models[type_name] = (actor_model, critic_model, actor_lr, critic_lr, actor_weight_decay, critic_weight_decay)
    
    
#------------------------------------------------------------------------------

    """Returns the number of model definitions on hand."""

    def get_num_models(self):
        return len(self.models)
    

#------------------------------------------------------------------------------

    """Returns the actor NN for the specified agent type."""

    def get_actor_nn_for(self,
                         type_name  : str   # name of the agent type
                        ):

        if type_name in self.models:
            return self.models[type_name][0]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

#------------------------------------------------------------------------------

    """Returns critic NN for the specified agent type."""

    def get_critic_nn_for(self,
                         type_name  : str   # name of the agent type
                        ):

        if type_name in self.models:
            return self.models[type_name][1]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

#------------------------------------------------------------------------------

    """Returns actor learning rate for the specified agent type."""

    def get_actor_lr_for(self,
                         type_name  : str   # name of the agent type
                        ):

        if type_name in self.models:
            return self.models[type_name][2]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

#------------------------------------------------------------------------------

    """Returns critic learning rate for the specified agent type."""

    def get_critic_lr_for(self,
                         type_name  : str   # name of the agent type
                        ):

        if type_name in self.models:
            return self.models[type_name][3]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

#------------------------------------------------------------------------------

    """Returns actor weight decay for the specified agent type."""

    def get_actor_weight_decay_for(self,
                                    type_name  : str   # name of the agent type
                                   ):

        if type_name in self.models:
            return self.models[type_name][4]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

#------------------------------------------------------------------------------

    """Returns critic weight decay for the specified agent type."""

    def get_critic_weight_decay_for(self,
                                    type_name  : str   # name of the agent type
                                   ):

        if type_name in self.models:
            return self.models[type_name][5]
        else:
            print("Agent type '{}' unknown.".format(type_name))
            return None

