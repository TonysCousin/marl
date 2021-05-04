"""AgentMgr class manages all of the multiple agents of multiple types in the scenario.

    Note:  there is to be only one object of this type in the system.

    Note:  we want to use only one pseudo-random number generator (PRNG) throughout this
    system, rather than have each class instantiate & seed its own. When several instances
    are defined with the same seed we would have many identical sequences, which may risk
    some correlations and therefore instabilities.
"""

import numpy as np

from unityagents    import UnityEnvironment
from agent_type     import AgentType

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentMgr:

    """Initialize the AgentMgr using a pre-established environment model."""

    def __init__(self,
                 env            : UnityEnvironment, # the envronment model that agents live in
                 random_seed    : int = 0,          # seed for the PRNG
                 batch_size     : int = 32,         # number of experiences in a learning batch
                 buffer_size    : int = 100000,     # capacity of the experience replay buffer
                 use_noise      : bool = False      # should we inject random noise into actions?
                ):

        self.prng = default_rng(random_seed) #the one and only PRNG in the entire system (must be passed around)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self-use_noise = use_noise
        
        # store the info for each type of ageint in use
        self.agent_types = {}
        env_info = env.reset(train_mode=True)
        for name in env.brain_names:
            brain = env.brains[name]
            type_info = env_info[name]
            ns = len(type_info.vector_observations[0])
            max_a = brain.vector_action_space_size
            num_agents = len(type_info.agents)
            self.agent_types[name] = AgentType(name, brain, ns, max_a, num_agents)
            # Note that this loop leaves the NNs for each type undefined. They need
            # to be defined in this constructor, however.
        
        # initialize other internal stuff
        self.learning_underway = True #set to False here if buffer priming is introduced

        # define simple experience replay buffer common to all agents
        self.erb = ReplayBuffer(action_size, buffer_size, batch_size, buffer_prime_size, self.prng)

        
        
        #TODO: need to define all the NNs here and add them to the agent_types dict.
        #       Can I generalize this, since the model file will be application-specific,
        #       and do some dependency injection of the NNs here?



        # Instantiate the neural networks that give each agent its behavioral personality.
        # This is where the code cannot be generalized - each agent type needs its own NN
        # structure, depending on its desired functioins, in puts and outputs.  
        # They are structured for MADDPG learning, with a pair of critics (policy & target)
        # and a pair of actors (policy & target) for each agent type.

    #------------------------------------------------------------------------------

    """Returns a dict of AgentType that provides info on all the types in use."""

    def get_agent_types(self):
        return self.agent_types

    #------------------------------------------------------------------------------

    """Returns true if learning is underway, false if replay buffer is being primed."""
    
    def is_learning_underway(self):
        return self.learning_underway
    
    #------------------------------------------------------------------------------


    def reset(self):
        pass #TODO: dummy

    #------------------------------------------------------------------------------


    def save_checkpoint(self, path, name, episode):
        pass #TODO: dummy

    #------------------------------------------------------------------------------


    def get_memory_stats(self):
        return (42, 4) #TODO: dummy

    #------------------------------------------------------------------------------


    def act(self, states):
        #TODO: dummy logic
        a = {}
        for t in self.agent_types:
            actions = np.array((1, self.agent_types[t].num_agents), dtype=int) #one element for each agent of this type
            for i in range(self.agent_types[t].num_agents):
                actions[i] = i
            a[t] = actions
        return a

    #------------------------------------------------------------------------------


    def step(self, states, actions, rewards, next_states, dones):
        pass #TODO: dummy