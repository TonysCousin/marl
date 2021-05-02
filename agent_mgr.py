"""Manages all of the multiple agents of multiple types in the scenario.

    Note that there is to be only one object of this type in the system.
"""

from agent_type     import AgentType

class AgentMgr:

    """Initialize the AgentMgr using a pre-established environment model."""

    def __init__(self,
                 env    : UnityEnvironment  # the envronment model that agents live in
                ):

        # store the info for each type of ageint in use
        self.agent_types = {}
        env_info = env.reset(train_mode=True)
        for name in env.brain_names:
            brain = env.brains[name]
            type_info = env_info[name]
            ns = len(type_info.vector_observations[0])
            na = brain.vector_action_space_size
            num_agents = len(type_info.agents)
            self.agent_types[name] = AgentType(name, brain, ns, na, num_agents)
            # Note that this loop leaves the NNs for each type undefined. They need
            # to be defined in this constructor, however.
        
        #TODO: fill in remainder of this method from the maddpg code base

    #------------------------------------------------------------------------------

    """Returns a dict of AgentType that provides info on all the types in use."""

    def get_agent_types(self):
        return self.agent_types