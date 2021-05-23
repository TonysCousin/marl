"""AgentMgr class manages all of the multiple agents of multiple types in the scenario.

    Note:  there is to be only one object of this type in the system.

    Note:  we want to use only one pseudo-random number generator (PRNG) throughout this
    system, rather than have each class instantiate & seed its own. When several instances
    are defined with the same seed we would have many identical sequences, which may risk
    some correlations and therefore instabilities.
"""

import numpy as np
import torch
import torch.nn as nn

from unityagents    import UnityEnvironment
from agent_type     import AgentType
from agent_models   import AgentModels
from replay_buffer  import ReplayBuffer
import utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentMgr:

    """Initialize the AgentMgr using a pre-established environment model."""

    def __init__(self,
                 env            : UnityEnvironment, # the envronment model that agents live in
                 agent_models   : AgentModels,      # holds all of the NN models for each type of agent
                 random_seed    : int = 0,          # seed for the PRNG
                 batch_size     : int = 32,         # number of experiences in a learning batch
                 buffer_size    : int = 100000,     # capacity of the experience replay buffer
                 use_noise      : bool = False,     # should we inject random noise into actions?
                 update_factor  : float = 0.001     # Tau factor for performing soft updates to target NN models
                ):

        self.prng = np.random.default_rng(random_seed) #the one and only PRNG in the entire system (must be passed around)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_noise = use_noise
        self.tau = update_factor
        
        # store the info for each type of ageint in use
        self.agent_types = {}
        env_info = env.reset(train_mode=True)
        for name in env.brain_names:
            brain = env.brains[name]
            type_info = env_info[name]
            ns = len(type_info.vector_observations[0])
            max_a = brain.vector_action_space_size
            num_agents = len(type_info.agents)
            actor = agent_models.get_actor_nn_for(name)
            critic = agent_models.get_critic_nn_for(name)
            actor_lr = agent_models.get_actor_lr_for(name)
            critic_lr = agent_models.get_critic_lr_for(name)
            actor_weight_decay = agent_models.get_actor_weight_decay_for(name)
            critic_weight_decay = agent_models.get_critic_weight_decay_for(name)
            self.agent_types[name] = AgentType(DEVICE, name, brain, ns, max_a, num_agents, actor, critic,
                                               actor_lr, critic_lr, actor_weight_decay, critic_weight_decay)
        
        # initialize other internal stuff
        self.learning_underway = False 
        self.learn_control = 0          #num time steps between learning events
        self.learn_every = 1            #number of time steps between learning events

        # define simple experience replay buffer common to all agents
        self.erb = ReplayBuffer(buffer_size, batch_size, buffer_prime_size, self.prng)

    #------------------------------------------------------------------------------

    """Returns a dict of AgentType that provides info on all the types in use."""

    def get_agent_types(self):
        return self.agent_types

    #------------------------------------------------------------------------------

    """Returns true if learning is underway, false if replay buffer is being primed."""
    
    def is_learning_underway(self):
        return self.learning_underway
    
    #------------------------------------------------------------------------------

    """Performs any resets that are necessary when a new episode is to begin."""

    def reset(self):
        pass #Nothing needed at this time

    #------------------------------------------------------------------------------

    """Computes the next actions for all agents, given the current state info.

        Return is a dict of actions, one entry for each agent type.  Each ndarray contains one entry for each 
        agent of that type.
    """

    def act(self, 
            states          : {},           # dict of current states; each entry represents an agent type,
                                            #   which is ndarray[num_agents, x]
            is_inference    : bool = False, # are we performing an inference using the current policy?
            add_noise       : bool = False  # are we to add random noise to the action output?
           ):

        act = {}

        if self.learning_underway  or  is_inference:
            for t in self.agent_types:
                actions = np.array((1, t.num_agents), dtype=int) #one element for each agent of this type
                t.actor_policy.eval()

                for i in range(t.num_agents):

                    # get the action for this agent
                    s = torch.from_numpy(states[t][i]).float().to(DEVICE)
                    with torch.no_grad():
                        actions[i] = t.actor_policy(s).cpu().data.numpy()

                    # add noise if appropriate
                    if add_noise:
                        pass

                t.actor_policy.train()

            act[t] = actions

        else: # must be priming the replay buffer
            for t in self.agent_types:
                actions = np.array((1, t.num_agents), dtype=int)
                actions = self.prng.integers(0, t.max_action_val, t.num_agents, dtype=int, endpoint=False)
            act[t] = actions

        return act

    #------------------------------------------------------------------------------

        """Stores a new experience from the environment in replay buffer, if appropriate,
           and advances the agents by one time step.

           Return:  none
        """

    def step(self, 
             states         : {},           # dict of current states; each entry represents an agent type,
                                            #   which is ndarray[num_agents, x]
             actions        : {},           # dict of actions taken; each entry is an agent type, which is 
                                            #   an ndarray[num_agents, x]
             rewards        : {},           # dict of rewards, each entry an agent type, which is a list of floats
             next_states    : {},           # dict of next states after actions are taken; each entry an agent type
             dones          : {}            # dict of done flags, each entry an agent type, which is a list of bools
            ):
        
        # set up probability of keeping bad experiences based upon whether the buffer is
        # full enough to start learning
        if len(self.erb) > max(self.batch_size, self.buffer_prime_size):
            threshold = self.bad_step_keep_prob
            self.learning_underway = True
        else:
            threshold = BAD_STEP_KEEP_PROB_INIT

        # if this step got some reward then keep it;
        # if it did not score any points, then use random draw to decide if it's a keeper
        if get_max(rewards) > 0.0  or  self.prng.random() < threshold:
            self.erb.add(states, actions, rewards, next_states, dones)

        # initiate learning on each agent, but only every N time steps
        self.learn_control += 1
        if self.learning_underway:

            # perform the learning if it is time
            if self.learn_control >= self.learn_every:
                self.learn_control = 0
                experiences = self.erb.sample()
                self.learn(experiences)

            """For possible future implementation:
            # update learning rate annealing; this is counting episodes, not time steps
            if is_episode_done(dones):
                for i in range(self.num_agents):
                    self.actor_scheduler[i].step()
                    self.critic_scheduler[i].step()
                    clr = self.critic_scheduler[i].get_lr()[0]
                    if clr < self.prev_clr: # assumes both critics have same LR schedule
                        if clr < 1.0e-7:
                            print("\n*** CAUTION: low learning rates: {:.7f}, {:.7f}" \
                                  .format(self.actor_scheduler[0].get_lr()[0], clr))
                        self.prev_clr = clr
            """

    #------------------------------------------------------------------------------

    """Update policy and value parameters using the given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Each agent type will learn in its own loop. Since all agents of a given type share a NN model, the number
        of learning iterations can be dependent on the number of agents of that type. An experience tuple will be
        weighted toward those types with the most individual agents. Therefore, types with few agents may be better
        trained for multiple iterations at each training step, in order to keep total learning iterations on the
        models more-or-less even across all NN models.
    """

    def learn(self,
              experiences   : ()    # tuple of dicts (s, a, r, s', done), where each dict contains an entry for each
                                    #   agent type. Each of these entries is a tensor of [b, a, x], where b is
                                    #   batch size, a is number of agents of that type, and
                                    #       x is number of states (for one agent of that type) for s and s'
                                    #       x is number of action values (for one agent of that type) for a
                                    #       x is 1 for r and done
             ):

        #.........Prepare the input data

        # extract the elements of the replayed batch of experiences
        states, actions, rewards, next_states, dones = experiences

        # find the total numbers of states & actions across all agents
        num_states_all = 0
        num_actions_all = 0
        for t in self.agent_types:
            num_states_all += t.state_size * t.num_agents
            num_actions_all += 1 * t.num_agents #for now we only define 1 enumerated action per agent
        
        # create tensors to hold states, actions and next_states for all agents where all agents are
        # represented in a single row (each row is an experience in the training batch) so size is [b, x]
        first = True
        for t in self.agent_types:
            s = states[t].view(self.batch_size, -1)
            a = actions[t].view(self.batch_size, -1)
            n = next_states[t].view(self.batch_size, -1)
            if first:
                states_all = s.to(DEVICE)
                actions_all = a.to(DEVICE)
                next_states_all = n.to(DEVICE)
                first = False

            else:
                states_all = torch.cat((states_all, s), dim=1)
                actions_all = torch.cat((actions_all, a), dim=1)
                next_states_all = torch.cat((next_states_all, n), dim=1)

        #.........Use the current actor NNs to get possible action values

        # need to do this for all agents before updating the critics, since critics see all
        first = True
        for t in self.agent_types:
            for agent in range(t.num_agents):

                # grab next state vectors and use this agent's target network to predict next actions
                ns = next_states[t][:, agent, :]
                ta = t.actor_target(ns)

                # grab current state vector and us this agent's current policy to decide current actions
                cs = states[t][:, agent, :]
                ca = t.actor_policy(cs)

                if first:
                    target_actions = ta.to(DEVICE)
                    cur_actions = ca.to(DEVICE)
                    first = False
                else:
                    target_actions = tensor.cat((target_actions, ta), dim=1)
                    cur_actions = tensor.cat((cur_actions, ca), dim=1)
                
                # resulting target_actions and cur_actions tensors are of shape [b, z], where z is the
                # sum of all agents' action spaces (all agents are represented in a single row)

        #.........Update the critic NNs based on learning losses

        print("///// DIDN'T FINISH WRITING learn()!")

        #.........Update the actor NNs based on learning losses


        #.........Update the target NNs for both critics & actors







    #------------------------------------------------------------------------------


    def save_checkpoint(self, path, name, episode):
        pass #TODO: dummy

    #------------------------------------------------------------------------------


    """Gets statistics on the replay buffer memory contents.

        Return:  tuple of (size, good_exp), where size is total number
                    of items in the buffer, and good_exp is the number of those
                    items with a reward that exceeds the threshold of "good".
    """

    def get_erb_stats(self):

        return (len(self.erb), self.erb.num_rewards_exceeding_threshold())

    #------------------------------------------------------------------------------

    """Returns true if at least one flag in the input dict is true, false otherwise."""

    def is_episode_done(self,
                        flags   : {}    # dict of lists of bool flags indicating completion
                       ):
        
        for flag_list in flags:
            for flag in flag_list:
                if flag:
                    return True
        
        return False

    

    #------------------------------------------------------------------------------


    """Updates the target model parameters to be a little closer to those of the policy model.
        θ_target = τ*θ_policy + (1 - τ)*θ_target; where tau < 1
    """

    def soft_update(self, 
                    policy_model    : nn.Module,    # weights copied from here
                    target_model    : nn.Module     # weights copied to here
                   ):

        for tgt, pol in zip(target_model.parameters(), policy_model.parameters()):
            tgt.data.copy_(self.tau*pol.data + (1.0-self.tau)*tgt.data)

