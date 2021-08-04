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
import torch.nn.functional as F

from unityagents    import UnityEnvironment
from agent_type     import AgentBehavior, AgentType
from agent_models   import AgentModels
from replay_buffer  import ReplayBuffer
from utils          import get_max

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# probability of keeping experiences with bad rewards during replay buffer priming
BAD_STEP_KEEP_PROB_INIT = 1.0

# fraction of time that is considered "most of it" for smoothing random actions
MOST_OF_TIME = 0.0

#TODO: this threshold should be a param, since it may be game dependent
# experience reward value above which is considered a desirable experience
REWARD_THRESHOLD = 0.0

# normal distribution when random action values are assigned
RANDOM_MEAN = 0.0
RANDOM_SD = 0.04




#------------------------------------------------------------------------------

def debug_actions(types, actions, states, flag):
        for t in types:
            for agent in range(states[t].shape[0]):

                # get each of the 14 ray traces from the current time step & see if first element indicates it sees the ball
                start = 2*112
                is_ball = np.empty(14, dtype=bool)
                for ray in range(14):
                    is_ball[ray] = states[t][agent, start + 8*ray] > 0.5 #first element in each 8-element ray indicates it sees the ball
                
                # if it sees the ball to the left
                if is_ball[4]  or  is_ball[11]  or  is_ball[3]  or  is_ball[10]:
                    print("{}\t{}: Ball left\tAction {}{}".format(t, agent, np.argmax(actions[t][agent]), flag))
                
                # if it sees the ball to the right
                elif is_ball[0]  or  is_ball[7]  or  is_ball[8]  or  is_ball[1]:
                    print("{}\t{}: Ball right\tAction {}{}".format(t, agent, np.argmax(actions[t][agent]), flag))
                
                # if it sees the ball in front
                elif is_ball[12]  or  is_ball[13]  or  is_ball[2]  or  is_ball[5]  or is_ball[6]  or  is_ball[9]:
                    print("{}\t{}: Ball fwd\tAction {}{}".format(t, agent, np.argmax(actions[t][agent]), flag))

                # else we don't know where the ball is
                else:
                    print("{}\t{}: Ball unknown\tAction {}{}".format(t, agent, np.argmax(actions[t][agent]), flag))
                




class AgentMgr:

    """Initialize the AgentMgr using a pre-established environment model."""

    def __init__(self,
                 env            : UnityEnvironment, # the envronment model that agents live in
                 agent_models   : AgentModels,      # holds all of the NN models for each type of agent
                 random_seed    : int = 0,          # seed for the PRNG; only used if the prng param is omitted
                 batch_size     : int = 32,         # number of experiences in a learning batch
                 buffer_size    : int = 100000,     # capacity of the experience replay buffer
                 buffer_prime   : int = 1000,       # number of experiences to be stored in replay buffer before learning begins
                 bad_step_prob  : float = 1.0,      # probability of keeping an experience (after buffer priming) with a low reward
                 use_noise      : bool = False,     # should we inject random noise into actions?
                 noise_init     : float = 1.0,      # initial probability of noise being added if it is turned on
                 noise_decay    : float = 0.9999,   # the amount the noise probability will be reduced after each experience
                 discount_factor: float = 0.99,     # Gamma factor for discounting future time step results
                 update_factor  : float = 0.001,    # Tau factor for performing soft updates to target NN models
                 prng           : np.random.Generator = None    # random number generator
                ):

        if prng == None:
            self.prng = np.random.default_rng(random_seed) #the one and only PRNG in the entire system (must be passed around)
        else:
            self.prng = prng

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_prime_size = buffer_prime
        self.bad_step_keep_prob = bad_step_prob
        self.use_noise = use_noise
        self.noise_decay = noise_decay
        self.noise_level = noise_init
        self.noise_reported1 = False #has first decay threshold been met and reported to user?
        self.noise_reported2 = False #has second decay threshold been met and reported to user?
        self.gamma = discount_factor
        self.tau = update_factor
        
        # store the info for each type of agent in use - for now we assume every agent is to be trained & used
        self.agent_types = {}
        env_info = env.reset(train_mode=True)
        for name in env.brain_names:
            brain = env.brains[name]
            type_info = env_info[name]
            ns = len(type_info.vector_observations[0])
            na = brain.vector_action_space_size
            num_agents = len(type_info.agents)
            actor = agent_models.get_actor_nn_for(name)
            critic = agent_models.get_critic_nn_for(name)
            actor_lr = agent_models.get_actor_lr_for(name)
            critic_lr = agent_models.get_critic_lr_for(name)
            actor_weight_decay = agent_models.get_actor_weight_decay_for(name)
            critic_weight_decay = agent_models.get_critic_weight_decay_for(name)
            self.agent_types[name] = AgentType(DEVICE, name, brain, ns, na, num_agents,
                                               actor, critic, actor_lr, critic_lr, actor_weight_decay, critic_weight_decay)
        
        # initialize other internal stuff
        self.learning_underway = True #if priming replay buffer, use False here and have step() flip it when buffer is primed
        self.learn_control = 0          #num time steps between learning events
        self.prev_act = {}              #holds actions from previous time step; each entry is an agent type tensor
        self.first_time_step = True     #is this the first time step of an episode?

        # find the total numbers of states & actions across all agents
        self.num_states_all = 0
        self.num_actions_all = 0
        self.num_agents_all = 0
        for t in self.agent_types:
            at = self.agent_types[t]
            self.num_states_all += at.state_size * at.num_agents
            self.num_actions_all += 1 * at.num_agents #for now we only define 1 enumerated action per agent
            self.num_agents_all += at.num_agents
            self.prev_act[t] = np.zeros((at.num_agents, 1), dtype=int) #one element for each agent of this type
        
        # define simple experience replay buffer common to all agents
        self.erb = ReplayBuffer(buffer_size, batch_size, buffer_prime, REWARD_THRESHOLD, self.prng)

        # set up accumulators for the step() method to record all the experiences for a single episode;
        # each list will hold dicts
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_next_st = []
        self.ep_dones = []

    #------------------------------------------------------------------------------

    """Flags a specific agent for modified behavior.  When training, an agent flagged for skipped training
        will be passed over in the learning algorithm (i.e. it will not contribute to NN param updates.
        An agent flagged for random actions will draw random actions instead of using the type's policy NN.
        If the random actions flag is false then the type's policy NN will be used to assign an action to
        this agent.  This may be useful even if it is skipping training, because it would be exercising a
        static policy (vice one that is continuing to learn), for example if this agent is playing a role
        in the scenario environment for other agents to learn about.
    """

    def modify_behavior(self,
                        agent_type     : AgentType,     # type of agent to be modified
                        agent_id       : int,           # sequential ID of the individual agent of this type to be modified
                        behavior       : AgentBehavior, # the behavior to be used by this agent
                        uniform_act    : [] = None,     # vector of raw action values to be used in a Uniform behavior
                        anneal_rate    : float = 1.0    # annealing rate to be used for AnnealedRandom behavior; value must be in [0, 1]
                       ):
        
        self.agent_types[agent_type].set_behavior(agent_id, behavior, uniform_act, anneal_rate)

    #------------------------------------------------------------------------------

    """Returns a score (double) for a single experience, based on the rewards earned
        by each agent that is flagged to be trained in that time step."""

    def compute_time_step_score(self, 
                                rewards : {}    # dict with each entry is a list of rewards for each agent of that type
                               ):
        
        score = 0.0
        for t in rewards:
            for i, r in enumerate(rewards[t]):
                if self.agent_types[t].behavior[i] == AgentBehavior.Learn:
                    score += r
        
        return score

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

    """Computes the next actions for all agents, given the current state info, and applies noise
        if appropriate.

        Return is a dict of actions, one entry for each agent type.  Each ndarray contains one int for each 
        agent of that type, with the shape [num_agents, 1].
    """

    def act(self, 
            states          : {},           # dict of current states; each entry represents an agent type,
                                            #   which is ndarray[num_agents, x]
            is_inference    : bool = False, # are we performing an inference using the current policy?
            add_noise       : bool = False  # are we to add random noise to the action output?
           ):

        raw = self.get_raw_action_vector(states, is_inference, add_noise)
        actions = self.find_best_action(raw)

        return actions

    #------------------------------------------------------------------------------

    """Computes the next actions for all agents, given the current state info. Randomizes
        those actions if appropriate.

        Return is a dict of ndarrays of actions, one entry for each agent type.  Each ndarray 
        contains a full row vector of all possible actions for each agent of that type.  
        Shape is [num_agents, num_actions].
    """

    def get_raw_action_vector(self, 
                              states          : {},           # dict of current states; each entry represents an agent type,
                                                              #   which is ndarray[num_agents, x]
                              is_inference    : bool = False, # are we performing an inference using the current policy?
                              add_noise       : bool = False  # are we to add random noise to the action output?
                             ):

        act = {}


        flags = "" #TODO: debug only



        if self.learning_underway  or  is_inference:
            for t in self.agent_types:
                at = self.agent_types[t]
                actions = np.empty((at.num_agents, at.action_size), dtype=float) #one row for each agent of this type
                at.actor_policy.eval()

                for i in range(at.num_agents):
                    noise_added = False

                    # if this agent is assigned to be random or annealing toward randomness then assing random action
                    if at.behavior[i] == AgentBehavior.Random  or  at.behavior[i] == AgentBehavior.AnnealedRandom:

                        # for now, only Random is supported
                        actions[i, :] = self.prng.normal(RANDOM_MEAN, RANDOM_SD, at.action_size) #approximate the policy NN outputs
                        noise_added = True
                        flags = flags + '-'

                    # else if this agent is to be learning then
                    elif at.behavior[i] == AgentBehavior.Learn:

                        # add noise if appropriate by selecting a random action
                        if add_noise  or  self.use_noise:
                            if self.prng.random() < self.noise_level:

                                # Now that we have decided to make some noise for this agent, we would like the action to be
                                # not quite random.  Most of the time we want it to continue doing what it did in the previous
                                # time step so as to exhibit smooth motion, not herky-jerky.  Therefore, if a random draw in
                                # [0, 1) < "most of the time" threshold, just copy the previous action
                                if not self.first_time_step  and  self.prng.random() < MOST_OF_TIME:
                                    actions[i, :] = self.prev_act[t][i, :]
                                
                                # otherwise, let's pick a truly random action to have fun with
                                else:
                                    actions[i, :] = self.prng.normal(RANDOM_MEAN, RANDOM_SD, at.action_size) #approximate the policy NN outputs

                                noise_added = True
                                flags = flags + '-'

                        # else get the action for this agent from its policy NN
                        if not noise_added:
                            s = torch.from_numpy(states[t][i]).float().to(DEVICE)
                            with torch.no_grad():
                                actions[i] = at.actor_policy(s).cpu().data.numpy() #returns an array representing all possible actions for this agent
                            flags = flags + '!'
                    
                    # else if this agent is to follow policy only then do so
                    elif at.behavior[i] == AgentBehavior.Policy:
                        s = torch.from_numpy(states[t][i]).float().to(DEVICE)
                        with torch.no_grad():
                            actions[i] = at.actor_policy(s).cpu().data.numpy() #returns an array representing all possible actions for this agent
                        flags = flags + '!'
                    
                    # else if this agent is to exhibit uniform actions then
                    elif at.behavior[i] == AgentBehavior.Uniform:

                        # use the given uniform action vector as the output
                        actions[i] = at.uniform_action_vector[i]
                    
                    # else - unknown behavior
                    else:
                        print("\n///// ERROR: AgentMgr.get_raw_action_vector: unkown agent behavior ", at.behavior[i])
                        return act


                at.actor_policy.train()

                act[t] = actions
                self.prev_act[t] = actions

            # reduce the noise probability
            if add_noise  or  self.use_noise:
                self.noise_level *= self.noise_decay
                if self.noise_level <= 0.1  and  not self.noise_reported1:
                    print("\n* Noise decayed to 0.1")
                    self.noise_reported1 = True
                if self.noise_level <= 0.01  and  not self.noise_reported2:
                    print("\n* Noise decayed to 0.01")
                    self.noise_reported2 = True

        else: # not learning or inference, so must be priming the replay buffer
            for t in self.agent_types:
                at = self.agent_types[t]
                actions = np.empty((at.num_agents, at.action_size), dtype=float) #one row for each agent of this type
                for i in range(at.num_agents):
                    actions[i, :] = self.prng.normal(RANDOM_MEAN, RANDOM_SD, at.action_size) #approximate the policy NN outputs
                act[t] = actions

        #print("act: ", flags)
        #debug_actions(self.agent_types, act, states, " ")

        return act

    #------------------------------------------------------------------------------

    """Given a vector of possible action choices, it returns the best action (the
        index of the element with the largest score).

        Return:  dict of ndarrays holding integer index of the best action for each agent (on cpu)
    """

    def find_best_action(self,
                         action_vector  : {}    # dict of ndarrays of raw values for each possible action;
                                                #   each entry is of shape [a, x], one row for each agent of that type
                        ):
        
        best = {}
        
        for t in self.agent_types:
            at = self.agent_types[t]
            actions = np.empty(at.num_agents, dtype=int)
            for agent in range(at.num_agents):
                vec = action_vector[t][agent, :]
                actions[agent] = np.argmax(vec)
            best[t] = actions
        
        return best

    #------------------------------------------------------------------------------

    """Accumulates data for each experience in an episode. 

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

        # accumulate the data from this time step into the episode lists
        self.ep_states.append(states)
        self.ep_actions.append(actions)
        self.ep_rewards.append(rewards)
        self.ep_next_st.append(next_states)
        self.ep_dones.append(dones)

        self.first_time_step = False

    #------------------------------------------------------------------------------

    """Calculate discounted rewards from the previous episode (accumulated by the step() method)
        and store the new experiences in the replay buffer. Do this for all agents, whether they
        are learning or not, and whether they are using their policies or not; this way we will have
        a full experience across all agents to store for those who are learning.
    """
    
    def store_discounted_experiences(self):

        # compute the discounted rewards for each time step to the end of the episode
        num_time_steps = len(self.ep_rewards)
        print("num time steps = ", num_time_steps)


        discount = self.gamma ** np.arange(num_time_steps) #multipliers to be applied to future time steps

        # for each agent pull out its raw rewards then apply the discount factors, sum the future rewards, then
        # replace them in the original data structure; ep_rewards is a list of dicts of lists
        for t in self.agent_types:
            at = self.agent_types[t]
            for agent in range(at.num_agents):
                dr = []
                for step in range(num_time_steps):
                    dr.append(self.ep_rewards[step][t][agent]) #becomes a list of raw rewards for this agent
                
                dr = np.array(dr) * discount
                future_reward = dr[::-1].cumsum(axis=0)[::-1]

                for step in range(num_time_steps):
                    self.ep_rewards[step][t][agent] = future_reward[step]

        # each time step is an experience that needs to be added to the replay buffer
        for step in range(num_time_steps):
            s = self.ep_states[step]    #each of these items is a dict
            a = self.ep_actions[step]
            r = self.ep_rewards[step]
            n = self.ep_next_st[step]
            d = self.ep_dones[step]
            self.erb.add(s, a, r, n, d)

        # clear the episode accumulators to prepare for the next episode
        self.ep_states.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()
        self.ep_next_st.clear()
        self.ep_dones.clear()
        self.first_time_step = True

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

        CAUTION: assumes it will only be called after step() has been called at least once for a new episode, so
        that the episode accumulator lists are not empty (intention is to only call this method at the end of
        an episode).

        Return:  none
    """

    def learn(self):

        #.........Prepare the learning batch data

        # make sure we have enough experiences in the buffer to do a learning batch
        if len(self.erb) < self.batch_size:
            return

        # sample a batch of experiences from the replay buffer and extract the elements
        states, actions, rewards, next_states, dones = self.erb.sample()

        # create tensors to hold states, actions and next_states for all agents where all agents are
        # represented in a single row (each row is an experience in the training batch) so size is [b, x].
        first = True
        for t in self.agent_types:
            s = states[t].view(self.batch_size, -1) #puts all agents of this type onto one row
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
            at = self.agent_types[t]
            for agent in range(at.num_agents):

                # if this agent is using its policy NN then
                if at.behavior[agent] == AgentBehavior.Learn  or  at.behavior[agent] == AgentBehavior.Policy:

                    # grab next state vectors and use this agent's target network to predict next actions
                    ns = next_states[t][:, agent, :]
                    ta = at.actor_target(ns).detach() #vector of all possible actions

                    # grab current state vector and use this agent's current policy to decide current actions
                    cs = states[t][:, agent, :]
                    ca = at.actor_policy(cs).detach() #vector of all possbile actions
                
                # else, assign a random action vector to both the target and current actions
                else:
                    av_raw = self.prng.normal(RANDOM_MEAN, RANDOM_SD, self.batch_size * at.action_size)
                    ta = torch.from_numpy(av_raw).float().view(self.batch_size, -1)
                    ca = ta

                if first:
                    target_actions = ta
                    cur_actions = ca
                    first = False
                else:
                    target_actions = torch.cat((target_actions, ta), dim=1)
                    cur_actions = torch.cat((cur_actions, ca), dim=1)
                
                # resulting target_actions and cur_actions tensors are of shape [b, z], where z is the
                # sum of all agents' action spaces for that agent type (all agents are represented in each row)

        #.........Update the critic NNs based on learning losses

        for t in self.agent_types:
            at = self.agent_types[t]

            # if there is at least one agent of this type that is being trained then
            num_agents_being_trained = self.count_agents_being_trained(at)
            if num_agents_being_trained > 0:
                agents_updated = 0

                # compute the Q values for the next states/actions from the target model for this type
                q_targets_next = at.critic_target(next_states_all, target_actions).squeeze()

                # prepare the rewards & dones for the agent type
                r = rewards[t].squeeze(2).to(DEVICE)
                d = dones[t].squeeze(2).to(DEVICE)

                for agent in range(at.num_agents):

                    # if this agent is learning then
                    if at.behavior[agent] == AgentBehavior.Learn:

                        # Compute Q targets for current states (y_i) for this agent
                        q_targets = r[:, agent] + self.gamma*q_targets_next*(1.0 - d[:, agent])

                        # use the current policy to compute the expected Q value for current states & actions
                        q_expected = at.critic_policy(states_all, actions_all).squeeze()

                        # use the current policy to compute the critic loss for this agent
                        critic_loss = F.mse_loss(q_expected, q_targets) #q_targets was previously detached


                        #if t == "StrikerBrain":
                            #print("critic_loss = ", critic_loss)

                        # minimize the loss
                        at.critic_opt.zero_grad()
                        retain = agents_updated < num_agents_being_trained - 1
                        critic_loss.backward(retain_graph=retain)
                        torch.nn.utils.clip_grad_norm_(at.critic_policy.parameters(), 1.0)
                        at.critic_opt.step()
                        agents_updated += 1

        #.........Update the actor NNs based on learning losses

        for t in self.agent_types:
            at = self.agent_types[t]
            num_agents_being_trained = self.count_agents_being_trained(at)
            agents_updated = 0

            for agent in range(at.num_agents):

                # if we are training this agent then
                if at.behavior[agent] == AgentBehavior.Learn:

                    # compute the actor loss
                    actor_loss = -at.critic_policy(states_all, cur_actions).mean()

                    #if t == "StrikerBrain":
                        #print("actor_loss = ", actor_loss)

                    # minimize the loss
                    retain = agents_updated < num_agents_being_trained - 1 #retain graph for all but the final agent
                    at.actor_opt.zero_grad()
                    actor_loss.backward(retain_graph=retain)
                    torch.nn.utils.clip_grad_norm_(at.actor_policy.parameters(), 1.0)
                    at.actor_opt.step()
                    agents_updated += 1

        #.........Update the target NNs for both critics & actors

        # loop on each agent in the problem; it would be cheaper to use a larger update rate (tau), but that is
        # an approximation for applying the soft udpate multiple times, and it would assume that each agent type
        # is represented by the same number of agents as all other types
        for t in self.agent_types:
            at = self.agent_types[t]
            for agent in range(at.num_agents):

                # if this agent is being trained then
                if at.behavior[agent] == AgentBehavior.Learn:

                    # perform a soft update on the critic & actor target NNs for each agent type
                    self.soft_update(at.critic_policy, at.critic_target)
                    self.soft_update(at.actor_policy, at.actor_target)

    #------------------------------------------------------------------------------

    # TODO: debug only!  Assumes striker actor l1 = 1024 and l2 = 256, critic l1 = 2048 and l2 = 256

    def print_actor_params(self, num):
        p_iter = self.agent_types['StrikerBrain'].actor_policy.parameters()
        ps = []
        for i, p in enumerate(p_iter):
            if i == 0:
                ps.append(p[1, 1].item())
                ps.append(p[214, 112].item())
                ps.append(p[891, 55].item())
            elif i == 1:
                ps.append(p[999].item())
            elif i == 2:
                ps.append(p[6, 808].item())
                ps.append(p[77, 777].item())
                ps.append(p[200, 201].item())
            elif i == 3:
                ps.append(p[51].item())
            elif i == 4:
                ps.append(p[1, 120].item())
                ps.append(p[3, 0].item())
                ps.append(p[4, 19])
            elif i == 5:
                ps.append(p[2].item())
            else:
                print("Unknown parameter", i)
        print("Striker actor {:5d}:".format(num), end="")
        for i in range(len(ps)):
            print(" {:10.7f}".format(ps[i]), end="")
        print("")

    def print_critic_params(self, num):
        p_iter = self.agent_types['StrikerBrain'].critic_policy.parameters()
        ps = []
        for i, p in enumerate(p_iter):
            if i == 0:
                ps.append(p[1, 1].item())
                ps.append(p[214, 112].item())
                ps.append(p[891, 55].item())
            elif i == 1:
                ps.append(p[999].item())
            elif i == 2:
                ps.append(p[6, 808].item())
                ps.append(p[77, 777].item())
                ps.append(p[200, 201].item())
            elif i == 3:
                ps.append(p[51].item())
            elif i == 4:
                ps.append(p[0, 120].item())
                ps.append(p[0, 0].item())
                ps.append(p[0, 19])
            elif i == 5:
                ps.append(p[0].item())
            else:
                print("Unknown parameter", i)
        print("Striker critic {:5d}:".format(num), end="")
        for i in range(len(ps)):
            print(" {:10.7f}".format(ps[i]), end="")
        print("")

    #------------------------------------------------------------------------------

    """Counts the number of agents of the given type that are being asked to learn."""

    def count_agents_being_trained(self,
                                   at    : AgentType # the type of agent we are counting
                                  ):

            count = 0
            for a in range(at.num_agents):
                if at.behavior[a] == AgentBehavior.Learn:
                    count += 1
            return count

    #------------------------------------------------------------------------------

    """Stores a checkpoint file with all intermediate info for the NNs and optimizers.

        Return: none

        There is no replay buffer stored. Checkpoints are saved as a single file for
        the entire model, which is structured as a dictionary containing the following
        fields:
            version
            actor-*
            opt_actor-*
            critic-*
            opt_critic-*
        where the * represents the names of each agent type in use (there are four fields
        specific to each agent type). Each field except "version" holds a state_dict.
        Version is a string.

        Version marl1:  initial
                marl2:  updated NN actors to output an array of possible actions rather
                         than a single value representing the chosen action.
    """

    def save_checkpoint(self, 
                        path    : str = None,   # directory where the files will go (if not None, needs to end in /)
                        name    : str = "ZZ",   # arbitrary name for the test, run, etc.
                        episode : int    = 0    # learning episode that this checkpoint represents
                       ):

        checkpoint = {}
        checkpoint["version"] = "marl2"

        for t in self.agent_types:
            at = self.agent_types[t]
            key_a = "actor-{}".format(t)
            key_oa = "opt_actor-{}".format(t)
            checkpoint[key_a] = at.actor_policy.state_dict()
            checkpoint[key_oa] = at.actor_opt.state_dict()
            key_c = "critic-{}".format(t)
            key_oc = "opt_critic-{}".format(t)
            checkpoint[key_c] = at.critic_policy.state_dict()
            checkpoint[key_oc] = at.critic_opt.state_dict()

        filename = "{}{}_{}.pt".format(path, name, episode)
        torch.save(checkpoint, filename)

    #------------------------------------------------------------------------------

    """Retrieves a checkpoint file and loads its data to initialize both actor & critic policy NNs
        and optimizer parameters for all agent types.  Checkpoint structures are as defined above
        in the description for save_checkpoint().
    """

    def restore_checkpoint(self, 
                           path     : str,   # directory path where the checkpoint file is stored (if not None, must end in /)
                           name     : str,   # a unique name for the training run
                           episode  : int    # the training episode number at which the checkpoint was stored
                          ):

        filename = "{}{}_{}.pt".format(path, name, episode)
        checkpoint = torch.load(filename)

        # check that the data file is compatible with the current needs
        file_ver = checkpoint["version"]
        needed_ver = "marl2"

        if file_ver != needed_ver:
            print("\n///// ERROR: unable to load checkpoint {}, which was built with {} software." \
                    .format(filename, file_ver))
            print("             Version {} checkpoints required.".format(needed_ver))
            return

        for t in self.agent_types:
            at = self.agent_types[t]
            key_a = "actor-{}".format(t)
            key_oa = "opt_actor-{}".format(t)
            at.actor_policy.load_state_dict(checkpoint[key_a])
            at.actor_opt.load_state_dict(checkpoint[key_oa])
            key_c = "critic-{}".format(t)
            key_oc = "opt_critic-{}".format(t)
            at.critic_policy.load_state_dict(checkpoint[key_c])
            at.critic_opt.load_state_dict(checkpoint[key_oc])

        print("Checkpoint loaded from {}".format(filename))

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

    #TODO: consider only looking at last item in each list, since that should be the only one that has a true value?

    def is_episode_done(self,
                        flags   : {}    # dict of lists of bool flags indicating completion
                       ):
        
        for flag_list in flags:
            for flag in flag_list:
                if flag:
                    return True
        
        return False

    #------------------------------------------------------------------------------

    """Returns the noise level currently in use."""

    def get_noise_level(self):
        return self.noise_level

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

