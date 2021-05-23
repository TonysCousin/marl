""" ReplayBuffer class provides a fixed-size replay buffer (without true priority) for randomly sampling past
    experiences. However, it recognizes experiences with large rewards as valuable,
    so will retain them when they have hit the left end of the queue.
    Assumes that the initial experiences added (until batch_size is reached) will be
    random garbage to prime the buffer, and will allow these primes to be pushed off the
    end, regardless of whether they have a good reward, and will not count them when
    reporting "good" experiences.
"""

import numpy as np
import torch
import random
from collections import namedtuple, deque

import utils

REWARD_THRESHOLD = 0.0 # value above which is considered a "good" experience


class ReplayBuffer:

    """ Initialize a ReplayBuffer object."""

    def __init__(self, 
                 action_size, #TODO figure out what to do with this!



                 buffer_size    : int,                  # max num experiences that can be stored
                 batch_size     : int,                  # num experiences pulled for each training batch in a sample
                 prime_size     : int,                  # num initial experiences that are considered random priming
                                                        #   data; don't need to be kept once buffer is full
                 prng           : np.random.Generator   # the pseudo-random number generator
                ):

        # sanity check some inputs
        if batch_size >= buffer_size:
            print("\n\n///// WARNING: batch size {} > replay buffer size of {}\n".format(batch_size, buffer_size))
        if prime_size >= buffer_size:
            print("\n\n///// WARNING: replay buffer prime size {} > buffer size of {}\n".format(prime_size, buffer_size))

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  #this is the buffer
        self.batch_size = batch_size
        self.prime_size = prime_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.rewards_exceed_threshold = 0
        self.num_primes = 0

    #------------------------------------------------------------------------------

    """Add a new experience to memory."""

    def add(self,
             states         : {},           # dict of current states; each entry represents an agent type,
                                            #   which is ndarray[num_agents, x]
             actions,       : {},           # dict of actions taken; each entry is an agent type, which is 
                                            #   an ndarray[num_agents, x]
             rewards,       : {},           # dict of rewards, each entry an agent type, which is a list of floats
             next_states    : {},           # dict of next states after actions are taken; each entry an agent type
             dones          : {}            # dict of done flags, each entry an agent type, which is a list of bools
           ):

        # if the buffer is already full then (we don't want to lose good experiences)
        if len(self.memory) == self.buffer_size:

            # if we have already pushed the initial priming experiences off the end then
            if self.num_primes == 0:

                # if < 50% of the buffer's contents are good experiences then
                if self.rewards_exceed_threshold < self.buffer_size//2:

                    # while we have a desirable reward at the left end of the deque
                    while max(self.memory[0].reward) > REWARD_THRESHOLD:

                        # pop it off and push it back onto the right end to save it
                        self.memory.rotate(-1)

            # else (some priming experiences still exist)
            else:

                # reduce the prime count since one will be pushed off the left end
                self.num_primes -= 1

        # if number of entries < prime size, then count the entry as a prime
        if len(self.memory) < self.prime_size:
            self.num_primes += 1

        else:
            # if the incoming experience has a good reward, then increment the count
            if get_max(reward) > REWARD_THRESHOLD:
                self.rewards_exceed_threshold += 1
    
        # add the experience to the right end of the deque
        e = self.experience(state, action, reward, next_state, dones)
        self.memory.append(e)

    #------------------------------------------------------------------------------

    """Returns the number of rewards in database that exceed the threshold of 'good' """

    def num_rewards_exceeding_threshold(self):
        return self.rewards_exceed_threshold

    #------------------------------------------------------------------------------

    """Randomly sample a batch of experiences from memory.

        Return: tuple of dicts of the individual elements of experience:
                    states, actions, rewards, next_states, dones
                Each dict holds elements that represent each agent type in use
                Each dict element is a tensor of shape (b, a, x), where
                    b is the number of items in a training batch (same for all elements)
                    a is the number of agents of that agent type (same for all elements)
                    x is the number of items in that element (different for each element)
                Each "row" of a tensor (set of x values) represents a single agent.
    """

    def sample(self):

        # randomly sample enough experiences from the buffer for one batch
        if len(self.memory) >= self.batch_size:

            # pull a random sample of experiences from the buffer
            experiences = prng.choice(self.memory, self.batch_size).tolist() #choice returns ndarray
            if len(experiences) == self.batch_size:

                # Since the internal storage of an experience is in the form of lists and ndarrays,
                # this is where we need to mash all those together into tensors so that a single tensor
                # will represent several experiences (batch_size) at once. For illustration, look at the
                # state data; other elements will be similar. The experiences we start with includes a
                # dict named experiences.state, which contains
                #   {agent_type_1 : s1, agent_type_2 : s2, ...}
                # where s1's type is ndarray[a1, x] and s2's type is ndarray[a2, y]
                # where a1 and a2 are num agents of those types, and x  and y are the sizes of the state
                # vectors for each agent of that type.
                # The desired output (for the state element) is a dict that contains
                #   {agent_type_1 : os1, agent_type_2 : os2, ...}
                # where os1's type is tensor[b, a1, x] and os2's type is tensor[b, a2, y]
                # where b is the batch size (number of elements in the starting experiences list).

                states = {}
                actions = {}
                rewards = {}
                next_states = {}
                dones = {}

                # loop through each agent type in use (assume all element dicts have the same keys)
                for agent_type in experiences[0].state:

                    # get num agents of this type (assume the same number of agents is represented in each element)
                    e0 = experiences[0]
                    num_agents = e0.state[agent_type].shape[0]
                    print("ReplayBuffer.sample: agent_type = ", agent_type, ", num_agents = ", num_agents)

                    # create empty tensors to hold all of the experience data for this agent type
                    ts = torch.zeros(self.batch_size, num_agents, e0.state[agent_type].shape[1], dtype=torch.float)
                    ta = torch.zeros(self.batch_size, num_agents, e0.action[agent_type].shape[1], dtype=torch.float)
                    tr = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)
                    tn = torch.zeros(self.batch_size, num_agents, e0.next_state[agent_type].shape[1], dtype=torch.float)
                    td = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)
                    print("                     ts = ", ts.shape)
                    print("                     ta = ", ta.shape)
                    print("                     tr = ", tr.shape)
                    print("                     tn = ", tn.shape)
                    print("                     td = ", td.shape)

                    # loop through all the experiences, assigning each to a layer in the output tensor
                    for i, e in enumerate(experiences):
                        ts[i, :, :] = torch.from_numpy(e.state[agent_type])
                        ta[i, :, :] = torch.from_numpy(e.action[agent_type])
                        tn[i, :, :] = torch.from_numpy(e.next_state[agent_type])

                        # incoming reward and done are lists, not tensors
                        tr[i, :, :] = torch.tensor(e.reward[agent_type]).view(num_agents, -1)[:, :]
                        td[i, :, :] = torch.tensor(e.done[agent_type]).view(num_agents, -1)[:, :]

                    # add these tensors into the output dictionaries
                    states[agent_type] = ts
                    actions[agent_type] = ta
                    rewards[agent_type] = tr
                    next_states[agent_type] = tn
                    dones[agent_type] = td

                return (states, actions, rewards, next_states, dones)

            else:
                print("\n///// ReplayBuffer.sample: unexpected experiences length = {} but batch size = {}"
                    .format(len(experiences), self.batch_size))
                return None
        
        else: #not enough content to fill a batch
            print("\n///// ReplayBuffer.sample: only {} experiences stored. Insufficient to fill a batch of {}."
                  .format(len(self.memory), self.batch_size))
            return None

    #------------------------------------------------------------------------------

    """Return the current number of experiences in the buffer."""

    def __len__(self):
        return len(self.memory)
