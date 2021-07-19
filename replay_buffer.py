""" ReplayBuffer class provides a fixed-size replay buffer (without true priority) for randomly sampling past
    experiences. However, it recognizes experiences with large rewards as valuable,
    so will retain them when they have hit the left end of the queue and the queue is full.
    Assumes that the initial experiences added (until prime_size is reached) will be
    random content to prime the buffer, and will allow these primes to be pushed off the
    end, regardless of whether they have a good reward, and will not count them when
    reporting "good" experiences.
"""

import numpy as np
import torch
import random
from collections import namedtuple, deque

from utils      import get_max

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])



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
                
def debug_actions_tensor(types, actions, states, flag):
    first_key = list(types.keys())[0]
    for sample in range(states[first_key].shape[0]): #loop over number of samples in the batch
        print("...sample {}:".format(sample))
        for t in types:
            for agent in range(states[t].shape[1]):

                # get each of the 14 ray traces from the current time step & see if first element indicates it sees the ball
                start = 2*112
                is_ball = np.empty(14, dtype=bool)
                for ray in range(14):
                    is_ball[ray] = states[t][sample, agent, start + 8*ray] > 0.5 #first element in each 8-element ray indicates it sees the ball
                
                # if it sees the ball to the left
                if is_ball[4]  or  is_ball[11]  or  is_ball[3]  or  is_ball[10]:
                    print("{}\t{}: Ball left\tAction ".format(t, agent), end="")
                
                # if it sees the ball to the right
                elif is_ball[0]  or  is_ball[7]  or  is_ball[8]  or  is_ball[1]:
                    print("{}\t{}: Ball right\tAction ".format(t, agent), end="")
                
                # if it sees the ball in front
                elif is_ball[12]  or  is_ball[13]  or  is_ball[2]  or  is_ball[5]  or is_ball[6]  or  is_ball[9]:
                    print("{}\t{}: Ball fwd\tAction ".format(t, agent), end="")

                # else we don't know where the ball is
                else:
                    print("{}\t{}: Ball unknown\tAction ".format(t, agent), end="")

                for j in range(actions[t].shape[2]):
                    print("{:6.4f} ".format(actions[t][sample, agent, j]), end="")
                print("")
                


class ReplayBuffer:

    """ Initialize a ReplayBuffer object."""

    def __init__(self, 
                 buffer_size    : int,                  # max num experiences that can be stored
                 batch_size     : int,                  # num experiences pulled for each training batch in a sample
                 prime_size     : int,                  # num initial experiences that are considered random priming
                                                        #   data; don't need to be kept once buffer is full
                 reward_thresh  : float,                # reward value above which is considered a "good" experience
                 prng           : np.random.Generator   # the pseudo-random number generator
                ):

        # sanity check some inputs
        if batch_size >= buffer_size:
            print("\n\n///// WARNING: batch size {} > replay buffer size of {}\n".format(batch_size, buffer_size))
        if prime_size >= buffer_size:
            print("\n\n///// WARNING: replay buffer prime size {} > buffer size of {}\n".format(prime_size, buffer_size))

        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  #this is the buffer
        self.batch_size = batch_size
        self.prime_size = prime_size
        self.reward_threshold = reward_thresh
        self.prng = prng

        self.rewards_exceed_threshold = 0
        self.num_primes = 0

    #------------------------------------------------------------------------------

        #TODO:  this section for debug only!
        self.agent_types = None

    def store_types(self, t):
        self.agent_types = t

    #------------------------------------------------------------------------------

    """Add a new experience to memory."""
    """Add a new experience to the buffer.
		We want to preserve existing 'good' experiences to a large extent.  When the
		buffer gets full, it will push priming experiences off, but should retain any
		other good experiences unless more than 50% of the buffer is already filled
		with good ones.
	"""

    def add(self,
             states         : {},           # dict of current states; each entry represents an agent type,
                                            #   which is ndarray[num_agents, x]
             actions        : {},           # dict of actions taken; each entry is an agent type, which is 
                                            #   an ndarray[num_agents, x]
             rewards        : {},           # dict of rewards, each entry an agent type, which is a list of floats
             next_states    : {},           # dict of next states after actions are taken; each entry an agent type
             dones          : {}            # dict of done flags, each entry an agent type, which is a list of bools
           ):




        #print("replay_buffer.add():")
        #debug_actions(self.agent_types, actions, states, " ")





        # if the buffer is already full then (we don't want to lose good experiences)
        if len(self.memory) == self.buffer_size:

            # if we have already pushed the initial priming experiences off the end then
            if self.num_primes == 0:

                # if < 50% of the buffer's contents are good experiences then
                if self.rewards_exceed_threshold < self.buffer_size//2:

                    # while we have a desirable reward at the left end of the deque
                    while get_max(self.memory[0].reward) > self.reward_threshold:

                        # pop it off and push it back onto the right end to save it
                        self.memory.rotate(-1)

            # else (some priming experiences still exist)
            else:

                # reduce the prime count since one will be pushed off the left end
                self.num_primes -= 1
                if self.num_primes == 0:
                    print("\n* Replay buffer - all primes flushed out.")

        # if number of entries < prime size, then count the entry as a prime
        if len(self.memory) < self.prime_size:
            self.num_primes += 1

        else:
            # if the incoming experience has a good reward, then increment the count
            if get_max(rewards) > self.reward_threshold:
                self.rewards_exceed_threshold += 1
    
        # add the experience to the right end of the deque
        e = Experience(states, actions, rewards, next_states, dones)
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
            experiences = self.prng.choice(self.memory, self.batch_size).tolist() #choice returns ndarray
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
                # NOTE: e0 is supposed to be a named tuple, but prng.choice is returning lists instead. So we index
                #       elements of the list:
                #           0..state
                #           1..action
                #           2..reward
                #           3..next_state
                #           4..done
                e0 = experiences[0] #experiences is a list of length batch size
                for agent_type in e0[0]:

                    # get num agents of this type (assume the same number of agents is represented in each element)
                    num_agents = e0[0][agent_type].shape[0]

                    # create empty tensors to hold all of the experience data for this agent type
                    ts = torch.zeros(self.batch_size, num_agents, e0[0][agent_type].shape[1], dtype=torch.float)
                    ta = torch.zeros(self.batch_size, num_agents, e0[1][agent_type].shape[1], dtype=torch.float)
                    tr = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)
                    tn = torch.zeros(self.batch_size, num_agents, e0[3][agent_type].shape[1], dtype=torch.float)
                    td = torch.zeros(self.batch_size, num_agents, 1, dtype=torch.float)

                    # loop through all the experiences, assigning each to a layer in the output tensor
                    for i, e in enumerate(experiences):
                        ts[i, :, :] = torch.from_numpy(e[0][agent_type])
                        ta[i, :, :] = torch.from_numpy(e[1][agent_type])
                        tn[i, :, :] = torch.from_numpy(e[3][agent_type])

                        # incoming reward and done are lists, not tensors
                        tr[i, :, :] = torch.tensor(e[2][agent_type]).view(num_agents, -1)[:, :]
                        td[i, :, :] = torch.tensor(e[4][agent_type]).view(num_agents, -1)[:, :]

                    # add these tensors into the output dictionaries
                    states[agent_type] = ts
                    actions[agent_type] = ta
                    rewards[agent_type] = tr
                    next_states[agent_type] = tn
                    dones[agent_type] = td



                #TODO debug only
                #print("replay_buffer.sample():")
                #debug_actions_tensor(self.agent_types, actions, states, " ")




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
