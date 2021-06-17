# Performs the training of all agents of all types. For now, it assumes the environment is
# based on the Unity ML-Agents framework.

import numpy as np
import torch
import time
import copy
from collections    import deque
from unityagents    import UnityEnvironment

from agent_type     import AgentType
from agent_mgr      import AgentMgr

AVG_SCORE_EXTENT = 100 # number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" # can be empty string, but if a dir is named, needs trailing '/'
ABORT_EPISODE = 400 # num episodes after which training will abort if insignificant learning is detected
PRIME_FEEDBACK_INTERVAL = 500 # num time steps between visual feedback of priming progress


"""Trains a set of DRL agents in a Unity ML-Agents environment.

   Return: tuple of (list of episode scores (floats), list of running avg scores (floats))
"""

def train(mgr               : AgentMgr,         # manages all agents and their learning process
          env               : UnityEnvironment, # the envronment model that agents live in
          run_name          : str = "UNDEF",    # tag for config control of learining session
          starting_episode  : int    = 0,       # episode to begin counting from (used if restarting
                                                #   from a checkpoint)
          max_episodes      : int    = 2,       # max num episodes to train if goal isn't met
          max_time_steps    : int    = 100,     # num time steps allowed in an episode
          sleeping          : bool   = False,   # should the code pause for a few sec after selected
                                                #   episodes? (allows for better visualizing)
          training_goal     : float  = 0.0,     # when avg score (over AVG_SCORE_EXTENT consecutive
                                                #   episodes) exceeds this value, training is done
          chkpt_interval    : int    = 100      # num episodes between checkpoints being stored
         ):

    #TODO Future: Have AgentMgr manage the env also so that train() never has to see it?

    # Initialize Unity simulation environment
    states = {}
    actions = {}
    rewards = {}
    dones = {}

    # set up to collect raw & running avg scores at each episode
    scores = []
    avg_scores = []
    sum_steps = 0 #accumulates number of time steps exercised
    max_steps_experienced = 0
    recent_scores = deque(maxlen=AVG_SCORE_EXTENT)
    start_time = 0
    agent_types = mgr.get_agent_types()

    # run the simulation for several time steps to prime the replay buffer
    print("Priming the replay buffer", end="")
    pc = 0
    env_info = env.reset(train_mode=True)
    states = all_agent_states(env_info, agent_types) #initial state vectors after env reset
    while not mgr.is_learning_underway():
        states, rewards, dones = advance_time_step(mgr, env, agent_types, states)
        if pc % PRIME_FEEDBACK_INTERVAL == 0:
            print(".", end="")
        pc += 1
        if any_dones(agent_types, dones): # if episode ends just keep going
            env_info = env.reset(train_mode=True)
            states = all_agent_states(env_info, agent_types)
    print("!\n")

    # loop on episodes for training
    start_time = time.perf_counter()
    for ep in range(starting_episode, max_episodes):
        
        # Reset the enviroment & agents and get their initial states
        env_info = env.reset(train_mode=True)
        states = all_agent_states(env_info, agent_types)
        mgr.reset()
        score = 0 # total score for this episode

        # loop over time steps
        for i in range(max_time_steps):

            # advance the MADDPG model and its environment by one time step
            states, rewards, dones = advance_time_step(mgr, env, agent_types, states)

            # add this step's reward to the episode score
            score += max_rewards(agent_types, rewards)

            # if the episode is complete, update the record time steps in an episode and stop the time step loop
            if any_dones(agent_types, dones):
                sum_steps += i
                if i > max_steps_experienced:
                    max_steps_experienced = i
                break

        # determine episode duration and estimate remaining time
        current_time = time.perf_counter()
        rem_time = 0.0
        if start_time > 0:
            timed_episodes = ep - starting_episode + 1
            avg_duration = (current_time - start_time) / timed_episodes / 60.0 # minutes
            remaining_time_minutes = (starting_episode + max_episodes - ep - 1) * avg_duration
            rem_time = remaining_time_minutes / 60.0 # hours
            time_est_msg = "{:4.1f} hr rem".format(rem_time)
        else:
            avg_duration = 1.0 # avoids divide-by-zero
            time_est_msg = "???"

        # update score bookkeeping, report status
        scores.append(score)
        recent_scores.append(score)
        # don't compute avg score until several episodes have completed to avoid a meaningless
        # spike in the average near the very beginning
        avg_score = 0.0
        if ep > 30:
            avg_score = np.mean(recent_scores)
        max_recent = np.max(recent_scores)
        avg_scores.append(avg_score)
        mem_stats = mgr.get_erb_stats() #element 0 is total size, 1 is num good experiences
        mem_pct = 0.0
        if mem_stats[0] > 0:
            mem_pct = min(100.0*float(mem_stats[1])/mem_stats[0], 99.9)
        print("\r{}\tRunning avg/max: {:.3f}/{:.3f},  mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min   "
              .format(ep, avg_score, max_recent, mem_stats[0], mem_stats[1], mem_pct, 1.0/avg_duration), end="")
        
        # save a checkpoint at planned intervals and print summary performance stats
        if ep > 0  and  ep % chkpt_interval == 0:
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, ep)
            print("\r{}\tAverage score:   {:.3f},        mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min; {}   "
                  .format(ep, avg_score, mem_stats[0], mem_stats[1], mem_pct, 1.0/avg_duration, time_est_msg))

        # if sleeping is chosen, then pause for viewing after selected episodes
        if sleeping:
            if ep % 100 < 5:
                time.sleep(1) #allow time to view the Unity window

        # if we have met the winning criterion, save a checkpoint and terminate
        if ep >= AVG_SCORE_EXTENT  and  avg_score >= training_goal:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(ep, avg_score))
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, ep)
            print("\nMost recent individual episode scores:")
            for j, sc in enumerate(recent_scores):
                print("{:2d}: {:.2f}".format(j, sc))
            break

        # if this solution is clearly going nowhere, then abort early
        if ep > starting_episode + ABORT_EPISODE:
            hit_rate = float(mem_stats[1]) / ep
            if hit_rate < 0.01  or  (rem_time > 1.0  and  hit_rate < 0.05):
                print("\n* Aborting due to inadequate progress.")
                break

    print("\nAvg/max time steps/episode = {:.1f}/{:d}"
          .format(float(sum_steps)/float(max_episodes-starting_episode), max_steps_experienced))
    return (scores, avg_scores)

#------------------------------------------------------------------------------

"""Returns the current state vectors for all agents (a dict of ndarray[n, x] for each agent type)."""

def all_agent_states(env_info   : {},               # dict of unityagent.brain.BrainInfo for each brain
                     types      : {}                # dict of AgentType describing all agent types
                    ):

    all_states = {}
    for t in types:
        all_states[t] = env_info[t].vector_observations
    
    return all_states

#------------------------------------------------------------------------------

"""Returns a tuple of (rewards, dones), which are each a dict of lists, where each list represents an agent
   type, and each list entry represents a single agent of that type.
"""

def all_agent_results(env_info  : {},               # dict of unityagent.brain.BrainInfo for each brain
                      types     : {}                # dict of AgentType describing all agent types
                     ):
    
    rewards = {}
    dones = {}
    for t in types:
        rewards[t] = env_info[t].rewards
        dones[t]   = env_info[t].local_done
    
    return (rewards, dones)

#------------------------------------------------------------------------------

"""Determines if any agent has raised its 'done' flag.  Returns True if so, False otherwise."""

def any_dones(types : {},   # dict of AgentType describing all agent types
              dones : {}    # dict of lists of done flags for each agent of each type
             ):

    result = False
    for t in types:
        result |= any(dones[t])

    return result

#------------------------------------------------------------------------------

"""Finds the max reward value assigned to any agent and returns that value."""

def max_rewards(types: {},  # dict of AgentType describing all agent types
                rewards: {} # dict of lists of rewards for each agent of each type
               ):
    
    reward = -np.inf
    for t in types:
        mr = max(rewards[t])
        if mr > reward:
            reward = mr
    
    return reward

#------------------------------------------------------------------------------

"""Advances the agent models and the environment to the next time step, passing data
    between the two as needed. Note that states is both an input and output; this is
    necessary to preserve its value, even though the caller probably won't need it
    between calls.

    Returns: tuple of (states, rewards, dones) where each is a dict of agent types
"""

def advance_time_step(model         : AgentMgr,         # manager for all agetns and environment
                      env           : UnityEnvironment, # the environment object in which all action occurs
                      agent_types   : {},               # dict of AgentType
                      states        : {}                # dict of current states; each entry represents an agent type,
                                                        #   which is ndarray[num_agents, x]
                     ):

    # Predict the best actions for the current state and store them in a single ndarray
    actions = model.act(states) #returns dict of ndarray, with each entry having one item for each agent
    #print("\nactions = ", actions)

    # get the new state & reward based on this action
    ea = copy.deepcopy(actions) # disposable copy because env.step() changes the elements to lists!
    env_info = env.step(ea)
    next_states = all_agent_states(env_info, agent_types)
    rewards, dones = all_agent_results(env_info, agent_types)

    # update the agents with this new info
    model.step(states, actions, rewards, next_states, dones) 

    # roll over new state
    states = next_states

    return (states, rewards, dones)
