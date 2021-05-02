# Performs the training of all agents of all types. For now, it assumes the environment is
# based on the Unity ML-Agents framework.

import numpy as np
import torch
import time
from collections    import deque

from agent_type     import AgentType

AVG_SCORE_EXTENT = 100 # number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" # can be empty string, but if a dir is named, needs trailing '/'
ABORT_EPISODE = 400 # num episodes after which training will abort if insignificant learning is detected


"""Trains a set of DRL agents in a Unity ML-Agents environment.

   Return: tuple of (list of episode scores (floats), list of running avg scores (floats))
"""

def train(mgr               : AgentMgr,        # manages all agents and their learning process
          env               : UnityEnvironment,# the envronment model that agents live in
          run_name          : string = "UNDEF",# tag for config control of learining session
          starting_episode  : int    = 0,      # episode to begin counting from (used if restarting
                                               #   from a checkpoint)
          max_episodes      : int    = 2,      # max num episodes to train if goal isn't met
          max_time_steps    : int    = 100,    # num time steps allowed in an episode
          sleeping          : bool   = False,  # should the code pause for a few sec after selected
                                               #   episodes? (allows for better visualizing)
          training_goal     : float  = 0.0,    # when avg score (over AVG_SCORE_EXTENT consecutive
                                               #   episodes) exceeds this value, training is done
          chkpt_interval    : int    = 100     # num episodes between checkpoints being stored
         ):

"""
JOHN TODO:
- Store states and actions here as a tuple with a [n, x] ndarray for each agent type. I don't
think we want all states/actions in a single array/tensor, since it requres zero padding.
- Future: Have AgentMgr manage the env also so that train() never has to see it?
If so, we can remove all brain references here.
"""

    # Initialize Unity simulation environment
    agent_types = {}
    states = {}
    actions = {}
    next_states = {}
    rewards = {}
    dones = {}
    for n in env.brain_names:

        # store the type info
        b = env.brains[n]
        info = env.reset(train_mode=True)[n]
        s = len(info.vector_observations[0])
        a = b.vector_action_space_size
        num_agents = len(info.agents)
        type = AgentType(n,b, s, a, num_agents)
        agent_types[n] = type

        # store initial states and define empty structures for the other performance variables of this type
        states[n] = info.vector_observations # gets the initial states for these agents
        actions[n] = np.ndarray((num_agents, a))
        next_states[n] = np.ndarray((num_agents, s))
        rewards[n] = []
        dones[n] = []



#...STOPPED WORKING HERE



    # collect raw & running avg scores at each episode
    scores = []
    avg_scores = []
    sum_steps = 0 #accumulates number of time steps exercised
    max_steps_experienced = 0
    recent_scores = deque(maxlen=AVG_SCORE_EXTENT)
    start_time = 0

    # run the simulation several time steps to prime the replay buffer
    print("Priming the replay buffer", end="")
    pc = 0
    while not mgr.is_learning_underway():
        states, actions, rewards, next_states, dones = \
          advance_time_step(mgr, env, states, actions, rewards, next_states, dones)
        if pc % 4000 == 0:
            print(".", end="")
        pc += 1
        if any(dones):
            env_info = env.reset(train_mode=True)
            states = env_info.vector_observations #returns ndarray(2, state_size)
    print("!\n")


    # loop on episodes for training
    start_time = time.perf_counter()
    for e in range(starting_episode, max_episodes):
        
        # Reset the enviroment & agents and get their initial states
        env_info = env.reset(train_mode=True)
        states = env_info.vector_observations #returns ndarray(2, state_size)
        score = 0 #total score for this episode
        mgr.reset()

        # loop over time steps
        for i in range(max_time_steps):

            # advance the MADDPG model and its environment by one time step
            states, actions, rewards, next_states, dones = \
              advance_time_step(mgr, env, states, actions, rewards, next_states, dones)

            # check for episode completion
            score += np.max(rewards) #use the highest reward from all agents
            if np.any(dones):
                sum_steps += i
                if i > max_steps_experienced:
                    max_steps_experienced = i
                break

        # determine episode duration and estimate remaining time
        current_time = time.perf_counter()
        rem_time = 0.0
        if start_time > 0:
            timed_episodes = e - starting_episode + 1
            avg_duration = (current_time - start_time) / timed_episodes / 60.0 #minutes
            remaining_time_minutes = (starting_episode + max_episodes - e - 1) * avg_duration
            rem_time = remaining_time_minutes / 60.0
            time_est_msg = "{:4.1f} hr rem".format(rem_time)
        else:
            avg_duration = 1.0 #avoids divide-by-zero
            time_est_msg = "???"

        # update score bookkeeping, report status
        scores.append(score)
        recent_scores.append(score)
        # don't compute avg score until several episodes have completed to avoid a meaningless
        # spike in the average near the very beginning
        avg_score = 0.0
        if e > 50:
            avg_score = np.mean(recent_scores)
        max_recent = np.max(recent_scores)
        avg_scores.append(avg_score)
        mem_stats = mgr.get_memory_stats() #element 0 is total size, 1 is num good experiences
        mem_pct = 0.0
        if mem_stats[0] > 0:
            mem_pct = min(100.0*float(mem_stats[1])/mem_stats[0], 99.9)
        print("\r{}\tRunning avg/max: {:.3f}/{:.3f},  mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min   "
              .format(e, avg_score, max_recent, mem_stats[0], mem_stats[1], mem_pct, 
                      1.0/avg_duration), end="")
        if e > 0  and  e % chkpt_interval == 0:
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, e)
            print("\r{}\tAverage score:   {:.3f},        mem: {:6d}/{:6d} ({:4.1f}%), avg {:.1f} eps/min; {}   "
                  .format(e, avg_score, mem_stats[0], mem_stats[1], mem_pct,
                          1.0/avg_duration, time_est_msg))

        # if sleeping is chosen, then pause for viewing after selected episodes
        if sleeping:
            if e % 100 < 20:
                time.sleep(1) #allow time to view the Unity window

        # if we have met the winning criterion, save a checkpoint and terminate
        if e > 100  and  avg_score >= training_goal:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}"
                  .format(e, avg_score))
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, e)
            print("\nMost recent individual episode scores:")
            for j, sc in enumerate(recent_scores):
                print("{:2d}: {:.2f}".format(j, sc))
            break

        # if this solution is clearly going nowhere, then abort early
        if e > starting_episode + ABORT_EPISODE:
            hit_rate = float(mem_stats[1]) / e
            if hit_rate < 0.01  or  (rem_time > 1.0  and  hit_rate < 0.05):
                print("\n* Aborting due to inadequate progress.")
                break

    print("\nAvg/max time steps/episode = {:.1f}/{:d}"
          .format(float(sum_steps)/float(max_episodes-starting_episode),
                  max_steps_experienced))
    return (scores, avg_scores)


def advance_time_step(model, env, states, actions, rewards, next_states, dones):
    """Advances the agents' model and the environment to the next time step, passing data
       between the two as needed.

       Params
           model (Maddpg):         the MADDPG model that manages all agents
           env (UnityEnvironment): the environment object in which all action occurs
           states (ndarray):       array of current states of all agents and environment [n, x]
           actions (ndarray):      array of actions by all agents [n, x]
           rewards (list):         list of rewards from all agents [n]
           next_states (ndarray):  array of next states (after action applied) [n, x]
           dones (list):           list of done flags (int, 1=done, 0=in work) [n]
       where, in each param, n is the number of agents and x is the number of items per agent.

       Returns: tuple of (s, a, r, s', done) values
    """

    # Predict the best actions for the current state and store them in a single ndarray
    actions = model.act(states) #returns ndarray, one row for each agent

    # get the new state & reward based on this action
    env_info = env.step(actions)
    next_states = env_info.vector_observations #returns ndarray, one row for each agent
    rewards = env_info.rewards #returns list of floats, one for each agent
    dones = env_info.local_done #returns list of bools, one for each agent

    # update the agents with this new info
    model.step(states, actions, rewards, next_states, dones) 

    # roll over new state
    states = next_states

    return (states, actions, rewards, next_states, dones)
