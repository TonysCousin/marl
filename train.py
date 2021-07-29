# Performs the training of all agents of all types. For now, it assumes the environment is
# based on the Unity ML-Agents framework.

import numpy as np
import time
import copy
import sys
import time
from collections    import deque
from unityagents    import UnityEnvironment

from agent_type     import AgentType
from agent_models   import AgentModels
from model          import GoalieActor, GoalieCritic, StrikerActor, StrikerCritic
from agent_mgr      import AgentMgr

AVG_SCORE_EXTENT = 100 # number of episodes over which running average scores are computed
CHECKPOINT_PATH = "checkpoint/" # can be empty string, but if a dir is named, needs trailing '/'
ABORT_EPISODE = 300 # num episodes after which training will abort if insignificant learning is detected
PRIME_FEEDBACK_INTERVAL = 2000 # num time steps between visual feedback of priming progress


#----------------------------------------------------------------------

"""Constructs the neural network(s) for the AI model."""

def build_model(actor_lr              : float,
                critic_lr             : float,
                actor_nn_l1           : int,
                actor_nn_l2           : int,
                critic_nn_l1          : int,
                critic_nn_l2          : int
               ):

    # TODO: this info should be gotten from the AgentTypes objects

    # define NNs specifically for the soccer game (could add one extra action for each agent type, which is "do nothing";
    #   this hack works with this environment, but doing so requires changing code in AgentMgr.__init__ to not get the
    #   number of actions from the environment directly)
    goalie_states = 336
    goalie_actions = 4
    num_goalie_agents = 2

    striker_states = 336
    striker_actions = 6
    num_striker_agents = 2

    total_states = num_goalie_agents*goalie_states + num_striker_agents*striker_states
    total_actions = num_goalie_agents*goalie_actions + num_striker_agents*striker_actions
    agent_models = AgentModels()

    agent_models.add_actor_critic("GoalieBrain", GoalieActor(goalie_states, goalie_actions, fc1_units=actor_nn_l1, fc2_units=actor_nn_l2), 
                                    GoalieCritic(total_states, total_actions, fcs1_units=critic_nn_l1, fc2_units=critic_nn_l2), 
                                    actor_lr, critic_lr)

    agent_models.add_actor_critic("StrikerBrain", StrikerActor(striker_states, striker_actions, fc1_units=actor_nn_l1, fc2_units=actor_nn_l2),
                                    StrikerCritic(total_states, total_actions, fcs1_units=critic_nn_l1, fc2_units=critic_nn_l2),
                                    actor_lr, critic_lr)
    
    return agent_models

#----------------------------------------------------------------------

"""Defines the model to be trained and executes a brand-new training plan from scratch
    for the specified duration.
"""

def build_and_train_model(env                   : UnityEnvironment,
                          name                  : str,
                          use_coaching          : bool,
                          batch                 : int,
                          prime                 : int,
                          learn_every           : int,
                          seed                  : int,
                          goal                  : float,
                          start_episode         : int,
                          episodes              : int,
                          chkpt_every           : int,
                          init_time_steps       : int,
                          incr_time_steps_every : int,
                          final_time_steps      : int,
                          bad_step_prob         : float,
                          use_noise             : bool,
                          noise_init            : float,
                          noise_decay           : float,
                          actor_lr              : float,
                          critic_lr             : float,
                          actor_nn_l1           : int,
                          actor_nn_l2           : int,
                          critic_nn_l1          : int,
                          critic_nn_l2          : int,
                          update_factor         : float
                         ):

    # build the NN models for the agents, using randomly initialized params
    agent_models = build_model(actor_lr, critic_lr, actor_nn_l1, actor_nn_l2, critic_nn_l1, critic_nn_l2)

    # define the PRNG that is to be used for this whole system
    prng = np.random.default_rng(seed)

    # create the agent manager
    mgr = AgentMgr(env, agent_models, batch_size=batch, buffer_prime=prime, learn_every=learn_every, bad_step_prob=bad_step_prob, random_seed=seed,
                    use_noise=use_noise, noise_init=noise_init, noise_decay=noise_decay, update_factor=update_factor, prng=prng)

    # if the starting episode is non-zero, then use the agent manager to restore the NN models from an appropriate checkpoint
    if start_episode > 0:
        mgr.restore_checkpoint(CHECKPOINT_PATH, name, start_episode)

    #print("Haltus")

    train(mgr, env, run_name=name, starting_episode=start_episode, max_episodes=episodes, chkpt_interval=chkpt_every, training_goal=goal,
          init_time_steps=init_time_steps, incr_time_steps_every=incr_time_steps_every, final_time_steps=final_time_steps, prng=prng,
          use_coaching=use_coaching)

#----------------------------------------------------------------------

"""Trains a set of DRL agents in a Unity ML-Agents environment.

   Return: tuple of (list of episode scores (floats), list of running avg scores (floats))
"""

def train(mgr               : AgentMgr,         # manages all agents and their learning process
          env               : UnityEnvironment, # the envronment model that agents live in
          run_name          : str = "UNDEF",    # tag for config control of learining session
          starting_episode  : int    = 0,       # episode to begin counting from (used if restarting
                                                #   from a checkpoint)
          max_episodes      : int    = 2,       # max num episodes to train if goal isn't met
          init_time_steps   : int    = 100,     # max num time steps in the first episode
          final_time_steps  : int    = 500,     # max num time steps ever allowed in an episode
          incr_time_steps_every: int = 1,       # num episodes between increment of max time steps allowed
          sleeping          : bool   = False,   # should the code pause for a few sec after selected
                                                #   episodes? (allows for better visualizing)
          training_goal     : float  = 0.0,     # when avg score (over AVG_SCORE_EXTENT consecutive
                                                #   episodes) exceeds this value, training is done
          chkpt_interval    : int    = 100,     # num episodes between checkpoints being stored
          prng              : np.random.Generator = None, # random number generator
          use_coaching      : bool   = False    # should we coach the players with modified actions & rewards?
         ):

    #TODO Future: Have AgentMgr manage the env also so that train() never has to see it?

    # Initialize Unity simulation environment
    states = {}
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

    timestamp = time.strftime("%a %H:%M")
    print("!\n\n\n///// Training underway at {}".format(timestamp))
    sys.stdout.flush()

    # loop on episodes for training
    start_time = time.perf_counter()
    for ep in range(starting_episode, max_episodes):
        
        # Reset the enviroment & agents and get their initial states
        env_info = env.reset(train_mode=True)
        states = all_agent_states(env_info, agent_types)
        mgr.reset()
        score = 0 # total score for this episode

        #print("\n\n/////Begin episode ", ep, "/////\n")

        # set the time step limit for this episode
        max_time_steps = min(init_time_steps + ep//incr_time_steps_every, final_time_steps)

        # loop over time steps
        for i in range(max_time_steps):

            #print("\n\nTime step ", i, ". states =\n", states)

            # advance the MADDPG model and its environment by one time step
            states, rewards, dones = time_step(mgr, env, use_coaching, prng, agent_types, states)

            # add this step's reward to the episode score
            score += max_rewards(agent_types, rewards)

            # if the episode is complete, update the record time steps in an episode and stop the time step loop
            if any_dones(agent_types, dones):
                sum_steps += i
                if i > max_steps_experienced:
                    max_steps_experienced = i
                break
        
        # invoke the learning algorithm on each desired agent
        mgr.learn(






            

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
        
        # generate a timestamp for reporting purposes
        timestamp = time.strftime("%a %H:%M")

        # update score bookkeeping, report status
        scores.append(score)
        recent_scores.append(score)
        # don't compute avg score until several episodes have completed to avoid a meaningless
        # spike in the average near the very beginning
        avg_score = 0.0
        if ep - starting_episode > 30:
            avg_score = np.mean(recent_scores)
        max_recent = np.max(recent_scores)
        avg_scores.append(avg_score)
        mem_stats = mgr.get_erb_stats() #element 0 is total size, 1 is num good experiences
        mem_pct = 0.0
        if mem_stats[0] > 0:
            mem_pct = min(100.0*float(mem_stats[1])/mem_stats[0], 99.9)
        print("\r{} {}\tRunning avg/max: {:.3f}/{:.3f},  buf: {:6d}/{:6d} ({:4.1f}%), avg {:.2f} eps/min   "
              .format(timestamp, ep, avg_score, max_recent, mem_stats[0], mem_stats[1], mem_pct, 1.0/avg_duration), end="")
        
        # save a checkpoint at planned intervals and print summary performance stats
        if ep > 0  and  ep % chkpt_interval == 0:
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, ep)
            print("\r{} {}\tAverage score:   {:.3f},        buf: {:6d}/{:6d} ({:4.1f}%), avg {:.2f} eps/min; {}   "
                  .format(timestamp, ep, avg_score, mem_stats[0], mem_stats[1], mem_pct, 1.0/avg_duration, time_est_msg))

        # if sleeping is chosen, then pause for viewing after selected episodes
        if sleeping:
            if ep % 100 < 5:
                time.sleep(1) #allow time to view the Unity window

        # if we have met the winning criterion, save a checkpoint and terminate
        if ep - starting_episode >= AVG_SCORE_EXTENT  and  avg_score >= training_goal:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(ep, avg_score))
            mgr.save_checkpoint(CHECKPOINT_PATH, run_name, ep)
            print("\nMost recent individual episode scores:")
            for j, sc in enumerate(recent_scores):
                print("{:2d}: {:.2f}".format(j, sc))
            break

        # if this solution is clearly going nowhere, then abort early
        if ep > starting_episode + ABORT_EPISODE:
            hit_rate = float(mem_stats[1]) / ep
            if hit_rate < 0.025  or  (rem_time > 2.0  and  ep > starting_episode + 2*ABORT_EPISODE  and   hit_rate < 0.06):
                print("\n* Aborting due to inadequate progress.")
                print("Final noise level = {:6.4f}".format(mgr.get_noise_level()))
                break
        
        sys.stdout.flush()

    # in games that normally terminate with a done flag, it makes sense to see how quickly that flag
    # is raised; this doesn't make sense in games where the episode normally times out (hits max steps)
    #print("\nAvg/max time steps/episode = {:.1f}/{:d}"
    #      .format(float(sum_steps)/float(max_episodes-starting_episode), max_steps_experienced))
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

    for t in types:
        if any(dones[t]):
            return True

    return False

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

#TODO: make this a callback supplied by the game-specific code

"""Modifies the actions beyond what the model provides (i.e. coaching guidance), in order to
    accelerate learning. Early experience shows that learning finds a big local maximum
    in the rewards, achieved by strikers standing around doing nothing, thus their
    goalies don't get scored against. This function encourages strikers to move toward
    the ball in order to keep it in motion.  If the striker doesn't see the ball, it
    is encouraged to move backwards (it can't see behind); if it sees the ball to the right
    it is encouraged to move right; if it sees the ball to the left, it is encouraged to
    move left; and if it sees the ball in front it is encouranged to move forward.
    Encouragement is in the form of an action override "most of the time".

    Return: tuple of (updated actions dict, bool indicating whether updates were done)
"""

def modify_actions(prng         : np.random.Generator, # random number generator
                   types        : {},   # dict of AgentType describing all agent types
                   actions      : {},   # dict of arrays of all possible action scores for each agent of each type
                   states       : {}    # dict of current states for each agent type; each entry is
                                        # ndarray[n, x], where n is number of agents of that type
                  ):

    mods_done = False
    
    # if the prng has not been defined then we can't do anything here
    if prng == None:
        return (actions, mods_done)

    # observe the final 112 elements of the state vector, which is the current time step
    start = 2*112

    # define how often coaching will be injected
    MOST_OF_TIME = 0.6

    # if it is randomly selected then
    if prng.random() < MOST_OF_TIME:

        # set up a mapping of desired actions vs observations
        ball_forward_action = {"GoalieBrain": 1, "StrikerBrain": 0} #goalie back, striker forward
        ball_left_action    = {"GoalieBrain": 3, "StrikerBrain": 4} #everybody moves left
        ball_right_action   = {"GoalieBrain": 2, "StrikerBrain": 5} #everybody moves right
        ball_unknown_action = {"GoalieBrain": 1, "StrikerBrain": 1} #everybody back up
        
        # loop through each agent
        for t in types:
            for agent in range(states[t].shape[0]):

                # get each of the 14 ray traces from the current time step & see if first element indicates it sees the ball
                is_ball = np.empty(14, dtype=bool)
                for ray in range(14):
                    is_ball[ray] = states[t][agent, start + 8*ray] > 0.5 #first element in each 8-element ray indicates it sees the ball

                # find the current largest vector element and its value                
                am = np.argmax(actions[t][agent])
                max_val = actions[t][agent, am]

                # if it sees the ball to the left
                if is_ball[4]  or  is_ball[11]  or  is_ball[3]  or  is_ball[10]:
                    actions[t][agent, ball_left_action[t]] = max_val + 0.0001 #this is now the largest value in the vector
                
                # if it sees the ball to the right
                elif is_ball[0]  or  is_ball[7]  or  is_ball[8]  or  is_ball[1]:
                    actions[t][agent, ball_right_action[t]] = max_val + 0.0001 #this is now the largest value in the vector
                
                # if it sees the ball in front
                elif is_ball[12]  or  is_ball[13]  or  is_ball[2]  or  is_ball[5]  or is_ball[6]  or  is_ball[9]:
                    actions[t][agent, ball_forward_action[t]] = max_val + 0.0001 #this is now the largest value in the vector

                # else we don't know where the ball is
                else:
                    actions[t][agent, ball_unknown_action[t]] = max_val + 0.0001 #this is now the largest value in the vector
        
        mods_done = True

    return (actions, mods_done)

#------------------------------------------------------------------------------

#TODO: make this a callback supplied by the game-specific code

"""Modifies the rewards beyond what the Unity environment provides, in order to
    accelerate learning. Early experience shows that learning finds a big local maximum
    in the rewards, achieved by strikers standing around doing nothing, thus their
    goalies don't get scored against. This function encourages strikers to move toward
    the ball in order to keep it in action, by giving a slight positive reward for
    minimizing distance to the ball.

    Return: updated rewards dict.
"""

def modify_rewards(types        : {},   # dict of AgentType describing all agent types
                   rewards      : {},   # dict of lists of rewards for each agent of each type
                   states       : {}    # dict of latest states for each agent type; each entry is
                                        # ndarray[n, x], where n is number of agents of that type
                  ):
    
    # observe the final 112 elements of the state vector, which is the current time step
    start = 2*112
            
    # look for the strikers only
    for t in types:
        if t == "StrikerBrain":

            for agent in range(states[t].shape[0]):

                # look for any ray trace that observes the ball, which is element 0 of the ray, and average their distance
                count = 0
                sum = 0.0
                ray_state = np.empty(8)
                for ray in range(14):

                    # only look at the rays that are toward the front of the striker
                    if ray == 2  or  ray == 5  or  ray == 6  or  ray == 9  or  ray == 12  or  ray == 13:
                        ray_state = states[t][agent, start + 8*ray : start + 8*ray + 8]
                        if ray_state[0] > 0.99: #one-hot vector element indicating it sees the ball
                            count += 1
                            sum += ray_state[7]

                # figure out the distance to the ball, and award bonus points if it sees it and it's close
                bonus = 0.0
                if count > 0:
                    distance = max(sum / count, 0.02) #0.02 is closest distance ever observed
                
                    # add a small reward if the distance to ball is close
                    bonus = 0.001 * (0.05 / distance)
                
                rewards[t][agent] += bonus
    
    return rewards

#------------------------------------------------------------------------------

def debug_actions(types, actions, states, flag):
        
    print("\n-----  entering learning_time_step:")
    for t in types:
        for agent in range(states[t].shape[0]):

            # get each of the 14 ray traces from the current time step & see if first element indicates it sees the ball
            start = 2*112
            is_ball = np.empty(14, dtype=bool)
            for ray in range(14):
                is_ball[ray] = states[t][agent, start + 8*ray] > 0.5 #first element in each 8-element ray indicates it sees the ball
            
            # if it sees the ball to the left
            if is_ball[4]  or  is_ball[11]  or  is_ball[3]  or  is_ball[10]:
                print("{}\t{}: Ball left\tAction {}{}".format(t, agent, actions[t][agent], flag))
            
            # if it sees the ball to the right
            elif is_ball[0]  or  is_ball[7]  or  is_ball[8]  or  is_ball[1]:
                print("{}\t{}: Ball right\tAction {}{}".format(t, agent, actions[t][agent], flag))
            
            # if it sees the ball in front
            elif is_ball[12]  or  is_ball[13]  or  is_ball[2]  or  is_ball[5]  or is_ball[6]  or  is_ball[9]:
                print("{}\t{}: Ball fwd\tAction {}{}".format(t, agent, actions[t][agent], flag))

            # else we don't know where the ball is
            else:
                print("{}\t{}: Ball unknown\tAction {}{}".format(t, agent, actions[t][agent], flag))
            
    print(" ")


#------------------------------------------------------------------------------

"""Advances the agent models and the environment to the next time step, passing data
    between the two as needed for a learning iteration. Note that states is both
    an input and output; this is necessary to preserve its value, even though the
    caller probably won't need it between calls.

    Returns: tuple of (states, rewards, dones) where each is a dict of agent types
"""

def time_step(model         : AgentMgr,         # manager for all agetns and environment
              env           : UnityEnvironment, # the environment object in which all action occurs
              use_coaching  : bool,             # will we invoke coaching to modify actions & rewards?
              prng          : np.random.Generator,# random number generator
              agent_types   : {},               # dict of AgentType
              states        : {}                # dict of current states; each entry represents an agent type,
                                                #   which is ndarray[num_agents, x]
             ):

    # Predict the best actions for the current state and store them in a single ndarray
    av = model.get_raw_action_vector(states) #returns dict of ndarray, with each entry having a row for each agent (scores for all possible actions)
    coaching_flag = " "
    if use_coaching:
        actions, was_coached = modify_actions(prng, agent_types, av, states)
        if was_coached:
            coaching_flag = "*"
    actions = model.find_best_action(av)
    #debug_actions(agent_types, actions, states, coaching_flag) #takes in actions as integer values (one per agent)

    # get the new state & reward based on this action
    ea = copy.deepcopy(actions) # disposable copy because env.step() changes the elements to lists!
    env_info = env.step(ea)
    next_states = all_agent_states(env_info, agent_types)
    rewards, dones = all_agent_results(env_info, agent_types)

    # allow modification of the rewards based on state values
    if use_coaching:
        rewards = modify_rewards(agent_types, rewards, next_states)

    # update the agents with this new info (need to pass raw action vector)
    model.step(states, av, rewards, next_states, dones) 

    # roll over new state
    states = next_states

    return (states, rewards, dones)

#------------------------------------------------------------------------------

"""Advances the agent models and the environment to the next time step, passing data
    between the two as needed for simple inference. Note that states is both
    an input and output; this is necessary to preserve its value, even though the
    caller probably won't need it between calls.

    Returns: tuple of (states, rewards, dones) where each is a dict of agent types
"""

def inference_time_step(model         : AgentMgr,         # manager for all agetns and environment
                        env           : UnityEnvironment, # the environment object in which all action occurs
                        agent_types   : {},               # dict of AgentType
                        states        : {}                # dict of current states; each entry represents an agent type,
                                                          #   which is ndarray[num_agents, x]
                       ):

    # Predict the best actions for the current state and store them in a single ndarray
    actions = model.act(states, is_inference=True) #returns dict of ndarray, with each entry having one item for each agent

    # get the new state & reward based on this action
    ea = copy.deepcopy(actions) # disposable copy because env.step() changes the elements to lists!
    env_info = env.step(ea)
    next_states = all_agent_states(env_info, agent_types)
    rewards, dones = all_agent_results(env_info, agent_types)

    # roll over new state
    states = next_states

    return (states, rewards, dones)
