# # 1. It renders instances for 500 timesteps, performing random actions.
# import gym
# env = gym.make('Acrobot-v1')
# env.reset()
# for _ in range(500):
#     env.render()
#     env.step(env.action_space.sample())
# # 2. To check all env available, uninstalled ones are also shown.
# from gym import envs
# print(envs.registry.items())

#
# import gym
# env = gym.make('MountainCarContinuous-v0') # try for different environments
# observation = env.reset()
# for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(observation, reward, done, info)
#         if done:
#             print("Finished after {} timesteps".format(t+1))
#             break


# import gym
# env = gym.make('CartPole-v0')
# print(env.action_space) #[Output: ] Discrete(2)
# print(env.observation_space) # [Output: ] Box(4,)
# env = gym.make('MountainCarContinuous-v0')
# print(env.action_space) #[Output: ] Box(1,)
# print(env.observation_space) #[Output: ] Box(2,)

import gym
import numpy as np
# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v1')
Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.observation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-learning
eta = .628
gma = .9
epis = 5000
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(epis):
    # Reset environment
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        env.render()
        j+=1
        # Choose action from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state & reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rev_list.append(rAll)
    env.render()
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)