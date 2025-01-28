import gymnasium as gym
import numpy as np
import time


# create environment
env=gym.make('CartPole-v1',render_mode='human')
# reset the environment,
# returns an initial state
(state,_)=env.reset()
# states are
# cart position, cart velocity
# pole angle, pole angular velocity


# render the environment
env.render()
# close the environment
#env.close()

# push cart in one direction
env.step(1)

# observation space limits
print(env.observation_space)

# upper limit
print(env.observation_space.shape)
# print(env.observation_space.high)
#
# # lower limit
# print(env.observation_space.low)


# action space
print(env.action_space)

# all the specs
print(env.spec)

# maximum number of steps per episode
print(env.spec.max_episode_steps)

# reward threshold per episode
print(env.spec.reward_threshold)

# simulate the environment
episodeNumber=100
timeSteps=1000


for episodeIndex in range(episodeNumber):
    initial_state=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        observation, reward, terminated, truncated, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.1)
        if terminated:
            time.sleep(1)
            break
env.close()