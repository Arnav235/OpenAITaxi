import gym
import numpy as np
import random
import time

env = gym.make("Taxi-v2")

qTable = np.loadtxt('taxiQtable-50000.txt', dtype=float)

#hyperperameters
totalTestEpisodes = 100
maxSteps = 99

rewards = []

for episode in range(totalTestEpisodes):
    state = env.reset()
    done = False
    totalReward = 0

    while True:
        env.render()
        action = np.argmax(qTable[state,:])
        newState, reward, done, info = env.step(action)
        totalReward += reward

        if done == True:
            rewards.append(totalReward)
            print ("Score",totalReward)
            break

        state = newState

env.close()

print ("sum of scores", sum(rewards))
print ("Score over time")
print (sum(rewards) / totalTestEpisodes)