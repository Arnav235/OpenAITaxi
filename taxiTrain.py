import gym
import numpy as np
import random

env = gym.make("Taxi-v2")

actionSize = env.action_space.n
stateSize = env.observation_space.n

qTable = np.zeros((stateSize,actionSize))

#hyperperameters
totalEpisodes = 50000
totalTestEpisodes = 100
maxSteps = 99
learningRate = 0.07
discount = 0.6

#exploration rate
epsilon = 1
maxEpsilon = 1
minEpsilon = 0.01
epsilonDecay = 0.001

for episode in range(totalEpisodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(maxSteps):
        expTradeoff = random.uniform(0,1)

        if expTradeoff > epsilon:
            action = np.argmax(qTable[state,:])
        else:
            action = env.action_space.sample()

        newState, reward, done, info = env.step(action)

        qTable[state, action] = qTable[state,action] + learningRate * (reward + discount * np.max(qTable[newState,:]) - qTable[state,action])

        state = newState

        if done == True:
            break
    
        epsilon = minEpsilon + (maxEpsilon - minEpsilon) *np.exp (-epsilonDecay*episode)

np.savetxt("taxiQtable-50000.txt", qTable)