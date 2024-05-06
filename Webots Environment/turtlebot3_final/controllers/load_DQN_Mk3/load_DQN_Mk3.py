import gymnasium as gym
import sys
from stable_baselines3 import DQN
import numpy as np
import time
import os

# webots project folder and the controller used for environment simulation
sys.path.append(r'F:/Webots/turtlebot3_final')
#from controllers.DQN_Mk1.DQN_Mk1 import Turtlebot3_RL
#from controllers.DQN_Mk2.DQN_Mk2 import Turtlebot3_RL
from controllers.DQN_Mk3.DQN_Mk3 import Turtlebot3_RL
# the extra imports here are for testing the Mk1 model in Mk2 or Mk3 scenarios without the specific training for it

# path to the collection of models trained for the project
#model_path = 'F:/Webots/SIX_MODELS/validation models/DQN3_5500000.zip' # alternative model
model_path = 'F:/Webots/SIX_MODELS/validation models/DQN3_9300000.zip' # alternative model

# path to saving the run data
folder_path = 'F:/Webots/SIX_MODELS/run_data_sim'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# for data saving
max_steps = 5001

env = Turtlebot3_RL()
env.reset()
model = DQN.load(model_path, env=env)

episodes = 1
for e in range(episodes):
    # swap out for different setup S5, ..., S7
    filename = 'DQN3.S5.SIM.' + str(int(time.time())) + '.txt'
    file_path = os.path.join(folder_path, filename)
    data = np.zeros((max_steps,1))
    
    r_sum = 0
    i = 0
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        data[i] = reward
        r_sum += reward
        i += 1
    print(i-1, '\t', r_sum) # one tab forward/backward to get sum after every timestep or after every full run
    np.savetxt(file_path, data[:-1])
    
env.simulationSetMode(env.SIMULATION_MODE_PAUSE)