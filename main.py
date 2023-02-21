from matplotlib import rc
from matplotlib import colors
rc('animation', html='jshtml')

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy import signal

import Agents
import GameEnvironment

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
#import Learning

fig = plt.figure(figsize=(6,6))


FRIENDLY = 2
ENEMY = 3

def frame(w, args):
    ax = args
    ax.clear()
    
    plot_list = []

    # [empty, wall, friendly, enemy]
    cmap = colors.ListedColormap(['white','black', 'red','blue'])
    bounds=[-0.5,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for i in w:
      plot_list.append(ax.imshow(w, cmap = cmap, norm = norm))
    return plot_list

#frames - list of map matrices [-1] enemy, [0] neutral, [1] friendly
def anim_builder(frames):
  ax = fig.add_subplot(111)
  anim = animation.FuncAnimation(fig, frame, frames=frames, fargs = [ax], blit=True, repeat=True)
  plt.show()
  return anim

class cgol:
  def  __init__(self, agent_1, agent_2, board_h = 10, board_w = 10, attack_width = 5, map = None):

    self.board_h = board_h
    self.board_w = board_w
    self.view_width = attack_width
    self.view_height = attack_width
    if map is None:
      self.map = np.zeros((self.board_h,self.board_w))
    else:
      self.map = map

    self.over_pop =6

    self.state_mat = np.zeros((self.board_h,self.board_w)) + self.map

    self.agent_1 = agent_1
    self.agent_2 = agent_2

    start_num = 1
    #set starting locations for agent

    self.state_mat[2, 2] = 2
    self.state_mat[5, 5] = 3
    
    #mapping for action int to coordinate 
    x_view, y_view = np.indices((5,5))
    x_view-=2
    y_view-=2
    self.x_view = x_view.flatten()
    self.y_view = y_view.flatten()

    #recording history
    self.state_history = []
    self.p1_history = []
    self.p2_history = []
    self.reward = []
    self.count = 0

    self.state_history.append(self.state_mat)


  def get_state(self):
    return self.state_mat

  #translates individual cell action from int to cordinate
  def translate_action(self, x_cord, y_cord, action_int):
    action_x = self.x_view[action_int]
    action_y = self.y_view[action_int]

    x_add = (x_cord + action_x)%self.board_h
    y_add = (y_cord + action_y)%self.board_w

    return np.array([x_add,y_add])


  #moves from cords to attack to attack_map
  #action_map - map of board with actions assigned integer based on coord
  #ex:
  # [1 3 4 5 0 9]      [sum of adjent attacks]
  # [3 4 9 8 6 2]      [                     ]
  # [1 0 6 7 5 3] ---> [                     ]
  # [0 0 0 4 5 6]      [                     ]
  def convert_action_map(self, action_map, self_cords):
    
    x_cord = self_cords[0]
    y_cord = self_cords[1]

    if len(y_cord) > 0: #check agent has at least one element
      action_list = [action_map[x_cord[i],y_cord[i]] for i in range(len(x_cord))]
      action_cords = np.stack([self.translate_action(x_cord[i], y_cord[i], action_list[i]) for i in range(len(action_list))])

    else: #case where there are no more agents on the field
      return np.zeros((self.board_h, self.board_w))

    action_count = np.zeros((self.board_h, self.board_w))
    action_count[action_cords[:,0], action_cords[:,1]]+=1
    return action_count

  #allys are always 2, so modify it here. 
  def modify_view(self, state_mat, agent_num = 2):
    view = state_mat.copy()

    view[state_mat==agent_num] = 2
    view[state_mat==2] = agent_num
    return view

  #action_mat_1 - the action being played for this step by agent_1
  #action_mat_2 - the action being played for this step by agent_2
  #returns - reward, done
  def step(self, action_mat_1 = None, action_mat_2 = None, return_state = False):

    if self.count == 0:
      self.count+=1
      if return_state:
        return self.state_mat.copy()

    p1_view = self.modify_view(self.state_mat, 2) #player 1 shows up as 2
    p2_view = self.modify_view(self.state_mat, 3) #player 2 shows up as 3

    #retrieve agent 1 actions
    if action_mat_1 is not None:
      p1_action = action_mat_1 #here agent 1's action replaced by input action
    else:
      p1_action = self.agent_1.policy(p1_view)
    
    #retrieve agent 2 actions
    if action_mat_2 is not None:
      p2_action = action_mat_2
    else:
      p2_action = self.agent_2.policy(p2_view)

    ##retrieve adjent coordinates
    p1_cords = np.where(p1_view==2) #where p1 sees allys
    p2_cords = np.where(p2_view==2) #where p2 sees allys

    #translate adjent actions into sum attack matrices
    p1_action = self.convert_action_map(p1_action, p1_cords)
    p2_action = self.convert_action_map(p2_action, p2_cords)

    net_action = p1_action - p2_action #change in map from actions

    # WHERE OUR MODEL STEP FUNCTION BEGINS

    p1_gain = net_action>=1 #player 1 gains
    p2_gain = net_action<=-1 #player 2 gains

    #update game state
    self.state_mat[net_action>=1] = 2
    self.state_mat[net_action<=-1] = 3
    self.state_mat[self.map==1]=1 #walls always equal 1

    abs_map = (self.state_mat==2)+(self.state_mat==3) #get locations where there is an agent
    sum_mat = signal.convolve2d(abs_map, np.ones((3,3)), boundary = 'wrap', mode = 'same')
    excess_mat = sum_mat>self.over_pop
    self.state_mat = self.state_mat * (1-(1*excess_mat))
    # TODO: Add overpopulation punishment

    self.state_mat[self.map==1]=1 #walls always equal 1

    # Add to state history
    self.state_history.append(self.state_mat.copy())
    self.p1_history.append(p1_action)
    self.p2_history.append(p2_action)

    agent_mat = self.state_mat.copy()
    agent_mat[agent_mat == 1] = 0
    agent_mat[agent_mat == FRIENDLY] = 1
    agent_mat[agent_mat == ENEMY] = -1
    # TODO: Update reward kernal
    """
    reward_kernal = np.array([
    [0   , .25 ,.5 , .25,  0],
    [.25 , .5  ,.75, .5 ,.25],
    [.5  , .75 , 1 , .75, .5],
    [.25 , .5  ,.75, .5 ,.25],
    [0   , .25 ,.5 , .25,  0]
    ])
    """
    reward_kernal = np.array([
    [.75, .75 ,.75 ,.75, .75],
    [.75, .5,  .5 , .5,  .75],
    [.75, .5 ,  1,  .5 , .75],
    [.75, .5  ,.5 , .5 , .75],
    [.75, .75, .75, .75, .75]
    ])
  

    reward = signal.convolve2d(agent_mat, reward_kernal, boundary = 'wrap', mode = 'same')

    rewardSum = np.sum(reward)
    done = False
    # print("HERE")
    # print(reward)
    if np.min(self.state_mat) == 0:
      done = True
      #reward = 10000
    if np.max(self.state_mat) == 0:
      done = True
      #reward = -100
    if self.count >= 100:
      done = True

    self.count+=1

    if return_state:
      return self.state_mat.copy()

    return reward, done


  #returns a history of the game
  def run_game(self, n_steps = 10):
    state_mats = [self.step(return_state = True) for i in range(n_steps)]
    return state_mats
  
  


def main():

  agent_1 = Agents.first_enemy_agent()
  agent_2 = Agents.dummy_agent()

  environment = GameEnvironment.GameEnv(agent_1=agent_1,agent_2=agent_2)
  tf_env = tf_py_environment.TFPyEnvironment(environment)
  time_step = tf_env.reset()

  # History
  rewards = []
  steps = []
  episodes = 1000
  action = environment.getAction()
  rewards.append(environment.step(action).reward)
  print(tf_env.step(action))

  # for _ in range(episodes):
  #   reward_t = 0
  #   steps_t = 0
  #   tf_env.reset()
  #   while True:
  #     action = environment.getAction()
  #     environment.step(action)
  #     next_time_step = tf_env.step(action)
  #     if tf_env.current_time_step().is_last():
  #       break
  #     episode_steps += 1
  #     episode_reward += next_time_step.reward.numpy()
  #   rewards.append(episode_reward)
  #   steps.append(episode_steps)
  
  print(rewards)


    

  
  """
  #walls = np.random.binomial(1,0.2, (10,10))

  test_game = cgol(agent_1, agent_2, 10,10)

  game_hist = test_game.run_game(30)

  anim_builder(game_hist)
"""


if __name__ == "__main__":
  main()