### SETUP ###
from matplotlib import rc
from matplotlib import colors
rc('animation', html='jshtml')

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import abc
from numpy.typing import NDArray
import gymnasium as gym
from scipy import signal

VISON_SIZE = 5

fig = plt.figure(figsize=(6,6))


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

class Agent():
  __metaclass__ = abc.ABCMeta

  def __init__(self,
                 obs_onehot: bool = False,
                 attack_map: bool = False,
                 attack_map_logits: bool = False,
                 ):
        """
        :param obs_onehot: Determines whether the observation that is received is one-hot encoded or not
        :type obs_onehot: bool
        :param attack_map: Determines whether you pass in a probability map for each cell's attack
        (4 dimensional array for your total action)
        :type attack_map: bool
        :param attack_map_logits: If you pass in a map for each cell's attack, this determines whether it
        is a logit or probability map
        :type attack_map_logits: bool
        """
        self.obs_onehot = obs_onehot
        self.attack_map = attack_map
        self.attack_map_logits = attack_map_logits

  @abc.abstractmethod
  def policy(self, obs: NDArray, action_space, obs_space) -> NDArray:

      """
      :param obs: height*width array of game_state
      :action_space: unused - from AI gym in python project version
      :obs_space: unused - from AI gym in python project version
      """
      raise NotImplementedError("Please implement the policy method!")





      ### WRITTEN ENEMY AGENTS ###

class dummy_agent(Agent):
  def policy(self, state_mat, **kwargs):
    h,w = state_mat.shape
    attack_mat = np.random.randint(0,25,(h,w)) #get cell feats returns adjacent cells

    return attack_mat.astype(int)

class ReinforcementLearningAgent(Agent):
  
  
  def adjacent_cells(self, state_mat, x, y):
    """
    
    """
    w, h = state_mat.shape

    rowIndices, colIndices = np.indices((VISON_SIZE, VISON_SIZE))
    rowIndices = (rowIndices + x - (VISON_SIZE/2)) % h
    colIndices = (colIndices + y - (VISON_SIZE/2)) % w

    adj_cells = state_mat[rowIndices, colIndices]
    return adj_cells

  def policy(self, state_mat, **kwargs):
    pass

class first_enemy_agent(Agent):

  def simple_adj(self, feat_mat):
    feat_mat = feat_mat.reshape(-1,1)

    enemy_list = np.argwhere(feat_mat >= 3)[:,0]
    empty_list = np.argwhere(feat_mat == 0)[:,0]

    if len(enemy_list)>=1: #if there is at least one enemy
      return enemy_list[0]
    elif len(empty_list)>=1: #if there is at least one empty
      return empty_list[0]
    else:
      return 12
    
  #get the adjacent features for a given x,y cell
  def get_cell_feats(self,state_mat, x,y, n = 2):
    w,h = state_mat.shape
    adj_width = 2*n+1

    xind,yind = np.indices((adj_width,adj_width))
    xind = (xind -n +x)%h
    yind = (yind -n +y)%w

    #gets a matrix of the adjacent cells by 1
    adj_vals = state_mat[xind,yind]
    return adj_vals

  def policy(self, state_mat, **kwargs):
    attack_mat =np.zeros(state_mat.shape)

    for i in range(attack_mat.shape[0]):
      for j in range(attack_mat.shape[1]):
        attack_mat[i,j] = self.simple_adj(self.get_cell_feats(state_mat, i,j,2)) #get cell feats returns adjacent cells

    return attack_mat.astype(int)


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
    p1_gain = net_action>=1 #player 1 gains
    p2_gain = net_action<=-1 #player 2 gains

    #update game state
    self.state_mat[net_action>=1] = 2
    self.state_mat[net_action<=-1] = 3
    self.state_mat[self.map==1]=1 #walls always equal 1

    ### ADD PUNISHMENT FOR OVERPOPULATION #################################
    abs_map = (self.state_mat==2)+(self.state_mat==3) #get locations where there is an agent
    sum_mat = signal.convolve2d(abs_map, np.ones((3,3)), boundary = 'wrap', mode = 'same')
    excess_mat = sum_mat>self.over_pop
    self.state_mat = self.state_mat * (1-(1*excess_mat))
    ### ADD PUNISHMENT FOR OVERPOPULATION #################################

    self.state_mat[self.map==1]=1 #walls always equal 1

    ### ADD TO STATE HISTROY ######################
    self.state_history.append(self.state_mat.copy())
    self.p1_history.append(p1_action)
    self.p2_history.append(p2_action)


    #####reward! modify as needed (your objective is to win)##############
    reward_kernal = np.array([[0   , .25 ,.5 , .25,  0],
                       [.25 , .5  ,.75, .5 ,.25],
                       [.5  , .75 , 1 , .75, .5],
                       [.25 , .5  ,.75, .5 ,.25],
                       [0   , .25 ,.5 , .25,  0]])
    reward = signal.convolve2d(self.state_mat, reward_kernal, boundary = 'wrap', mode = 'same')
    #####reward! modify as needed (your objective is to win)

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
  


### TEST GAME ###
agent_1 = first_enemy_agent()
agent_2 = dummy_agent()

#walls = np.random.binomial(1,0.2, (10,10))

test_game = cgol(agent_1, agent_2,10,10)

game_hist = test_game.run_game(30)

anim_builder(game_hist)
