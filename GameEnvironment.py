import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from scipy import signal

FRIENDLY = 2
ENEMY = 3






class GameEnv(py_environment.PyEnvironment):
  def __init__(self, agent_1, agent_2, board_h=10, board_w=10, attack_width = 5, map = None, maxCount = 100):
    self.board_h = board_h
    self.board_w = board_w
    self.view_width = attack_width
    self.view_height = attack_width
    self.maxCount = maxCount
    self.attack_width = attack_width

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
    
    
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(board_h,board_w), dtype=np.int32, minimum=-1, maximum=1, name='play')

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(board_h,board_w), dtype=np.int32, minimum=0, maximum=3, name='board')

    self._episode_ended = False


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  
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

  def getAction(self):

    p1_view = self.modify_view(self.state_mat, 2) #player 1 shows up as 2
    p2_view = self.modify_view(self.state_mat, 3) #player 2 shows up as 3

    #retrieve agent 1 actions
    
    p1_action = self.agent_1.policy(p1_view)
    
    #retrieve agent 2 actions
  
    p2_action = self.agent_2.policy(p2_view)

    ##retrieve adjent coordinates
    p1_cords = np.where(p1_view==2) #where p1 sees allys
    p2_cords = np.where(p2_view==2) #where p2 sees allys

    #translate adjent actions into sum attack matrices
    p1_action = self.convert_action_map(p1_action, p1_cords)
    p2_action = self.convert_action_map(p2_action, p2_cords)

    net_action = p1_action - p2_action #change in map from actions

    net_action[net_action>0] = 1
    net_action[net_action<0] = -1
    return net_action

  def step(self,action):
    return self._step(action)

  def reset(self):
    return self._reset()


  def _step(self,action):
    
    if self._episode_ended == True:
        return self._reset()
     

    p1_gain = action>=1 #player 1 gains
    p2_gain = action<=-1 #player 2 gains

    #update game state
    self.state_mat[action>=1] = 2
    self.state_mat[action<=-1] = 3
    self.state_mat[self.map==1]=1 #walls always equal 1

    abs_map = (self.state_mat==2)+(self.state_mat==3) #get locations where there is an agent
    sum_mat = signal.convolve2d(abs_map, np.ones((3,3)), boundary = 'wrap', mode = 'same')
    excess_mat = sum_mat>self.over_pop
    self.state_mat = self.state_mat * (1-(1*excess_mat))
    # TODO: Add overpopulation punishment

    self.state_mat[self.map==1]=1 #walls always equal 1

    # Add to state history
    self.state_history.append(self.state_mat.copy())
    # self.p1_history.append(p1_action)
    # self.p2_history.append(p2_action)

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
  

    rewardMatrix = signal.convolve2d(agent_mat, reward_kernal, boundary = 'wrap', mode = 'same')

    reward = np.sum(rewardMatrix)
    done = False
    self.count+=1
    # print("HERE")
    # print(reward)
    if np.min(agent_mat) == 0:
      self.count = self.maxCount
      done = True
      reward += 10000
      return ts.termination(np.array([self.state_mat],dtype=np.int32),reward, 1)
    elif np.max(agent_mat) == 0:
      self.count = self.maxCount
      done = True
      reward += -10000
      return ts.termination(np.array([self.state_mat],dtype=np.int32),reward, 1)
    elif self.count >= self.maxCount:
      done = True
      count2 = self.state_mat.bincount(2)
      count3 = self.state_mat.bincount(3)
      if (count2 > count3):
        reward += 5000
      elif (count3 > count2):
        reward -= 5000
      return ts.termination(np.array([self.state_mat],dtype=np.int32),reward, 1)

    return ts.transition(np.array([self.state_mat],dtype=np.int32),reward, 1)
    
    # if return_state:
    #   return self.state_mat.copy()



  def _reset(self):
    self = GameEnv(self.agent_1,self.agent_2,self.board_h,self.board_w,self.attack_width,self.map,self.maxCount)
    return ts.restart(np.array([self.state_mat], dtype=np.int32))


  # class notAbstractGameEnv():
  #   def step(self,action):
  #     return self._step(action)
  #   def reset(self):
  #     return self._reset()
  #   def getAction(self):
  #     return self._getAction