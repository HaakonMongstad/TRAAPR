import numpy as np
from numpy.typing import NDArray
import abc

VISION_SIZE = 5

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
    Return the adjacent cells in a 5x5
    matrix around a cell located at (x, y)
    """
    w, h = state_mat.shape

    rowIndices, colIndices = np.indices((VISION_SIZE, VISION_SIZE))
    rowIndices = (rowIndices + x - (VISION_SIZE/2)) % h
    colIndices = (colIndices + y - (VISION_SIZE/2)) % w

    vision_mat = state_mat[rowIndices, colIndices]
    return vision_mat
  
  def action_space(self, feat_mat):
    """
    Return the indices of valid actions (i.e attacks)
    The only cells that cannot be attacked are walls
    """
    feat_mat = feat_mat.reshape(-1, 1)
    return np.argwhere(feat_mat != 1) # Anywhere that is not a wall (1)

  def policy(self, state_mat, **kwargs):
    return BoltzmannQPolicy()

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