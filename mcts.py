# code modified from https://github.com/haroldsultan/MCTS/blob/master/mcts.py
import random
import math
import hashlib
import numpy as np

from rdkit.Chem import AllChem

SCALAR = 1 / math.sqrt(2.0)

class State():
  max_score = (-99999.0, '')
  count = 0

  def __init__(self, gene, mol=None, smiles='', turn=60, max_children=25):
    self.gene = gene
    self.mol = mol
    self.turn = turn
    self.smiles = smiles
    self.max_children = 25

  def next_state(self):
    smiles = self.smiles

    for _ in range(100):
      mol = self.gene.add_atom(self.mol)
      smiles = AllChem.MolToSmiles(mol)
      if smiles != self.smiles:
        break

    return State(self.gene, mol, smiles, self.turn - 1)

  def terminal(self):
    target_size = self.gene.size_std * np.random.randn() + self.gene.size_mean
    num_atoms = 0

    if self.mol != None:
      num_atoms = self.mol.GetNumAtoms()

    if self.turn == 0 or num_atoms > target_size:
      self.mol = self.gene.expand_small_rings(self.mol)
      self.smiles = AllChem.MolToSmiles(self.mol)
      return True
    
    return False

  def reward(self):
    State.count += 1
    
    logP = self.gene.logP_score(self.mol)
 
    if logP > State.max_score[0]:
      State.max_score = (logP, self.smiles)
      return 1.0
    
    return 0.0
 
  def __hash__(self):
    return int(hashlib.md5(str(self.smiles).encode('utf-8')).hexdigest(), 16)
  def __eq__(self, other):
    return hash(self) == hash(other)
  def __repr__(self):
    s = "Turn %s" % (self.turn)
    return s
	

class Node():
  def __init__(self, state, parent=None):
    self.visits = 1
    self.reward = 0.0
    self.state = state
    self.children = []
    self.parent = parent

  def add_child(self, child_state):
    self.children.append(Node(child_state, self))

  def update(self, reward):
    self.reward += reward
    self.visits += 1

  def fully_expanded(self):
    return len(self.children) == self.state.max_children

  def __repr__(self):
    s = str(self.state.smiles)
    return s
		

class MCTS:
  @staticmethod
  def uct_search(budget, root):
    for _ in range(int(budget)):
      front = MCTS.tree_policy(root)

      for child in front.children:
        reward = MCTS.default_policy(child.state)
        MCTS.backup(child, reward)

    return MCTS.best_child(root, 0)

  @staticmethod
  def tree_policy(node):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.fully_expanded():
      node = MCTS.best_child(node, SCALAR)
    
    if not node.state.terminal():
      node = MCTS.expand_all(node)

    return node

  @staticmethod 
  def expand_all(node):

    c = 0
    while not node.fully_expanded() and c < node.state.max_children:
      node = MCTS.expand(node)
      c += 1

    return node

  @staticmethod
  def expand(node):
    expanded_states = [c.state for c in node.children]
    new_state = node.state.next_state()

    c = 0
    while new_state in expanded_states and c < new_state.max_children:
      new_state = node.state.next_state()
      c += 1

    node.add_child(new_state)
    return node

  # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
  @staticmethod
  def best_child(node, scalar):
    bestscore = 0.0
    bestscore = -99.0
    bestchildren = []

    for child in node.children:
      exploit = child.reward / child.visits
      explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
      score = exploit + scalar * explore

      if score == bestscore:
        bestchildren.append(child)
        
      if score > bestscore:
        bestchildren = [child]
        bestscore = score

    if len(bestchildren) == 0:
      raise Exception('MCTS Error: Could not expand node.') 

    return random.choice(bestchildren)

  @staticmethod
  def default_policy(state):
    while not state.terminal():
      state = state.next_state()

    return state.reward()

  @staticmethod
  def backup(node, reward):
    while node != None:
      node.visits += 1
      node.reward += reward
      node = node.parent