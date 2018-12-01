import sys
import time
import numpy as np
from rdkit.Chem import AllChem
from rdkit import rdBase
from mcts import Node, State, MCTS
from gene import Gene

rdBase.DisableLog('rdApp.error')

num_sims = 40 # 40 = 1000 logP evaluations

print('num_sims', num_sims)
print('max_children = 25\n')

results = []
size = []

t0 = time.time()

gene = Gene(39.15, 3.50, seed=22)

for i in range(10):
  State.max_score = (-99999.0, '')
  State.count = 0

  smiles = 'CC'
  mol = AllChem.MolFromSmiles(smiles)

  root = Node(State(gene, mol, smiles))
  current_node = MCTS.uct_search(num_sims, root)

  print(i, State.max_score[0], State.max_score[1], AllChem.MolFromSmiles(State.max_score[1]).GetNumAtoms())
  results.append(State.max_score[0])
  size.append(AllChem.MolFromSmiles(State.max_score[1]).GetNumAtoms())

t1 = time.time()

print('')
print('time, count ', t1 - t0, State.count)
print(max(results), np.array(results).mean(), np.array(results).std())
print(max(size), np.array(size).mean(), np.array(size).std())


