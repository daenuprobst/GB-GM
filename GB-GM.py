import time
import random
import pickle
import numpy as np
from rdkit.Chem import AllChem
from io import StringIO
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')

random.seed(22)
np.random.seed(22)


class Gene:
  def __init__(self, size_mean, size_std):
    # average_size, size_stdev = 23.2, 4.4
    self.size_mean = size_mean
    self.size_std = size_std

    self.p_ring = pickle.load(open('p_ring.p', 'rb'))
    self.rxn_smarts_ring_list = pickle.load(open('rs_ring.p', 'rb'))
    self.rxn_smarts_list = pickle.load(open('r_s1.p', 'rb'))
    self.p = pickle.load(open('p1.p', 'rb'))

    prob_double = 0.8
    self.p_make_ring = self.scale_p_ring(self.p_ring, prob_double)


  def run(self, file_name, smiles = 'CC'):
    with open(file_name, 'w') as f:
      count = 1
      while count <= 1000:
        mol = self.generate_mol(smiles, 50, self.size_mean, self.size_std)
        if mol:
          new_smiles = AllChem.MolToSmiles(mol)
          f.write(new_smiles + '\n')
          count += 1

  def valences_not_too_large(self, mol):
    valence_dict = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 16: 6, 17: 1, 35: 1, 53: 1}
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    valences = [valence_dict[atomic_num] for atomic_num in atomic_nums]
    adj = AllChem.GetAdjacencyMatrix(mol, useBO=True)

    n_bonds = adj.sum(axis=1)
    for valence, n in zip(valences, n_bonds):
      if n > valence:
        return False

    return True

  def run_rxn(self, rxn_smarts, mol):
    new_mol_list = []
    reactant = rxn_smarts.split('>>')[0]

    # work on a copy so an un-kekulized version is returned
    # if the molecule is not changed
    mol_copy = AllChem.Mol(mol)

    try:
      AllChem.Kekulize(mol_copy)
    except:
      pass

    if mol_copy.HasSubstructMatch(AllChem.MolFromSmarts(reactant)):
      rxn = AllChem.ReactionFromSmarts(rxn_smarts)
      new_mols = rxn.RunReactants((mol_copy,))

      for new_mol in new_mols:
        if AllChem.SanitizeMol(new_mol[0], catchErrors=True) == 0:
          new_mol_list.append(new_mol[0])
        else:
          pass
      if len(new_mol_list) > 0:
        new_mol = random.choice(new_mol_list) 
        return new_mol

    return mol

  def add_atom(self, mol):
    rxn_smarts_make_ring = pickle.load(open('rs_make_ring.p','rb'))

    # probability of adding ring atom
    if np.random.random() < 0.63:
      rxn_smarts = np.random.choice(self.rxn_smarts_ring_list, p=self.p_ring)
      smarts = AllChem.MolFromSmarts('[r3,r4,r5]')

      if not mol.HasSubstructMatch(smarts) or AllChem.CalcNumAliphaticRings(mol) == 0:
        rxn_smarts = np.random.choice(rxn_smarts_make_ring, p=self.p_make_ring)

        # probability of starting a fused ring
        if np.random.random() < 0.036:
          rxn_smarts = rxn_smarts.replace("!", "")
    else:
      if mol.HasSubstructMatch(AllChem.MolFromSmarts('[*]1=[*]-[*]=[*]-1')):
        rxn_smarts = '[r4:1][r4:2]>>[*:1]C[*:2]'
      else:
        rxn_smarts = np.random.choice(self.rxn_smarts_list, p=self.p)
      
    mol = self.run_rxn(rxn_smarts,mol)
    smiles = AllChem.MolToSmiles(mol)

    return mol, smiles

  def expand_small_rings(self, mol):  
    AllChem.Kekulize(mol, clearAromaticFlags=True)
    rxn_smarts = '[*;r3,r4;!R2:1][*;r3,r4:2]>>[*:1]C[*:2]'

    while mol.HasSubstructMatch(AllChem.MolFromSmarts('[r3,r4]=[r3,r4]')):
      mol = self.run_rxn(rxn_smarts,mol)

    return mol

  def generate_mol(self, smiles, max_atoms, average_size, size_stdev):
    mol = AllChem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
  
    target_size = size_stdev * np.random.randn() + average_size

    iteration = 0
    while num_atoms < max_atoms and iteration < max_atoms:
      iteration += 1
      mol, smiles = self.add_atom(mol)
      num_atoms = mol.GetNumAtoms()
      if num_atoms > target_size:
        break
    
    if self.valences_not_too_large(mol):   
      mol = self.expand_small_rings(mol)
      return mol
    else:
      return None

  def scale_p_ring(self, p_ring, new_prob_double):
    p_single = []
    p_double = []

    for smarts, p in zip(self.rxn_smarts_ring_list, p_ring):
      if '=' in smarts:
        p_double.append(p)
      else:
        p_single.append(p)
      
    prob_double = sum(p_double)
    prob_single = sum(p_single)
    scale_double = new_prob_double/prob_double
    scale_single = (1.0 - new_prob_double) / (1 - prob_double)

    for i, smarts in enumerate(self.rxn_smarts_ring_list):
      if '=' in smarts:
        p_ring[i] *= scale_double
      else:
        p_ring[i] *= scale_single
        
    print(scale_double, scale_single * prob_single, sum(p_ring))
    
    return p_ring


t0 = time.time()

gene = Gene(23.2, 4.4)
gene.run('grow1000.smi')


t1 = time.time()

print (t1 - t0)