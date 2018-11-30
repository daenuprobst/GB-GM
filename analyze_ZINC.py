import sys
import pickle
import re

import numpy as np

from operator import itemgetter
from collections import OrderedDict
from rdkit.Chem import AllChem

def read_file(file_name):
  smiles_list = []
  with open(file_name,'r') as file:
    for line in file:
      smiles_list.append(line.split()[0])

  return smiles_list

def get_probs(smarts_list, smiles, ring=False):
  probs = OrderedDict()

  for smarts in smarts_list:
    probs[smarts] = 0

  for s in smiles:
    mol = AllChem.MolFromSmiles(s)
    AllChem.Kekulize(mol)

    for smarts in smarts_list:
      matches = mol.GetSubstructMatches(AllChem.MolFromSmarts(smarts), uniquify=ring)  
      probs[smarts] += len(matches)
  
  probs = OrderedDict((k, v) for k, v in probs.items() if v > 0)
  total = sum(probs.values())
  
  return total, probs

def clean_probs(probs):
  exceptions = ['[#7]#','[#8]=','[#9]','[#17]','[#35]','[#53]']  
  probs = OrderedDict((k, v) for k, v in probs.items() if not any(substring in k for substring in exceptions))
  total = sum(probs.values())

  return total, probs

def get_p(probs):
  return np.fromiter(probs.values(), dtype=int) / sum(probs.values())

def get_rxn_smarts_make_rings(probs):
  X = {'[#6R': 'X4', '[#7R': 'X3'}
  rxn_smarts = []

  for key in probs:
    tokens = key.split(']')
    smarts = ''

    if '=' in key:
      smarts += tokens[0][:-1] + X[tokens[0]] + ';!R:1]'
    else:
      smarts += tokens[0][:-1] + ';!R:1]=,'

    smarts += tokens[2][:-1] + ';!R:2]>>'
    smarts += '[*:1]1' + tokens[1] + '][*:2]1'

    rxn_smarts.append(smarts)
    
  return rxn_smarts

def get_rxn_smarts_rings(probs):
  X = {'[#6R': 'X4', '[#7R': 'X3'}
  rxn_smarts = []

  for key in probs:
    tokens = key.split(']')

    smarts = ''
    if '=' in key:
      smarts += tokens[0] + X[tokens[0]] + ';!r6;!r7;!R2:1]'
    else:
      smarts += tokens[0] + ';!r6;!r7;!R2:1]'

    smarts += tokens[2] + ';!r6;!r7:2]>>'
    smarts += '[*:1]' + tokens[1] + '][*:2]'

    rxn_smarts.append(smarts)
    
  return rxn_smarts

def get_rxn_smarts(probs):
  rxn_smarts = []

  for key in probs:
    smarts = ''
    tokens = key.split(']')
    smarts = tokens[0]

    if '-' in key and '#16' not in smarts:  # key <-> smarts
      smarts += ';!H0:1]>>[*:1]'
    if '=' in key and '#16' not in smarts:  # key <-> smarts
      smarts += ';!H1;!H0:1]>>[*:1]'
    if ']#[' in key:
      smarts += ';H3:1]>>[*:1]'
    if '#16' in smarts:  # key <-> smarts
      smarts += ':1]>>[*:1]'
      
    smarts += tokens[-2] + ']'
    rxn_smarts.append(smarts)
    
  return rxn_smarts

def get_mean_size(smiles):
  size = []

  for s in smiles: 
    size.append(AllChem.MolFromSmiles(s).GetNumAtoms())
 
  return np.mean(size), np.std(size)

def get_macro_cycle_probs(smiles, smarts):
  probs = OrderedDict()

  for sm in smarts:
    probs[sm] = 0
  
  tot = 0
  for s in smiles:
    for sm in smarts:
      mol = AllChem.MolFromSmiles(s)
      AllChem.Kekulize(mol)
      matches = mol.GetSubstructMatches(AllChem.MolFromSmarts(sm), uniquify=True)
      if len(matches) > 0:
        probs[sm] += 1
        tot += 1
      
  return tot, probs

def print_props(smiles_list):
  smarts = ['[*]','[R]','[!R]','[R2]']
  mean_size, size_stdv = get_mean_size(smiles_list)
  print('mean number of non-H atoms' + str(mean_size) + '+/-' + str(size_stdv) + '\n')


  _ ,probs = get_probs(smarts, smiles_list)
  print('Probability of ring atoms', float(probs['[R]']) / probs['[*]'])
  print('Probability of non-ring atoms', float(probs['[!R]']) / probs['[*]'])
  print('Probability of fused-ring atoms', float(probs['[R2]']) / probs['[*]'])
  print('')


  smarts = ['[R]~[R]~[R]','[R]-[R]-[R]','[R]=[R]-[R]']

  _, probs = get_probs(smarts, smiles_list, ring=True)

  print('Probability of [R]-[R]-[R]', float(probs['[R]-[R]-[R]']) / probs['[R]~[R]~[R]'])
  print('Probability of [R]=[R]-[R]', float(probs['[R]=[R]-[R]']) / probs['[R]~[R]~[R]'])
  print('')


def pickle_smarts_probs_rings(smiles_list):
  elements = ['#5','#6','#7','#8','#9','#14','#15','#16','#17','#35','#53']

  smarts = []
  for element in elements:
    smarts.append('[' + element + 'R]')

  _, probs_Ratoms = get_probs(smarts, smiles_list)

  R_elements = []

  for key in probs_Ratoms:
    R_elements.append(key)

  smarts = []

  for i, e1 in enumerate(R_elements):
    for e2 in R_elements:
      for j, e3 in enumerate(R_elements):
        if j >= i:
          sm_s = e1 + '-' + e2 + '-' + e3
          if sm_s not in smarts:
            smarts.append(sm_s)
        sm_d = e1 + '=' + e2 + '-' + e3
        if sm_d not in smarts:
          smarts.append(sm_d)

  tot, probs = get_probs(smarts,smiles_list,ring=True)
  props_sorted = sorted(probs.items(), key=itemgetter(1), reverse=True)

  for i in range(len(props_sorted)):
    print (props_sorted[i][0], props_sorted[i][1] / tot)

  print('')

  rxn_smarts_rings = get_rxn_smarts_rings(probs)
  rxn_smarts_make_rings = get_rxn_smarts_make_rings(probs)
  p_rings = get_p(probs)

  pickle.dump(p_rings, open('p_ring.p', 'wb')) 
  pickle.dump(rxn_smarts_rings, open('rs_ring.p', 'wb'))
  pickle.dump(rxn_smarts_make_rings, open('rs_make_ring.p', 'wb'))



def pickle_smarts_probs(smiles_list):
  elements = ['#5','#6','#7','#8','#9','#14','#15','#16','#17','#35','#53']
  bonds = ['-','=','#']

  smarts = []
  for bond in bonds:
    for element1 in elements:
      for element2 in elements:
        smarts.append('[' + element1 + ']' + bond + '[' + element2 + ';!R]')

  _, probs = get_probs(smarts,smiles_list)
  tot, probs = clean_probs(probs)

  p = get_p(probs)

  sorted_props = sorted(probs.items(), key=itemgetter(1), reverse=True)

  for i in range(len(sorted_props)):
    print(sorted_props[i][0],sorted_props[i][1]/tot)

  rxn_smarts = get_rxn_smarts(probs)

  pickle.dump(p, open('p1.p', 'wb')) 
  pickle.dump(rxn_smarts, open('r_s1.p', 'wb')) 



def print_rings(smiles_list):
  smarts_list = ['[*]1-[*]-[*]-1','[*]1-[*]=[*]-1','[*]1-[*]-[*]-[*]-1','[*]1=[*]-[*]-[*]-1','[*]1=[*]-[*]=[*]-1',
    '[*]1-[*]-[*]-[*]-[*]-1','[*]1=[*]-[*]-[*]-[*]-1','[*]1=[*]-[*]=[*]-[*]-1',
    '[*]1-[*]-[*]-[*]-[*]-[*]-1','[*]1=[*]-[*]-[*]-[*]-[*]-1','[*]1=[*]-[*]=[*]-[*]-[*]-1',
    '[*]1=[*]-[*]-[*]=[*]-[*]-1','[*]1=[*]-[*]=[*]-[*]=[*]-1']

  smarts_macro = ['[r;!r3;!r4;!r5;!r6;!r8;!r9;!r10;!r11;!r12]','[r;!r3;!r4;!r5;!r6;!r7;!r9;!r10;!r11;!r12]',
    '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r10;!r11;!r12]','[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r11;!r12]',
    '[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r12]','[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r11]']

  total, probs = get_probs(smarts_list, smiles_list, ring=True)
  t, p = get_macro_cycle_probs(smiles_list, smarts_macro)

  total += t
  probs.update(p)

  print('')
  for key in probs:
    print(key, probs[key])

  print('')
  print('number of rings', sum(probs.values()))


file_name = 'ZINC_first_1000.smi'

if (len(sys.argv) > 1):
  file_name = sys.argv[1]

smiles_list = read_file(file_name)

print_props(smiles_list)
pickle_smarts_probs_rings(smiles_list)
pickle_smarts_probs(smiles_list)
print_rings(smiles_list)
