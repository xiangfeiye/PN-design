from rdkit.Chem import Descriptors
import math
import gzip
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from collections import Counter

import pickle

def checkgroups(group_smarts, mol=None, smile=None):
    if mol is None:
        mol = Chem.MolFromSmiles(smile)
    patt = Chem.MolFromSmarts(group_smarts)
    try:
        flag = mol.HasSubstructMatch(patt)
    except:
        print('failed to find patt')
    if flag:
        atomids = mol.GetSubstructMatches(patt)
        visited = set()
        for a in atomids:
            visited.update(set(a))
        
        return len(atomids), atomids
    else:
        # print('分子中不包含该基团')
        return False, None

def get_groups(smiles, groups_patt=None):
    if groups_patt is None:
        groups_patt = {
        'aryalkynyl': 'C#Cc',
        'alkynyl': 'C#C',
        'cyanate': 'OC#N',
        'PN': '[#7]#[#6]cc[#6]#[#7]',
        'cyano': 'C#N',        
        'epoxy': 'C1CO1',
        'BCB': 'c1CCc1',
        'maleimide': 'N1C(=O)C=CC1=O',
        'nadimide': 'O=C1[N]C(=O)C2C3C=CC(C3)C12',
        'benzoxazine': 'c1OC[#7]Cc1',
        'SiHN': '[SiH].[NH,NH2]',
        'norbornene': 'C12CCC(C2)C=C1'
        }
    type_list = []
    mol = Chem.MolFromSmiles(smiles)
    visited = set()
    for patt in groups_patt:
        flag, atom_ids = checkgroups(groups_patt[patt], mol=mol)
        if flag:
            for a in atom_ids:
                if set(a) <= visited:
                    flag -= 1
                visited.update(set(a))
            type_list.append(patt)
        else:
            pass
            # type_list.append(0)   
    return type_list

class smiles_split(object):
    def __init__(self):
        self.replace_dict = {
                    'Si': 'V',
                    "Cl": 'U',
                    "Br": 'K',
                    "@@": 'D'
                }
    def split(self, smiles):
        for key in self.replace_dict.keys():
            smiles = smiles.replace(key, self.replace_dict[key])
        new_smile = []
        for code in smiles:
            new_smile.append(code)
        return new_smile
    
def get_list(split, smiles):
    split_list = split.split(smiles)
    split_list += get_groups(smiles)
    return split_list

def get_new_smiles(smiles, splitor=None, dsrate=10, loss=5, gas='dq', mn=None):
    if splitor is None:
        splitor = smiles_split()
    if mn is None:
        mn = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    split_list = get_list(splitor, smiles)
    smiles_len = len(split_list)
    dsrate_ = [s for s in str(int(dsrate))]
    loss_ = [s for s in str(int(loss))]
    mn_ = [s for s in str(int(mn))]
    split_list = split_list + ['dsrate'] +  dsrate_ +  ['loss']  + loss_ + ['gas'] + [gas] +  ['mn'] + mn_
    
    record = ''
    for s in split_list:
        record = record + s + ' '
        
    return record[:-1], split_list, smiles_len

# 清理中括号，标注中括号本身的序号以及影响的原子序号
def check_braces_atoms(split_smiles):
    smiles_list = split_smiles
    stack = []
    brackets = []

    for i, char in enumerate(smiles_list):
        if char == "[":
            stack.append(i)
        elif char == "]":
            start = stack.pop()
            brackets.append([start, i+1])

    return brackets

def get_braces_atoms(split_smiles, atom_maps, contribution_dict=None):
    if contribution_dict is None:
        contribution_dict = {}
    braces = check_braces_atoms(split_smiles)
    clear_idx = []
    for brace in braces:
        atoms = []
        for i in range(brace[0], brace[1]):
            if i in atom_maps:
                atoms.append(i)
        for i in range(brace[0], brace[1]):
            if i in atoms:
                pass
            contribution_dict[i] = atoms
            clear_idx.append(i)
    return contribution_dict, clear_idx

# 清理括号
def check_brackets_atoms(split_smiles):
    smiles_list = split_smiles
    stack = []
    brackets = []

    for i, char in enumerate(smiles_list):
        if char == "(":
            stack.append(i)
        elif char == ")":
            start = stack.pop()
            brackets.append([start, i])

    return brackets

def get_brackets_atoms(split_smiles, atom_maps, contribution_dict=None):
    if contribution_dict is None:
        contribution_dict = {}
    brackets = check_brackets_atoms(split_smiles)
    clear_idx = []
    for bracket in brackets:
        atoms = []
        for i in range(bracket[0], bracket[1]):
            if i in atom_maps:
                atoms.append(i)
        contribution_dict[bracket[0]] = atoms
        contribution_dict[bracket[1]] = atoms
        clear_idx += bracket
    return contribution_dict, clear_idx

# 清理单键、双键、三键、顺反异构
def get_bonds_atoms(smiles_list, i, char_list, element_list=None):
    if element_list is None:
        element_list = ['c', 'C', 'O', 'n', 'N', 'F', 'U', 'V', 'S', 's', 'o', 'K', 'I', 'P', 'V', 'B']
    if smiles_list[i+1] in element_list:
        atom2_index = i+1  
    elif smiles_list[i+1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '%']:
        for char in char_list:
            if len(char) > 2:
                if i+1 == char[3]:
                     atom2_index = char[0]-1 
            else:
                if i+1 == char[1]:
                     atom2_index = char[0]-1 
    elif smiles_list[i+1] == '[':
        j = i+2
        while not smiles_list[j] in element_list:
            j = j + 1
        atom2_index = j
    else:
        print('error found for get bonds atom2 in', smiles_list[i:i+2])
        print(smiles_list)
        return None

    assert smiles_list[atom2_index] in element_list
    if smiles_list[i-1] in element_list:
        return [i-1, atom2_index]
    else:
        atom1_index = i-1
        while not smiles_list[atom1_index] in element_list:
            if smiles_list[atom1_index] == ')':
                brackets = check_brackets_atoms(smiles_list)
                for bracket in brackets:
                    if atom1_index == bracket[1]:
                        atom1_index = bracket[0]
                        break
            atom1_index -= 1
    return [atom1_index, atom2_index]

def check_bonds_atoms(split_smiles, char_list, contribution_dict=None, unvisited=None):
    if contribution_dict is None:
        contribution_dict = {}
    smiles_list = split_smiles
    stack = []
    visited = []

    for i in (unvisited):
        char = split_smiles[i]
        if char in ["-", "=", "#", '/', '\\']:
            tmp = get_bonds_atoms(smiles_list, i, char_list)
            if tmp is not None:
                contribution_dict[i] = tmp
                visited.append(i)
            else:
                return None

    return contribution_dict, visited

# 清理数字代表的环结构
def find_mol_rings(smiles, atom_maps):
    mol = Chem.MolFromSmiles(smiles)
    assert len(atom_maps) == len(mol.GetAtoms())
    ring_atoms = []     # 存储每个环中的原子
    ring_count = 0      # 记录环的数量
    for ring in mol.GetRingInfo().AtomRings():
        ring_count += 1
        ring_ = [atom_maps[atom] for atom in ring]
        ring_.sort()
        ring_atoms.append(ring_)
        ring_atoms.sort()
    return ring_atoms

def get_smiles_ring(split_smiles):
    current = 0
    smiles_list = split_smiles
    number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index_stack = []
    flag_stack = []
    rings_indexs = []

    for i, char in enumerate(smiles_list):
        if i < current:
            continue
        if i < len(smiles_list)-1 and smiles_list[i+1] == ']':
            continue
            
        flag = None
        if char == "%":
            ring_index = [i, i+1, i+2]
            flag = smiles_list[i:i+3]
            current = i+3
        elif char in number_list:
            ring_index = [i]
            flag = smiles_list[i:i+1]
            current = i+1
        else:
            current = i+1

        if flag is not None:
            if flag in flag_stack:
                tmp = flag_stack.index(flag)
                index = index_stack[tmp]
                rings_indexs.append(list(index)+list(ring_index))

                flag_stack = flag_stack[:tmp] + flag_stack[tmp+1:]
                index_stack = index_stack[:tmp] + index_stack[tmp+1:]
            else:
                flag_stack.append(flag)
                index_stack.append(ring_index)
        rings_indexs.sort()
    return rings_indexs

def check_ring_atoms(smiles, split_smiles, atom_maps, contribution_dict=None, un_visited=None):
    visited = []
    if contribution_dict is None:
        contribution_dict = {} 
    char_list = get_smiles_ring(split_smiles)
    ring_info = find_mol_rings(smiles, atom_maps=atom_maps)
    # assert len(char_list) == len(ring_info)
    
    if len(char_list) == len(ring_info):
        for i in range(len(char_list)):
            char_pair = char_list[i]
            for char in char_pair:
                visited.append(char)
                if char in contribution_dict:
                    contribution_dict[char] += ring_info[i]
                else:
                    contribution_dict[char] = ring_info[i]
    elif len(char_list) < len(ring_info):
        for i in range(len(char_list)):
            char_pair = char_list[i]
            visited += char_pair
            for j in range(len(ring_info)):
                if ring_info[j][0] <char_pair[0] < ring_info[j][1] and ring_info[j][-1] < char_pair[-1]:
                    for char in char_pair:
                        if char in contribution_dict:
                            contribution_dict[char] += ring_info[i]
                        else:
                            contribution_dict[char] = ring_info[i]
                else:
                    pass
                
    return contribution_dict, visited, char_list

def get_params(split_smiles, params_list=['dsrate', 'loss', 'gas', 'mn'], contribution_dict=None):
    if contribution_dict is None:
        contribution_dict = {}
    tmp_indexs = [split_smiles.index(param) for param in params_list]
    tmp_indexs.append(999)
    visited = []
    params_dict = {}
    for i in range(len(tmp_indexs)-1):
        params_dict[tmp_indexs[i]] = [j for j in range(tmp_indexs[i], min(tmp_indexs[i+1], len(split_smiles)))]
    visited = [i for i in range(tmp_indexs[0],len(split_smiles))]
    
    for param in params_dict:
        sources_ = params_dict[param]
        for char in sources_:
            if char in contribution_dict:
                contribution_dict[char] += param
            else:
                contribution_dict[char] = [param]
    
    return contribution_dict, visited

def get_resin_type(split_smiles, contribution_dict=None, resin_list=['aryalkynyl', 'alkynyl', 'cyanate', 'PN', 'cyano', 'epoxy', 'BCB', 'maleimide', 'nadimide', 'benzoxazine', 'SiHN', 'norbornene']):
    if contribution_dict is None:
        contribution_dict = {}
    visited = []
    for resin in resin_list:
        if resin in split_smiles:
            tmp = split_smiles.index(resin)
            if tmp in contribution_dict:
                contribution_dict[tmp] += tmp
            else:
                contribution_dict[tmp] = [tmp]
            visited.append(tmp)
    
    return contribution_dict, visited


# get map from mol atoms to tokens
    
def check_element(split_smiles, element_list=None):
    if element_list is None:
        element_list = ['c', 'C', 'O', 'n', 'N', 'F', 'U', 'V', 'S', 's', 'o', 'K', 'I', 'P', 'V', 'B']
    mask = np.zeros(len(split_smiles))
    for i in range(len(split_smiles)):
        if split_smiles[i] in element_list:
            mask[i] = 1
        else:
            mask[i] = 0
    return mask

def get_atom_map(split_smiles):
    mask = check_element(split_smiles)
    atom_maps = []
    for i in range(len(mask)):
        if mask[i] == 1:
            atom_maps.append(i) 
    
    return mask, atom_maps

def analyze_smiles_data(smiles, splitor, dsrate=10, loss=5, gas='dq', mn=None):
    if splitor is None:
        splitor = smiles_split()
    new_smiles, split_smiles, smiles_len = get_new_smiles(smiles, splitor=splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)
    
    contribution_dict = {}
    unvisited = set([i for i in range(len(split_smiles))])
    # 检查加工参数对应的indexs
    contribution_dict, visited = get_params(split_smiles,['dsrate', 'loss', 'gas', 'mn'], contribution_dict)
    unvisited = unvisited - set(visited)

    # 检查树脂类型对应的indexs
    contribution_dict, visited = get_resin_type(split_smiles, contribution_dict)
    unvisited = unvisited - set(visited)

    # 检查真实原子对应的indexs
    mask, atom_maps = get_atom_map(split_smiles)
    for atom in atom_maps:
        if atom in contribution_dict:
            print('error for atom check ')
        else:
            contribution_dict[atom] = [atom]
    unvisited = unvisited - set(atom_maps)
    
    # 检查中括号涉及到的原子
    contribution_dict, visited = get_braces_atoms(split_smiles, atom_maps, contribution_dict)
    unvisited = unvisited - set(visited)

    # 检查代表环的数字涉及的原子
    contribution_dict, visited, char_list = check_ring_atoms(smiles, split_smiles[:smiles_len], atom_maps, contribution_dict)
    unvisited = unvisited - set(visited)

    # 检查单键、双键、三键、（顺反异构）符号涉及的原子
    contribution_dict, visited = check_bonds_atoms(split_smiles, char_list, contribution_dict, unvisited)
    unvisited = unvisited - set(visited)

    # 检查括号涉及到的原子
    contribution_dict, visited = get_brackets_atoms(split_smiles, atom_maps,contribution_dict)
    unvisited = unvisited - set(visited)
    
    return contribution_dict, unvisited, atom_maps


def acount_mol(captum_result, contribution_dict, atom_maps):
    tmp_result = {}
    result = {}
    atom_result = []
    for i in range(len(contribution_dict)):
        tmp_result[i] = captum_result[i]
        result[i] = 0
    for i in range(len(contribution_dict)):
        for j in contribution_dict[i]:
            result[j] += (tmp_result[i]/len(contribution_dict[i]))
            
    for atom in atom_maps:
        atom_result.append(result[atom])
    
    return result, atom_result

if __name__ == '__main__':
    df = pd.read_csv('test_total_td5_5_type2_emb_320.csv')

    gastype = {'N2': 'dq',
            'air': 'air',
            'Ar': 'ar'}
    splitor = smiles_split()
    result = {}
    indexs = []
    error_ = []
    for i in range(len(df)):
        smiles = df['smiles'][i]
        dsrate = int(df['dsrate'][i])
        loss = int(df['loss'][i])
        gas = gastype[df['gas'][i]]
        mn = int(df['mn'][i])
        try:
            contribution_dict, unvisited, atom_maps = analyze_smiles_data(smiles, splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)
            assert len(unvisited) == 0
            indexs.append(i)
            result[i] = {'dict': contribution_dict, 'unvisited': unvisited, 'atom_maps': atom_maps}
        except:
            print(i)
            error_.append(i)

    result['index'] = indexs
    result['df'] = df
    with open('analysis.plk', 'wb') as f:
        pickle.dump(result, f)

    # i = 0
    # smiles = df['smiles'][i]
    # dsrate = int(df['dsrate'][i])
    # loss = int(df['loss'][i])
    # gas = gastype[df['gas'][i]]
    # mn = int(df['mn'][i])

    # new_smiles, split_smiles, smiles_len = get_new_smiles(smiles, splitor=splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)

    # contribution_dict, unvisited, atom_maps = analyze_smiles_data(smiles, splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)
    # captum_result = np.ones(200)
    # acount_mol(captum_result, contribution_dict, atom_maps)
