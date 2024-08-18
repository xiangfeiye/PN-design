from trainer_2 import *
from data import Vocab
from draw import *
from tqdm import trange
import captum
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from utils import *
# from rdkit.Chem.Draw import SimilarityMaps

from rdkit import Chem

from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import rdkit
print(rdkit.__version__)

import pickle

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol


def checkgroups(group_smarts, mol=None, smile=None):
    if mol is None:
        mol = Chem.MolFromSmiles(smile)
    mol = sanitize(mol, kekulize=False)
    patt = Chem.MolFromSmarts(group_smarts)
    try:
        flag = mol.HasSubstructMatch(patt)
    except:
        print('failed to find patt')
    if flag:
        atomids = mol.GetSubstructMatches(patt)

        return True, atomids
    else:
        # print('分子中不包含该基团')
        return False, None


with open('group.pkl', 'rb') as f:
    groups = pickle.load(f)


def get_result(smiles, groups, contribs=None):
    if contribs is None:
        _, _, contribs, _, _ = get_map_smiles(smiles)
        # contribution_dict, cp, contribs, result, captum_result
        # result, atom_result = acount_mol(captum_result, contribution_dict, atom_maps)

    mol = Chem.MolFromSmiles(smiles)
    result = {}
    for group in groups:
        flag, atomids = checkgroups(group, mol)
        if flag:
            result[group] = []
            for atoms in atomids:
                attributions = 0
                for atom in atoms:
                    attributions += contribs[atom]
                result[group].append(attributions)
    return result

if __name__ == "__main__":

    # # 必须定义
    # gastype = {'N2': 'dq',
    #         'air': 'air',
    #         'Ar': 'ar'}
    # splitor = smiles_split()
    #
    # device = torch.device('cuda:4')
    # model = torch.load('LSTMModelPre.ep27').to(device)
    # model.train()
    # vocab = torch.load('vocab.pt')
    # seq_length = model.max_len
    # PAD_IND = vocab.stoi['pad']
    # token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    # lig = LayerIntegratedGradients(model, model.encoder)
    #
    # def interpret_sentence(model, source, seq_length, device):
    #     model.zero_grad()
    #     source = source.to(device)
    #     pre = model(source)
    #     reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    #     attributions_ig, delta = lig.attribute(source, reference_indices,
    #                                            n_steps=128, return_convergence_delta=True)
    #
    #     attributions_ig = attributions_ig.sum(dim=2).squeeze(0)
    #     #     attributions_ig = attributions_ig / torch.norm(attributions_ig)
    #     attributions_ig = attributions_ig.cpu().detach().numpy()
    #
    #     return attributions_ig, delta
    #
    #
    # def get_map_smiles(smiles, dsrate=10, loss=5, gas='N2', mn=None):
    #     dsrate = int(dsrate)
    #     loss = int(loss)
    #     gas = gastype[gas]
    #
    #     new_smiles, split_smiles, smiles_len = get_new_smiles(smiles, splitor=splitor, dsrate=dsrate, loss=loss, gas=gas,
    #                                                           mn=mn)
    #
    #     contribution_dict, unvisited, atom_maps = analyze_smiles_data(smiles, splitor, dsrate=dsrate, loss=loss, gas=gas,
    #                                                                   mn=mn)
    #
    #     tmp = vocab.text2tensor(new_smiles)
    #     source = torch.zeros(model.max_len).int()
    #     source[:len(tmp)] = torch.tensor(tmp).int()
    #
    #     captum_result, delta = interpret_sentence(model, source.unsqueeze(0).to(device), seq_length=model.max_len,
    #                                               device=device)
    #
    #     result, atom_result = acount_mol(captum_result, contribution_dict, atom_maps)
    #
    #     mol = Chem.MolFromSmiles(smiles)
    #     cp = Chem.Mol(mol)
    #     contribs = atom_result
    #     return cp, contribution_dict, atom_maps, contribs, captum_result

    captum_result = torch.load('captum.pt')

    result = {}
    for i in captum_result:
        tmp = captum_result[i]
        # [smiles, contribution_dict, cp, contribs, result, captum_result]
        smiles = tmp[0]
        mol_att = get_result(smiles, groups, contribs=tmp[3])
        result[smiles] = mol_att

    torch.save(result, 'tm_att_result_.pt')

    group_result = {}
    for key in result:
        mol_att = result[key]
        for group in mol_att:
            if group in group_result:
                group_result[group] += mol_att[group]
            else:
                group_result[group] = mol_att[group]

    torch.save(group_result, 'group_result_.pt')

    pd_ = []
    for group in group_result:
        list1 = np.array(group_result[group])
        if len(list1) > 1000:
            if len(list1) > 1000:
                np.random.shuffle(list1)
            pd_.append(pd.DataFrame({group: list1[:1000]+[np.abs(list1).mean()]}))

    df = pd.concat(pd_, axis=1)
    df.fillna(0)

    # df = df.loc[np.argsort(-np.array(means))]
    df.to_csv('group_result_limit.csv')
