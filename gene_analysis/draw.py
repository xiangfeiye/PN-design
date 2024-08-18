from trainer_2 import *
from data import Vocab

import captum
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from utils import *
from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
# from rdkit.Chem.Draw import IPythonConsole
# from IPython.display import SVG
import rdkit

print(rdkit.__version__)
splitor = smiles_split()

def GetdataFromSmiles(smiles_list, vocab, max_len):
    datas = torch.zeros((len(smiles_list), max_len))
    x_lens = torch.zeros(len(smiles_list))
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        data = torch.tensor(vocab.smi2tensor(smiles))
        x_len = len(data)
        x_lens[i] = x_len
        pad1 = nn.ConstantPad1d((0, max_len - x_len), 0)
        datas[i, :] = pad1(data)
    return datas.int(), x_lens.int()

def smilereplace(smile):
    replace_dict = {
                    'Si': 'V',
                    "Cl": 'U',
                    "Br": 'K',
                    "@@": 'D',
                    }
    for key in replace_dict.keys():
        smile = smile.replace(key, replace_dict[key])

    return smile

def smi2smi(smiles):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smi


def interpret_sentence_model(model, sources, x_lens, device):
    model.zero_grad()
    # pre = model(source)
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    attributions_ig, delta = lig.attribute(sources, reference_indices, n_steps=50, 
                                           additional_forward_args = x_lens,
                                           return_convergence_delta=True)
    
    attributions_ig = attributions_ig.sum(dim=-1).squeeze(0)
#     attributions_ig = attributions_ig / torch.norm(attributions_ig)
    attributions_ig = attributions_ig.cpu().detach().numpy()
    
    return attributions_ig, delta

def analyze_smiles_data_(smiles, splitor=None):
    if splitor is None:
        splitor = smiles_split()
    split_smiles = splitor.split(smiles)
    # 初始化记录
    contribution_dict = {}
    unvisited = set([i for i in range(len(split_smiles))])
    
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
    contribution_dict, visited, char_list = check_ring_atoms(smiles, split_smiles, atom_maps, contribution_dict)
    unvisited = unvisited - set(visited)

    # 检查单键、双键、三键、（顺反异构）符号涉及的原子
    contribution_dict, visited = check_bonds_atoms(split_smiles, char_list, contribution_dict, unvisited)
    unvisited = unvisited - set(visited)

    # 检查括号涉及到的原子
    contribution_dict, visited = get_brackets_atoms(split_smiles, atom_maps,contribution_dict)
    unvisited = unvisited - set(visited)
    
    return contribution_dict, unvisited, atom_maps


def get_map_smiles(smiles):

    smi_list = [smiles]
    datas, x_lens = GetdataFromSmiles(smi_list, vocab, max_len=model.max_len)
    datas = datas.to(device)
    x_lens = x_lens.to(device)
    captum_result, delta = interpret_sentence_model(model, datas, x_lens, device)

    splitor = smiles_split()
    contribution_dict, unvisited, atom_maps = analyze_smiles_data_(smiles, splitor)
    result, atom_result = acount_mol(captum_result, contribution_dict, atom_maps)
    mol = Chem.MolFromSmiles(smiles)
    cp = Chem.Mol(mol)

    contribs = atom_result

    return contribution_dict, cp, contribs, result, captum_result


if __name__ == "__main__":
    model_name = '../tm_models/rewrite_tm_314.ep15'
    filename = '../NO2_O2_results_noF_td5.csv'
    smiles_name = 'smiles'

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda')
    model = torch.load(model_name).to(device)
    model.train()
    vocab = torch.load('vocab.pt')
    seq_length = model.max_len
    PAD_IND = vocab.stoi['pad']
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(model, model.encoder)

    df = pd.read_csv(filename)
    results = {}
    for i in range(len(df)):
        smiles = df[smiles_name][i]
        try:
            contribution_dict, cp, contribs, result, captum_result = get_map_smiles(smiles)
            results[i] = [smiles, contribution_dict, cp, contribs, result, captum_result]
        except:
            print(i, smiles)
    torch.save(results, 'captum.pt')