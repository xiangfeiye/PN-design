import os
import torch
import pandas as pd
from rdkit import Chem
import numpy as np
import rdkit.Chem.AllChem as AllChem
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import trange

HybridizationType = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
ChiralType = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
flag_list = ["False", "True"]
chiral_centers_list = ['R', 'S']
bond_dict = {"UNSPECIFIED": 1, "SINGLE": 2, "DOUBLE": 3, "TRIPLE": 4, "QUADRUPLE": 5, "QUINTUPLE": 6,
             "HEXTUPLE": 7, "ONEANDAHALF": 8, "TWOANDAHALF": 9, "THREEANDAHALF": 10, "FOURANDAHALF": 11,
             "FIVEANDAHALF": 12, "AROMATIC": 13, "IONIC": 14, "HYDROGEN": 15, "THREECENTER": 16, "DATIVEONE": 17,
             "DATIVE": 18, "DATIVEL": 19, "DATIVER": 20, "OTHER": 21, "ZERO": 22}
stero_dict = {"STEREONONE": 0, "STEREOANY": 1, "STEREOZ": 2, "STEREOE": 3, "STEREOCIS": 4, "STEREOTRANS": 5}


def convert_to_single_emb(x, offset=128):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def get_atom_feature(atom, mol):
    # rdkit_atom_features: 13
    atom_num = atom.GetAtomicNum()  # int
    chiral_tag = int(atom.GetChiralTag())  # 0,1,2,3
    degree = atom.GetDegree()  # int
    total_degree = atom.GetTotalDegree()  # int
    possible_formal_charge = atom.GetFormalCharge()  # int

    possible_numH = atom.GetTotalNumHs()  # int
    possible_number_radical_e = atom.GetNumRadicalElectrons()  # int
    possible_hybridization = HybridizationType.index(
        str(atom.GetHybridization()))  # ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S']
    possible_is_aromatic = flag_list.index(str(atom.GetIsAromatic()))
    possible_is_in_ring = 0
    for s in range(3, 13):
        if atom.IsInRingSize(s):
            possible_is_in_ring = s  # [3,13)]
            break
    explicit_valence = atom.GetExplicitValence()
    implicit_valence = atom.GetImplicitValence()
    total_valence = atom.GetTotalValence()
    # atom's GasteigerCharge: 1
    # atom_GasteigerCharge = float(atom.GetProp('_GasteigerCharge'))

    # atom's env feature: 7
    env_features = envatom_features(mol, atom)  # list

    atom_features = [atom_num, chiral_tag, degree, total_degree, possible_formal_charge, possible_numH,
                     possible_number_radical_e, possible_hybridization, possible_is_aromatic,
                     possible_is_in_ring, explicit_valence, implicit_valence, total_valence]

    atom_features.extend(env_features)  # shape: num_features(20)

    return atom_features


def envatom_feature(mol, radius, atom_idx):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx, useHs=True)
    submol = Chem.PathToSubmol(mol, env, atomMap={})
    return submol.GetNumAtoms()


def envatom_features(mol, atom):
    return [
        envatom_feature(mol, r, atom.GetIdx()) for r in range(2, 9)
    ]


def graph_transform(smile, max_node=120):
    smile_string = smile.strip()
    mol = Chem.MolFromSmiles(smile_string)
    try:
        mol = Chem.RemoveHs(mol)
    except:
        print(smile_string)

    num_node = mol.GetNumAtoms()
    assert num_node < max_node
    # 收集原子信息
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features = get_atom_feature(atom, mol)
        atom_features_list.append(atom_features)
    # 增加一个全局点，与每一个原子均连接
    """global_feature = get_global_feature(mol)
    atom_features_list.append(global_feature)
    atom_features_list = np.array(atom_features_list)"""

    atom_features_list = torch.tensor(atom_features_list, dtype=torch.float)
    # 获取邻接矩阵
    # adj_list = []
    adjtmp = Chem.rdmolops.GetAdjacencyMatrix(mol)
    if adjtmp.shape[0] <= max_node:
        adjtmp = adjtmp + np.eye(len(adjtmp))
    # 增加一个全局连接点
    """adj_list = np.zeros([adjtmp.shape[0] + 1, adjtmp.shape[0] + 1])
    adj_list[:, 0] = 1
    adj_list[0, :] = 1
    adj_list[1:, 1:] = adjtmp
    adj_list = torch.tensor(adj_list, dtype=torch.float)"""

    adj_list2 = np.ones([num_node, num_node])

    assert adjtmp.shape == adj_list2.shape
    adj_list = adj_list2 + adjtmp
    adj_list = torch.tensor(adj_list, dtype=torch.float)

    return num_node, atom_features_list, adj_list


def shuffle(graph, adj, mask, max_node):
    index = np.random.permutation(max_node)
    mask = mask[index]
    graph = graph[index, :]
    adj = adj[index, :]
    adj = adj[:, index]
    return graph, adj, mask


class MyDateset(Dataset):
    def __init__(self,
                 path,
                 filename,
                 smile_name="smiles",
                 y_name=None,
                 transform=graph_transform,
                 byfile=False,
                 max_node=120,
                 standard=False,
                 out_scaler=None,
                 shuffle_=False,
                 device=None):
        super().__init__()

        self.device = device
        self.path = path
        self.max_node = max_node
        self.y_name = y_name
        self.transform = transform
        self.byfile = byfile
        self.standard = standard
        self.shuffle_ = shuffle_
        self.filename = filename

        self.df = pd.read_csv(self.path + filename[0] + '.csv', encoding="utf8")
        self.smiles = list(self.df[smile_name])
        self.y = np.array(self.df[self.y_name].values, dtype=float)

        if self.byfile:
            try:
                self.data = torch.load(self.path + filename[1] + '.pt')
            except:
                self.transform2file(self.path + filename[1] + '.pt')
                self.data = torch.load(self.path + filename[1] + '.pt')
                self.df = pd.read_csv(self.path + '%s_%s.csv' % (self.filename[0], self.max_node), encoding="utf8")
                self.smiles = list(self.df[smile_name])
                self.y = np.array(self.df[self.y_name].values, dtype=float)

        if self.standard:
            if out_scaler is None:
                self.y_mean = torch.tensor(np.mean(self.y, axis=0), dtype=torch.float)
                self.y_std = torch.tensor(np.std(self.y, axis=0), dtype=torch.float)
            else:
                self.y_mean = torch.tensor(out_scaler[0])
                self.y_std = torch.tensor(out_scaler[1])

    def __getitem__(self, index):
        smile = self.smiles[index]
        y = torch.tensor(self.y[index], dtype=torch.float)

        if self.byfile:
            data = self.data[smile]
            num_node = data['num_node']
            graph = data['graph']
            adj = data['adj']
        else:
            num_node, graph, adj = self.transform(smile, max_node=self.max_node)

        graph = convert_to_single_emb(graph).long()
        mask = torch.zeros(self.max_node, 1)
        # if self.device is not None:
        #     graph = graph.to(self.device)
        #     adj = adj.to(self.device)
        #     mask = mask.to(self.device)
        #     y = y.to(self.device)
        #     self.y_mean = self.y_mean.to(self.device)
        #     self.y_std = self.y_std.to(self.device)
        if self.standard:
            y = (y - self.y_mean) / (self.y_std + 10 ** -9)

        pad1 = nn.ConstantPad2d((0, 0, 0, self.max_node - num_node), 0)
        pad2 = nn.ConstantPad2d((0, self.max_node - num_node, 0, self.max_node - num_node), 0)

        padded_graph = pad1(graph)
        padded_adj = pad2(adj)
        mask[:num_node] = 1 / num_node

        if self.shuffle_:
            padded_graph, padded_adj, mask = shuffle(padded_graph, padded_adj, mask, self.max_node)

        assert padded_adj.shape[0] == self.max_node
        assert padded_graph.shape[0] == self.max_node

        return smile, padded_graph, padded_adj, y, mask

    def transform2file(self, save_filename):
        print('create new datafile')
        indexs = []
        datas = {}
        for i in trange(len(self.df)):
            smile = self.smiles[i]
            num_node, graph, adj = self.transform(smile, max_node=self.max_node)
            if num_node < self.max_node:
                indexs.append(i)
            data = {'num_node': num_node, 'graph': graph, 'adj': adj}
            datas['%s' % smile] = data

        new_df = self.df.loc[indexs]
        new_df.to_csv(self.path + '%s_%s.csv' % (self.filename[0], self.max_node), index=False)
        torch.save(datas, save_filename)

    def __len__(self):
        return self.df.count()[0]

    def get_inputs_bysmile(self, smi, shuffle_=False):
        smile = smi
        num_node, graph, adj = self.transform(smile, max_node=self.max_node)
        graph = convert_to_single_emb(graph)
        mask = torch.zeros(self.max_node, 1)

        pad1 = nn.ConstantPad2d((0, 0, 0, self.max_node - num_node), 0)
        pad2 = nn.ConstantPad2d((0, self.max_node - num_node, 0, self.max_node - num_node), 0)

        padded_graph = pad1(graph)
        padded_adj = pad2(adj)
        mask[:num_node] = 1

        if shuffle_:
            padded_graph, padded_adj, mask = shuffle(padded_graph, padded_adj, mask, self.max_node)

        assert padded_adj.shape[0] == self.max_node
        assert padded_graph.shape[0] == self.max_node

        return smile, padded_graph, padded_adj, mask

    def get_standardscaler(self) -> list:
        if not self.standard:
            return None
        else:
            scaler = [self.y_mean, self.y_std]
            return scaler


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_GPU = True
    if use_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = None

    mydataset = MyDateset(path='../data/', filename=['isomer_55k_120', 'isomer_55'], smile_name='smiles',
                          y_name=['homo', 'lumo', 'hl'], transform=graph_transform, max_node=120,
                          shuffle_=False, byfile=True, standard=True, out_scaler=None, device=device)

    # smile1, padded_graph1, padded_adj1, mask1 = my_dataset.get_inputs_bysmile("CCOC(=O)c1ccccc1NC(=O)/C=C/c1ccccc1F")
    # smile, padded_graph, padded_adj, y, mask = my_dataset[0]
    # scaler = my_dataset.get_standardscaler()
    # print(padded_graph)
    # print(scaler)
    # smile, padded_graph, padded_adj, y, mask = my_dataset[0]
    # print(smile, padded_graph, y)
    # save_filename = 'prepared_59k.pt'
    for i in range(4):
        print(mydataset[i])
