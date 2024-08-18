import random
import collections
import torch
import numpy as np
import pandas as pd
import os
import io
import codecs
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import Dataset, IterableDataset, DataLoader
import re


class Vocabdict(object):
    def __init__(self,
                 smiles_list,
                 replace_dict=None,
                 min_fre=1):
        self.min_fre = min_fre
        self.smiles_list = smiles_list
        if replace_dict is None:
            self.replace_dict = {
                'Si': 'V',
                "Cl": 'U',
                "Br": 'K',
                "@@": 'D'
            }
        else:
            self.replace_dict = replace_dict
        self.orderdict = self.getdict()

    def getdict(self):
        orderdict = collections.OrderedDict()
        for smiles in self.smiles_list:
            for key in self.replace_dict.keys():
                smiles = smiles.replace(key, self.replace_dict[key])
            for code in smiles:
                if code in orderdict:
                    orderdict[code] += 1
                else:
                    orderdict[code] = 1
        for key in list(orderdict.keys()):
            if orderdict[key] <= self.min_fre:
                del orderdict[key]
        vd = collections.OrderedDict(sorted(orderdict.items(), key=lambda t: t[1], reverse=True))
        return vd


class Vocab(object):
    def __init__(self,
                 smiles_characters=None,
                 weight=None,
                 replace_dict=None,
                 use_AE=False,
                 use_pad=True,
                 use_mask=True,
                 use_unk=True):
        self.use_AE = use_AE
        self.use_pad = use_pad
        self.use_mask = use_mask
        self.use_unk = use_unk

        if smiles_characters is None:
            self.smiles_characters = ['c', 'C', '(', ')', 'O', '1', '2', '=', 'N', 'n', '3', '[', ']', '-',
                                      'F', '4', 'H', 'S', '@', 'Cl', 'o', 's', '5', '+', '#', '1', '/', '\\',
                                      'B', 'r', 'Br', '@@', '6', '7', '8', 'I', '9', 'P', '%', "0", 'Si', '.',
                                      'p', 'b', 'e']
            self.weight = None
        else:
            self.smiles_characters = smiles_characters
            self.weight = list(1 / np.log(weight)) if weight is not None else None
        if replace_dict is None:
            self.replace_dict = {
                'Si': 'V',
                "Cl": 'U',
                "Br": 'K',
                "@@": 'D'
            }
        else:
            self.replace_dict = replace_dict

        total_token = {}
        for i in self.smiles_characters:
            total_token[i] = i
            if i in self.replace_dict:
                total_token[i] = self.replace_dict[i]
        self.chars = list(total_token.values())

        if use_mask:
            self.chars = ['mask'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight
        if use_AE:
            self.chars = ['A'] + self.chars
            self.chars = ['E'] + self.chars
            self.weight = [np.mean(self.weight)] + [np.mean(self.weight)] + self.weight
        if use_unk:
            self.chars = ['unk'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight
        if use_pad:
            self.chars = ['pad'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def get_len(self):
        return len(self.chars)

    def smi2tensor(self, smiles):
        for key in self.replace_dict.keys():
            smiles = smiles.replace(key, self.replace_dict[key])

        new_smile = ''
        for code in smiles:
            new_code = code + ';'
            new_smile = new_smile + new_code

        tmp_list = new_smile.split(';')[:-1]
        if self.use_AE:
            tmp_list = ['A'] + tmp_list + ['E']
        # smiles_list = []
        # for s in tmp_list:
        #     try:
        #         smiles_list.append(self.stoi[s])
        #     except:
        #         smiles_list.append(self.stoi['unk'])
        if self.use_unk:
            smiles_list = [self.stoi[s] if s in self.stoi else self.stoi['unk'] for s in tmp_list]
        else:
            smiles_list = [self.stoi[s] for s in tmp_list]

        return smiles_list


class ReVocab(object):
    def __init__(self,
                 pattern=None,
                 use_AE=True,
                 use_pad=True,
                 use_mask=True,
                 use_unk=True,
                 min_fre=1
                 ):
        self.chars = None
        self.weight = None
        self.smiles_characters = None
        self.stoi = None
        self.itos = None
        self.use_AE = use_AE
        self.use_pad = use_pad
        self.use_mask = use_mask
        self.use_unk = use_unk
        self.min_fre = min_fre

        if pattern is None:
            self.pattern = "(:[0-9]|<|Br?|Cl?|Si?|@@?|H|N|O|S|P|F|I||B|b|c|n|o|s|p|\(" \
                           "|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        else:
            self.pattern = pattern
        self.regex = re.compile(self.pattern)

    def smi2tensor(self, smiles):
        if self.stoi is None:
            self.create_vocab()
        tmp_list = self.regex.findall(smiles)
        if self.use_AE:
            tmp_list = ['A'] + tmp_list + ['E']

        if self.use_unk:
            smiles_list = [self.stoi[s] if s in self.stoi else self.stoi['unk'] for s in tmp_list]
        else:
            smiles_list = [self.stoi[s] for s in tmp_list]

        return smiles_list

    def get_vocabdict(self, smiles_list):
        orderdict = collections.OrderedDict()
        for smiles in smiles_list:
            tmp_list = self.regex.findall(smiles)
            for code in tmp_list:
                if code in orderdict:
                    orderdict[code] += 1
                else:
                    orderdict[code] = 1

        for key in list(orderdict.keys()):
            if orderdict[key] <= self.min_fre:
                del orderdict[key]
        vd = collections.OrderedDict(sorted(orderdict.items(), key=lambda t: t[1], reverse=True))
        return vd

    def create_vocab(self, smiles_characters, weight):
        if smiles_characters is None:
            self.smiles_characters = ['c', 'C', '(', ')', 'O', '1', '2', '=', 'N', 'n', '3', '[', ']', '-',
                                      'F', '4', 'H', 'S', '@', 'Cl', 'o', 's', '5', '+', '#', '1', '/', '\\',
                                      'B', 'r', 'Br', '@@', '6', '7', '8', 'I', '9', 'P', '%', "0", 'Si', '.',
                                      'p', 'b', 'e']
            self.weight = None
        else:
            self.smiles_characters = smiles_characters
            self.weight = list(1 / np.log(weight)) if weight is not None else None

        total_token = {}
        self.chars = list(smiles_characters)

        if self.use_mask:
            self.chars = ['mask'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight
        if self.use_AE:
            self.chars = ['A'] + self.chars
            self.chars = ['E'] + self.chars
            self.weight = [np.mean(self.weight)] + [np.mean(self.weight)] + self.weight
        if self.use_unk:
            self.chars = ['unk'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight
        if self.use_pad:
            self.chars = ['pad'] + self.chars
            self.weight = [np.mean(self.weight)] + self.weight

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def get_len(self):
        return len(self.chars)


def getsmiledatadis(smile_data, begin=0, step=10):
    len_dict = {}
    for line in smile_data:
        len1 = len(line)
        index = (len1 - begin) // step * step
        if index in len_dict:
            len_dict[index] += 1
        else:
            len_dict[index] = 1

    return len_dict


class SmilesTranslatorDataset(Dataset):
    def __init__(self,
                 filename,
                 path='',
                 byfile=False,
                 max_len=200,
                 x_name='smiles',
                 device=None,
                 vocab_name=None,
                 dataname='smiles'):
        """
        Args:
            filename(str): define the csv file which contain smiles and other properties
            path(str): where the files are
            create_(True or False): if True, four files will be created, contain txts and vocab and ordered dict
            create_vocab(True or False): only available while create_. if a vocab is existed and used for a new data
            dataname(str): define how the creating or created file be named
            device: where the date will be send to (cpu or gpu)
        """

        self.path = path
        self.x_name = x_name
        self.df = pd.read_csv(path + filename + '.csv', encoding="utf8")
        self.byfile = byfile
        self.filename = filename
        self.dataname = dataname
        self.smiles = list(self.df[self.x_name])
        self.max_len = max_len
        self.device = device

        if vocab_name is None:
            self.vocabdict = Vocabdict(self.smiles)
            self.orderdict = self.vocabdict.orderdict

            self.vocab = Vocab(smiles_characters=list(self.orderdict.keys()),
                               weight=list(self.orderdict.values()),
                               use_AE=True,
                               use_pad=True,
                               use_mask=True,
                               use_unk=True)
            torch.save(self.vocab, path + 'vocab.pt')
        else:
            self.vocab = torch.load(path + vocab_name)

        if self.byfile:
            try:
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)
            except:
                self.data2file()
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)

    def __len__(self):
        return len(self.smiles)

    def get_vocab_len(self):
        """
        Note:
            give how many tokens in vocab
        """
        return self.vocab.get_len()

    def get_max_len(self):
        """
        Note:
            give a dictory contain {smiles_len: frequent}
        """
        if self.byfile:
            len_dict = getsmiledatadis(self.smile_data)
            return len_dict
        else:
            return None

    def __getitem__(self, index):
        smiles = self.smiles[index]
        if self.byfile:
            token_smile = self.smile_data[smiles]['list']
        else:
            token_smile = self.get_smiles_tensor(smiles)
        x_len = len(token_smile)

        target = np.zeros(self.max_len)
        target[:x_len - 1] = np.array(token_smile[1:])
        # target[-1] = 0
        # x_len = torch.tensor(len(token_smile), dtype=torch.int)
        if len(token_smile) <= self.max_len:
            padding = [self.vocab.stoi['pad'] for _ in range(self.max_len - x_len)]
            x = token_smile + padding
        else:
            x = token_smile[:self.max_len]

        target = torch.tensor(target, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)

        return x, target

    def get_smiles_tensor(self, smiles):
        smiles_tensor = self.vocab.smi2tensor(smiles)
        return smiles_tensor

    def data2file(self, ):
        dataname = self.dataname if self.dataname is not None else 'smiles_tensor.pt'
        data = {}
        indexs = []
        for i in range(len(self.smiles)):
            smiles = self.smiles[i]
            try:
                tmp = self.get_smiles_tensor(smiles)
                data[smiles] = {'list': tmp}
                if len(tmp) <= self.max_len:
                    indexs.append(i)
            except:
                print('failed for smiles: %s' % smiles)

        clean_df = self.df.loc[indexs]
        clean_df.to_csv(self.path + self.filename + '%s.csv' % self.max_len, index=False)

        self.df = pd.read_csv(self.path + self.filename + '%s.csv' % self.max_len)
        self.smiles = list(self.df[self.x_name])

        torch.save(data, self.path + 'smiles_tensor_%s.pt' % dataname)


class Smiles2CsmilesDataset(Dataset):
    def __init__(self,
                 filename,
                 path='',
                 byfile=False,
                 max_len=200,
                 x_name='smiles',
                 device=None,
                 vocab_name=None,
                 dataname='smiles'):
        """
        Args:
            filename(str): define the csv file which contain smiles and other properties
            path(str): where the files are
            create_(True or False): if True, four files will be created, contain txts and vocab and ordered dict
            create_vocab(True or False): only available while create_. if a vocab is existed and used for a new data
            dataname(str): define how the creating or created file be named
            device: where the date will be send to (cpu or gpu)
        """

        self.path = path
        self.x_name = x_name
        self.df = pd.read_csv(path + filename + '.csv', encoding="utf8")
        self.byfile = byfile
        self.filename = filename
        self.dataname = dataname
        self.smiles = list(self.df[self.x_name])
        self.csmiles = list(self.df['smiles'])
        self.max_len = max_len
        self.device = device

        if vocab_name is None:
            self.vocabdict = Vocabdict(self.smiles)
            self.orderdict = self.vocabdict.orderdict

            self.vocab = Vocab(smiles_characters=list(self.orderdict.keys()),
                               weight=list(self.orderdict.values()),
                               use_AE=True,
                               use_pad=True,
                               use_mask=True,
                               use_unk=True)
            torch.save(self.vocab, path + 'vocab.pt')
        else:
            self.vocab = torch.load(path + vocab_name)

        if self.byfile:
            try:
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)
            except:
                self.data2file()
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)

    def __len__(self):
        return len(self.smiles)

    def get_vocab_len(self):
        """
        Note:
            give how many tokens in vocab
        """
        return self.vocab.get_len()

    def get_max_len(self):
        """
        Note:
            give a dictory contain {smiles_len: frequent}
        """
        if self.byfile:
            len_dict = getsmiledatadis(self.smile_data)
            return len_dict
        else:
            return None

    def __getitem__(self, index):
        smiles = self.smiles[index]
        y_smiles = self.csmiles[index]
        if self.byfile:
            token_smile = self.smile_data[smiles]['list']
            y_smiles = self.smile_data[smiles]['y']
        else:
            token_smile = self.get_smiles_tensor(smiles)
            y_smiles = self.get_smiles_tensor(y_smiles)

        x_len = len(token_smile)
        y_len = len(y_smiles)

        # target = np.zeros(self.max_len)
        # target[:x_len-1] = np.array(token_smile[1:])
        # target[-1] = 0
        # x_len = torch.tensor(len(token_smile), dtype=torch.int)
        if len(token_smile) <= self.max_len:
            padding = [self.vocab.stoi['pad'] for _ in range(self.max_len - x_len)]
            x = token_smile + padding
        else:
            x = token_smile[:self.max_len]

        if len(y_smiles) <= self.max_len:
            padding = [self.vocab.stoi['pad'] for _ in range(self.max_len - y_len)]
            y = y_smiles + padding
        else:
            y = y_smiles[:self.max_len]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def get_smiles_tensor(self, smiles):
        smiles_tensor = self.vocab.smi2tensor(smiles)
        return smiles_tensor

    def data2file(self, ):
        dataname = self.dataname if self.dataname is not None else 'smiles_tensor.pt'
        data = {}
        indexs = []
        for i in range(len(self.smiles)):
            smiles = self.smiles[i]
            csmiles = self.csmiles[i]
            try:
                data[smiles] = {'list': self.get_smiles_tensor(smiles), 'y': self.get_smiles_tensor(csmiles)}
                indexs.append(i)
            except:
                print('failed for smiles: %s' % smiles)

        clean_df = self.df.loc[indexs]
        clean_df.to_csv(self.path + self.filename + '.csv', index=False)

        self.df = pd.read_csv(self.path + self.filename + '.csv')
        self.smiles = list(self.df[self.x_name])
        self.csmiles = list(self.df['smiles'])

        torch.save(data, self.path + 'smiles_tensor_%s.pt' % dataname)


class CommonDataset(Dataset):
    def __init__(self,
                 filename,
                 path='',
                 byfile=False,
                 max_len=350,
                 x_name='smiles',
                 other_name=[],
                 y_name=['td'],
                 vocab_name=None,
                 dataname='smiles',
                 standard=True,
                 device=None):

        self.scaler = None
        self.path = path
        self.x_name = x_name
        self.y_name = y_name
        self.df = pd.read_csv(path + filename + '.csv', encoding="utf8")
        self.byfile = byfile
        self.filename = filename
        self.dataname = dataname
        self.smiles = list(self.df[self.x_name])
        if len(other_name) > 0:
            self.others = torch.tensor(np.array(self.df[other_name].values), dtype=torch.float)
        else:
            self.other = None
        self.y = torch.tensor(np.array(self.df[y_name].values), dtype=torch.float)
        self.max_len = max_len
        self.device = device

        if vocab_name is None:
            self.vocabdict = Vocabdict(self.smiles)
            self.orderdict = self.vocabdict.orderdict

            self.vocab = Vocab(smiles_characters=list(self.orderdict.keys()),
                               weight=list(self.orderdict.values()),
                               use_AE=False,
                               use_pad=True,
                               use_mask=False,
                               use_unk=True)
            torch.save(self.vocab, path + 'vocab.pt')
        else:
            self.vocab = torch.load(path + vocab_name)

        if self.byfile:
            try:
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)
            except:
                self.data2file()
                self.smile_data = torch.load(path + 'smiles_tensor_%s.pt' % dataname)

        if standard:
            self._standard()

    def __len__(self):
        return len(self.smiles)

    def _standard(self):
        self.scaler = [torch.mean(self.y, dim=0), torch.std(self.y, dim=0)]
        self.y = (self.y - self.scaler[0]) / self.scaler[1]

    def get_vocab_len(self):
        """
        Note:
            give how many tokens in vocab
        """
        return self.vocab.get_len()

    def get_max_len(self):
        """
        Note:
            give a dictory contain {smiles_len: frequent}
        """
        if self.byfile:
            len_dict = getsmiledatadis(self.smile_data)
            return len_dict
        else:
            return None

    def get_smiles_tensor(self, smiles):
        smiles_tensor = self.vocab.smi2tensor(smiles)
        return smiles_tensor

    def data2file(self, ):
        dataname = self.dataname if self.dataname is not None else 'smiles_tensor.pt'
        data = {}
        indexs = []
        for i in range(len(self.smiles)):
            smiles = self.smiles[i]
            try:
                tmp = self.get_smiles_tensor(smiles)
                data[smiles] = {'list': tmp}
                if len(tmp) <= self.max_len:
                    indexs.append(i)
            except:
                print('failed for smiles: %s' % smiles)

        clean_df = self.df.loc[indexs]
        clean_df.to_csv(self.path + self.filename + '_%s.csv' % self.max_len, index=False)

        self.df = pd.read_csv(self.path + self.filename + '_%s.csv' % self.max_len)
        self.smiles = list(self.df[self.x_name])


        torch.save(data, self.path + 'smiles_tensor_%s.pt' % dataname)

    def __getitem__(self, index):
        smiles = self.smiles[index]

        token_smile = self.smile_data[smiles]['list']
        if self.other is not None:
            other = self.others[index]
        y = self.y[index]

        x_len = len(token_smile)

        if len(token_smile) <= self.max_len:
            padding = [self.vocab.stoi['pad'] for _ in range(self.max_len - x_len)]
            x = token_smile + padding
        else:
            x = token_smile[:self.max_len]

        x = torch.tensor(x, dtype=torch.long)
        x_len = torch.tensor(len(token_smile), dtype=torch.int)

        if self.other is not None:
            return x, y, x_len, other
        else:
            return x, y, x_len


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    filename = 'isomer_21m_2_300'
    data_name = "isomer_21m_2"  # 274171

    mydataset = CommonDataset(filename, path="data/", byfile=True, x_name='smiles_2', max_len=300, y_name=['mp'],
                              other_name=[],
                              vocab_name='rvocab.pt', dataname=data_name)

    # mydataset = Smiles2CsmilesDataset(filename, path="data/", byfile=False, x_name='Csmiles', max_len=120,
    #                                   vocab_name=None, dataname=data_name)
    #
    print(mydataset.vocab.stoi)
    print(mydataset.smiles[436])
    # print(mydataset.smile_data[mydataset.smiles[46]])
    print(mydataset[436])
    print(mydataset.get_max_len())
    print(mydataset.vocab.weight)
    # path = 'data/'
    # df = pd.read_csv(path + 'isomer_21m_2.csv')
    # smiles = list(df['smiles_2'])
    # # a = Vocabdict(smiles)

    # vocab = Vocab(smiles_characters=list(a.orderdict.keys()),
    #               weight=list(a.orderdict.values()),
    #               use_AE=True,
    #               use_pad=True,
    #               use_mask=True,
    #               use_unk=True)
    #
    # print(vocab.stoi)

    # rvocab = ReVocab()
    # orderdict = rvocab.get_vocabdict(smiles)
    # rvocab.create_vocab(smiles_characters=list(orderdict.keys()),
    #                     weight=list(orderdict.values()))
    #
    # print(rvocab.stoi, rvocab.weight)
    #
    # torch.save(rvocab, path + 'rvocab.pt')
