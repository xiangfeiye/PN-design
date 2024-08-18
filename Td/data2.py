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
# import torchtext
import re

import collections
import numpy as np


import collections
import numpy as np


class Vocab(object):
    """
    构建词表，并实现文本序列到tensor的转换
    """
    def __init__(self,
                 texts=None,
                 min_fre=1,
                 use_pad=True,
                 use_mask=True,
                 use_unk=True):
        """
        :param texts: 输入的文本序列列表
        :param min_fre: 最小出现频率，频率低于该值的词将被删除，默认为1
        :param use_pad: 是否使用pad字符
        :param use_mask: 是否使用mask字符
        :param use_unk: 是否使用unk字符
        """
        self.min_fre = min_fre
        self.use_pad = use_pad
        self.use_mask = use_mask
        self.use_unk = use_unk

        # 统计词频
        self.orderdict = self.getdict(texts)

        # 构建词表
        self.chars = list(self.orderdict.keys())

        # 添加特殊字符
        if use_mask:
            self.chars = ['mask'] + self.chars
        if use_unk:
            self.chars = ['unk'] + self.chars
        if use_pad:
            self.chars = ['pad'] + self.chars

        # 构建词表索引
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def getdict(self, texts):
        """
        统计词频
        """
        orderdict = collections.OrderedDict()
        for text in texts:
            tmp_list = text.split(' ')
            for word in tmp_list:
                if word in orderdict:
                    orderdict[word] += 1
                else:
                    orderdict[word] = 1
        for key in list(orderdict.keys()):
            if orderdict[key] <= self.min_fre:
                del orderdict[key]
        vd = collections.OrderedDict(sorted(orderdict.items(), key=lambda t: t[1], reverse=True))
        return vd

    def get_len(self):
        """
        获取词表大小
        """
        return len(self.chars)

    def text2tensor(self, text):
        """
        将文本序列转换为tensor
        """
        tmp_list = text.split(' ')
        if self.use_unk:
            text_list = [self.stoi[word] if word in self.stoi else self.stoi['unk'] for word in tmp_list]
        else:
            text_list = [self.stoi[word] for word in tmp_list]

        return text_list


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


class CommonDataset(Dataset):
    def __init__(self,
                 filename,
                 path='',
                 byfile=False,
                 max_len=350,
                 x_name='new_smiles',
                 y_name=['td'],
                 vocab_name=None,
                 dataname='new_smiles',
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
        self.index_list = None
        self.y = torch.tensor(np.array(self.df[y_name].values), dtype=torch.float)
        self.max_len = max_len
        self.device = device

        if vocab_name is None:
            self.vocab = Vocab(self.smiles,
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

    def check_element(self, token_smiles, element_list=None):
        if self.index_list is None:
            if element_list is None:
                element_list = ['c', 'C', 'O', 'n', 'N', 'F', 'U', 'S', 's', 'o', 'K', 'I', 'P', 'V', 'B']
            stoi = self.vocab.stoi
            self.index_list = []
            for element in element_list:
                self.index_list.append(stoi[element])
        mask = torch.zeros(self.max_len)
        for i in range(len(token_smiles)):
            if token_smiles[i] in self.index_list:
                mask[i] = 1
            else:
                mask[i] = 0
        return mask

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
        smiles_tensor = self.vocab.text2tensor(smiles)
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
                else:
                    print(smiles)
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
        y = self.y[index]

        x_len = len(token_smile)
        masks = self.check_element(token_smile)

        if len(token_smile) <= self.max_len:
            padding = [self.vocab.stoi['pad'] for _ in range(self.max_len - x_len)]
            x = token_smile + padding
        else:
            x = token_smile[:self.max_len]

        x = torch.tensor(x, dtype=torch.long)
        # x_len = torch.tensor(len(token_smile), dtype=torch.int)
        return x, y, masks


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    filename = 'test'
    data_name = 'test'
    mydataset = CommonDataset(filename, path="", byfile=True, x_name='smiles', max_len=320, y_name=['td'],
                              vocab_name=None, dataname=data_name)

    print(mydataset.vocab.stoi)
    print(mydataset.smiles[436])
    # print(mydataset.smile_data[mydataset.smiles[46]])
    print(mydataset[436])
    print(mydataset.get_max_len())