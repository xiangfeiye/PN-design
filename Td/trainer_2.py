import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data2 import Vocab, CommonDataset
from modellayers import LikeGPT2, TransformerPre, LSTMModelPre, LSTMModelNodePre
from torch import default_generator, randperm

from torch.optim.lr_scheduler import StepLR


def R2Score(true, pre, intype='tensor'):
    r2 = 0
    if intype == 'tensor':
        # a_mean = torch.mean(a)
        true_mean = torch.mean(true, dim=0)
        sst = torch.sum((true - true_mean) * (true - true_mean), dim=0)
        sse = torch.sum((true - pre) * (true - pre), dim=0)
        # ssr = torch.sum((a-b_mean)*(a-b_mean))
        r2 = 1 - sse / (sst + 10 ** -8)
    return r2


def dataset2dataloader(mydataset, train_ratio, batch_size, loader_shuffle, drop_last, seed=42):
    print('create dataloader')
    dataset_size = len(mydataset)
    indices = randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()

    train_size = int(dataset_size * train_ratio[0])
    test_size = int(dataset_size * train_ratio[1])

    train_dataset = Subset(mydataset, indices=indices[:train_size])
    test_dataset = Subset(mydataset, indices=indices[train_size:train_size + test_size])
    valid_dataset = Subset(mydataset, indices=indices[train_size + test_size:])

    traindataloader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        shuffle=loader_shuffle,
        drop_last=drop_last,
        num_workers=2
    )

    testdataloader = DataLoader(
        batch_size=min(batch_size, len(test_dataset)),
        dataset=test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    validdataloader = DataLoader(
        batch_size=min(batch_size, len(valid_dataset)),
        dataset=valid_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )
    return traindataloader, testdataloader, validdataloader


class TransformerPreTrainer(object):
    def __init__(self, model: TransformerPre, vocab,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 use_gpu: bool = True, log_freq: int = 10, ignore_index=0):
        gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device("cuda:0" if gpu else "cpu")

        self.model = model.to(self.device)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.valid_data = valid_dataloader

        self.optim = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.lr_scheduler = StepLR(self.optim, step_size=8, gamma=0.5)

        if self.model.scaler is not None:
            self.mean = self.model.scaler[0].to(self.device)
            self.std = self.model.scaler[1].to(self.device)
        else:
            self.mean = None
            self.std = None

    def train(self, epoch):
        self.iteration(epoch, self.train_data, training=True)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, training=False)

    def valid(self, epoch):
        self.iteration(epoch, self.valid_data, training=False)

    def iteration(self, epoch, data_loader, training=True):
        str_code = "train" if training else "test/valid"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        avg_mae = 0.0
        avg_r2 = 0.0
        avg_rmse = 0.0

        for i, (source, target, x_len) in data_iter:
            source = source.to(self.device)
            target = target.to(self.device)
            if training:
                pre = self.model(source)
            else:
                with torch.no_grad():
                    pre = self.model(source)

            loss = torch.sum(torch.mean(self.criterion(pre, target)))

            if training:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()

            if self.mean is not None:
                pre = torch.add(torch.multiply(pre, self.std), self.mean)
                target = torch.add(torch.multiply(target, self.std), self.mean)

            mae = torch.sum(torch.mean(self.mae(pre, target), dim=0))
            r2 = R2Score(target, pre)
            rmse = torch.sqrt(torch.sum(torch.mean(self.criterion(pre, target), dim=0)))

            avg_loss += loss.item()
            avg_mae += mae.item()
            avg_r2 += r2.item()
            avg_rmse += rmse.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
                'avg_mae': avg_mae / (i + 1),
                "avg_r2": avg_r2 / (i + 1),
                "avg_rmse": avg_rmse / (i + 1)
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss, avg_mae, avg_r2, avg_rmse =" % (epoch, str_code), avg_loss / len(data_iter),
              avg_mae / len(data_iter), avg_r2 / len(data_iter), avg_rmse / len(data_iter))

    def save(self, epoch, file_path="output/"):
        output_path = file_path + "TransformerPre.ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class LSTMModelPreTrainer(object):
    def __init__(self, model, vocab,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 lr=None, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 use_gpu: bool = True, log_freq: int = 10, ignore_index=0):
        if lr is None:
            lr = [0.001, 16, 0.5]
        gpu = torch.cuda.is_available() and use_gpu
        self.device = torch.device("cuda:0" if gpu else "cpu")

        self.model = model.to(self.device)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.valid_data = valid_dataloader

        self.optim = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr[0])
        self.criterion = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.lr_scheduler = StepLR(self.optim, step_size=lr[1], gamma=lr[2])

        if self.model.scaler is not None:
            self.mean = self.model.scaler[0].to(self.device)
            self.std = self.model.scaler[1].to(self.device)
        else:
            self.mean = None
            self.std = None

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data, training=True)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, training=False)
        return loss

    def valid(self, epoch):
        loss = self.iteration(epoch, self.valid_data, training=False)
        return loss

    def iteration(self, epoch, data_loader, training=True):
        str_code = "train" if training else "test/valid"
        self.model.train() if training else self.model.eval()
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                       desc="EP_%s:%d" % (str_code, epoch),
        #                       total=len(data_loader),
        #                       bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        avg_mae = 0.0
        avg_r2 = 0.0
        avg_rmse = 0.0

        for i, (source, target, x_len) in enumerate(data_loader):
            source = source.to(self.device)
            target = target.to(self.device)
            x_len = x_len.to(self.device)
            if training:
                pre = self.model(source)
            else:
                with torch.no_grad():
                    pre = self.model(source)

            loss = torch.sum(torch.mean(self.criterion(pre, target), dim=0))

            if training:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()

            if self.mean is not None:
                pre = torch.add(torch.multiply(pre, self.std), self.mean)
                target = torch.add(torch.multiply(target, self.std), self.mean)

            mae = torch.sum(torch.mean(self.mae(pre, target), dim=0))
            r2 = R2Score(target, pre)
            rmse = torch.sqrt(torch.sum(torch.mean(self.criterion(pre, target), dim=0)))

            avg_loss += loss.item()
            avg_mae += mae.item()
            avg_r2 += r2.item()
            avg_rmse += rmse.item()

            # post_fix = {
            #     "epoch": epoch,
            #     "iter": i,
            #     "avg_loss": avg_loss / (i + 1),
            #     "loss": loss.item(),
            #     'avg_mae': avg_mae / (i + 1),
            #     "avg_r2": avg_r2 / (i + 1),
            #     "avg_rmse": avg_rmse / (i + 1)
            # }

            # if i % self.log_freq == 0:
            #     data_iter.write(str(post_fix))

        print("EP%d, %s, %.4f, %.4f, %.4f, %.4f," % (epoch, str_code, avg_loss / len(data_loader),
                                                     avg_mae / len(data_loader), avg_r2 / len(data_loader),
                                                     avg_rmse / len(data_loader)))

        self.lr_scheduler.step()

        return avg_mae / len(data_loader)

    def save(self, epoch, file_path="output/"):
        output_path = file_path + "LSTMModelPre.ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def train(train_setting):
    data_path = train_setting['data_path']
    filename = train_setting['filename']
    byfile = train_setting['byfile']
    max_len = train_setting['max_len']
    x_name = train_setting['x_name']
    y_name = train_setting['y_name']
    vocab_name = train_setting['vocab_name']
    data_name = train_setting['data_name']
    standard = train_setting['standard']

    seed = train_setting['seed']
    train_ratio = train_setting['train_ratio']
    batch_size = train_setting['batch_size']
    loader_shuffle = train_setting['loader_shuffle']
    drop_last = train_setting['drop_last']

    n_layers = train_setting['n_layers']
    input_size = train_setting['input_size']
    decoder_size = train_setting['decoder_size']
    pre_dropout = train_setting['pre_dropout']
    hidden_size = train_setting['hidden_size']
    dropout = train_setting['dropout']
    bidirectional = train_setting['bidirectional']
    proj_size = train_setting['proj_size']
    learningrate = train_setting['learningrate']
    adam_beta1 = train_setting['adam_beta1']
    adam_beta2 = train_setting['adam_beta2']
    adam_weight_decay = train_setting['adam_weight_decay']
    warmup_steps = train_setting['warmup_steps']
    use_gpu = train_setting['use_gpu']
    log_freq = train_setting['log_freq']

    epochs = train_setting['epochs']
    output_path = train_setting['output_path']

    device = torch.device('cuda:0' if use_gpu else 'cpu')

    print('load data')
    mydataset = CommonDataset(path=data_path,
                              filename=filename,
                              byfile=byfile,
                              max_len=max_len,
                              x_name=x_name,
                              y_name=y_name,
                              device=None,
                              vocab_name=vocab_name,
                              dataname=data_name,
                              standard=standard)

    traindataloader, testdataloader, validdataloader = dataset2dataloader(mydataset, train_ratio, batch_size,
                                                                          loader_shuffle, drop_last, seed)

    print('build predict model')
    # pre_model = TransformerPre(vocab_size=mydataset.get_vocab_len(), hidden=hidden_size, n_layers=n_layers,
    #                            attn_heads=attn_heads,
    #                            padding_idx=mydataset.vocab.stoi['pad'], max_len=mydataset.max_len, dropout=dropout,
    #                            decoder_size=decoder_size, pre_dropout=pre_dropout, target_num=len(mydataset.y_name),
    #                            device=device)
    pre_model = LSTMModelNodePre(ntoken=mydataset.get_vocab_len(), max_len=mydataset.max_len, input_size=input_size,
                                 hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                                 bidirectional=bidirectional, proj_size=proj_size, mlp_dropout=pre_dropout,
                                 decoder_size=decoder_size, target_num=len(mydataset.y_name),
                                 device=device)

    pre_model.scaler = mydataset.scaler if mydataset.scaler is not None else None

    print("Creating Trainer")
    trainer = LSTMModelPreTrainer(pre_model, mydataset.vocab, train_dataloader=traindataloader,
                                  test_dataloader=testdataloader, valid_dataloader=validdataloader, lr=learningrate,
                                  betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                                  warmup_steps=warmup_steps,
                                  use_gpu=use_gpu, log_freq=log_freq, ignore_index=mydataset.vocab.stoi['pad'])

    print("Training Start")
    print('epoch, train/test, avg_loss, avg_mae, avg_r2, avg_rmse')
    for epoch in range(epochs):
        loss1 = trainer.train(epoch)
        loss2 = trainer.test(epoch)
        trainer.save(epoch, output_path)
        trainer.lr_scheduler.step()
    print('valid start')
    trainer.valid(epochs)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_setting = {'data_path':'',
                     'filename': 'data',
                     'byfile': True,
                     'max_len': 320,
                     'x_name': 'new_smiles',
                     'y_name': ['td'],
                     'vocab_name': None,
                     'data_name': 'clean',
                     'standard': True,
                     'seed': 0,
                     'train_ratio': [0.8, 0.1, 0.1],
                     'batch_size': 128,
                     'loader_shuffle': True,
                     'drop_last': True,
                     'n_layers': 7,
                     'input_size': 256,
                     'type_size': 8,
                     'bidirectional': True,
                     'proj_size': 0,
                     'decoder_size': [256, 32],
                     'dropout': 0.,
                     'pre_dropout': 0.3,
                     'hidden_size': 128,
                     'learningrate': [0.004307098183807684, 8, 0.8],
                     'adam_beta1': 0.9,
                     'adam_beta2': 0.999,
                     'adam_weight_decay': 0.1,
                     'warmup_steps': 2000,
                     'use_gpu': True,
                     'log_freq': 100,
                     'epochs': 64,
                     'output_path': 'modelsave/'}

    try:
        os.mkdir('modelsave/')
    except:
        print('output_path had been created')
    for i in [42, 1201,1226, 1997, 2023, 2102]:
        train_setting['seed'] = i
        train_setting['output_path'] = 'modelsave/%s'%i
        train(train_setting)