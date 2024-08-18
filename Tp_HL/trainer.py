import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data import MyDateset, graph_transform
from modellayers import GENEncoder
from torch import default_generator, randperm

from torch.optim.lr_scheduler import StepLR


class Regularization(nn.Module):
    def __init__(self, model, weight_decay=2, p=2):
        """
        :param model:  模型
        :param weight_decay: 正则化参数
        :param p: 指定范数
        """
        super().__init__()
        self.device = None
        if weight_decay <= 0:
            print("param weight_decay cannot <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        """
        获得模型的权重列表
        :param model: 模型
        :return: weight_list
        """
        weight_list = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_loss, weight_decay, p=2):
        """
        计算张量范数
        :param weight_loss:
        :param weight_decay:
        :param p: 范数计算中的幂指数值，默认求2范数
        :return:
        """
        reg_loss = 0
        for name, w in weight_loss:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss

        return reg_loss

    def weight_info(self, weight_list):
        """
        打印权重信息
        :param weight_list:
        :return:
        """
        print(f"regularization weights are:")
        for name, w in weight_list:
            print(name)


def R2Score(true, pre, intype='tensor'):
    r2 = None
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


class GENPreTrainer(object):
    def __init__(self, model: GENEncoder,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, valid_dataloader: DataLoader = None,
                 lr=None, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 use_gpu: bool = True, log_freq: int = 10):
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

    def train(self, epoch, use_reg=False, use_punish=False):
        loss = self.iteration(epoch, self.train_data, training=True, use_reg=use_reg, use_punish=use_punish)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, training=False)
        return loss

    def valid(self, epoch):
        loss = self.iteration(epoch, self.valid_data, training=False)
        return loss

    def iteration(self, epoch, data_loader, training=True, use_reg=False, use_punish=False):
        str_code = "train" if training else "test/valid"
        self.model.train() if training else self.model.eval()
        # data_iter = tqdm.tqdm(enumerate(data_loader),
        #                       desc="EP_%s:%d" % (str_code, epoch),
        #                       total=len(data_loader),
        #                       bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        avg_mae = np.zeros(self.model.target_num)
        avg_r2 = np.zeros(self.model.target_num)
        avg_rmse = np.zeros(self.model.target_num)

        for i, (smile, padded_graph, padded_adj, target, mask) in enumerate(data_loader):
            padded_graph = padded_graph.to(self.device)
            padded_adj = padded_adj.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)

            if training:
                pre = self.model([padded_graph, padded_adj, mask])
            else:
                with torch.no_grad():
                    pre = self.model([padded_graph, padded_adj, mask])

            loss = torch.sum(torch.mean(self.criterion(pre, target), dim=0))

            # 使用惩罚项在早期避免模型输出平均值
            if use_punish:
                panish_loss = torch.sum(torch.abs(torch.var(target, dim=0) - torch.var(pre, dim=0)))
                loss = loss + panish_loss

            # 使用正则化减少过拟合
            if training and use_reg:
                reg_loss = Regularization(self.model, p=2).to(self.device)

                r_loss = reg_loss(self.model)
                if r_loss / loss > 1:
                    regularization_d = int(r_loss / loss * 2)
                else:
                    regularization_d = 1
                r_loss = r_loss / regularization_d

                loss = loss + r_loss

            if training:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()

            if self.mean is not None:
                pre = torch.add(torch.multiply(pre, self.std), self.mean)
                target = torch.add(torch.multiply(target, self.std), self.mean)

            mae = torch.mean(self.mae(pre, target), dim=0).detach().cpu().numpy()
            r2 = R2Score(target, pre).detach().cpu().numpy()
            rmse = torch.sqrt(torch.mean(self.criterion(pre, target), dim=0)).detach().cpu().numpy()

            avg_loss += loss.item()
            avg_mae += mae
            avg_r2 += r2
            avg_rmse += rmse

            # post_fix = {
            #     "epoch": epoch,
            #     "iter": i,
            #     "avg_loss": avg_loss / (i + 1),
            #     "loss": loss.item(),
            #     'avg_mae': avg_mae / (i + 1),
            #     "avg_r2": avg_r2 / (i + 1),
            #     "avg_rmse": avg_rmse / (i + 1)
            # }
            #
            # if i % self.log_freq == 0:
            #     data_iter.write(str(post_fix))

        print(
            f"EP{epoch, str_code, avg_loss / len(data_loader), avg_mae / len(data_loader), avg_r2 / len(data_loader), avg_rmse / len(data_loader)}")

        self.lr_scheduler.step()

        return avg_loss / len(data_loader)

    def save(self, epoch, file_path="output/"):
        output_path = file_path + "GENEncoderPre.ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


def train(train_setting):
    data_path = train_setting['data_path']
    filename = train_setting['filename']
    byfile = train_setting['byfile']
    max_node = train_setting['max_node']
    x_name = train_setting['x_name']
    y_name = train_setting['y_name']
    data_name = train_setting['data_name']
    standard = train_setting['standard']
    out_scaler = train_setting['out_scaler']
    shuffle_ = train_setting['shuffle_']

    seed = train_setting['seed']
    train_ratio = train_setting['train_ratio']
    batch_size = train_setting['batch_size']
    loader_shuffle = train_setting['loader_shuffle']
    drop_last = train_setting['drop_last']

    n_layers = train_setting['n_layers']
    decoder_size = train_setting['decoder_size']
    pre_dropout = train_setting['pre_dropout']
    hidden_size = train_setting['hidden_size']
    dropout = train_setting['dropout']
    head_size = train_setting['head_size']
    num_features = train_setting['num_features']
    num_embeddings = train_setting['num_embeddings']
    embedding_size = train_setting['embedding_size']

    learningrate = train_setting['learningrate']
    adam_beta1 = train_setting['adam_beta1']
    adam_beta2 = train_setting['adam_beta2']
    adam_weight_decay = train_setting['adam_weight_decay']
    use_gpu = train_setting['use_gpu']
    log_freq = train_setting['log_freq']

    epochs = train_setting['epochs']
    output_path = train_setting['output_path']

    device = torch.device('cuda:0' if use_gpu else 'cpu')

    print('load data')
    mydataset = MyDateset(path=data_path, filename=[filename, data_name], smile_name=x_name,
                          y_name=y_name, transform=graph_transform, max_node=max_node,
                          shuffle_=shuffle_, byfile=byfile, standard=standard, out_scaler=out_scaler, device=None)

    traindataloader, testdataloader, validdataloader = dataset2dataloader(mydataset, train_ratio, batch_size,
                                                                          loader_shuffle, drop_last, seed)

    pre_model = GENEncoder(num_layers=n_layers,
                           hidden_size=hidden_size,
                           attention_dropout_rate=dropout,
                           dropout_rate=dropout,
                           ffn_size=hidden_size * 4,
                           latten_size=decoder_size,
                           pre_dropout=pre_dropout,
                           head_size=head_size,
                           num_features=num_features,
                           num_embeddings=num_embeddings,
                           embedding_size=embedding_size,
                           target_num=len(y_name),
                           max_node=max_node)

    pre_model.scaler = mydataset.get_standardscaler()

    print("Creating Trainer")
    trainer = GENPreTrainer(pre_model, train_dataloader=traindataloader, test_dataloader=testdataloader,
                            valid_dataloader=validdataloader, lr=learningrate,
                            betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                            use_gpu=use_gpu, log_freq=log_freq)

    print("Training Start")
    print('epoch, train/test, avg_loss, avg_mae, avg_r2, avg_rmse')
    for epoch in range(epochs):
        trainer.train(epoch, use_reg=(epoch > 16), use_punish=(epoch < 4))
        trainer.test(epoch)
        trainer.lr_scheduler.step()
        trainer.save(epoch, output_path)
    print('valid start')
    trainer.valid(epochs)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_setting = {'data_path': '../../data/',
                     'filename': 'HL_data',
                     'data_name': 'isomer_55',
                     'byfile': True,
                     'max_node': 120,
                     'x_name': 'smiles',
                     'y_name': ['homo', 'lumo', 'hl'],
                     'standard': True,
                     'out_scaler': None,
                     'shuffle_': False,
                     'seed': 42,
                     'train_ratio': [0.8, 0.1, 0.1],
                     'batch_size': 64,
                     'loader_shuffle': True,
                     'drop_last': True,
                     'n_layers': 3,
                     'decoder_size': [256, 32],
                     'dropout': 0.,
                     'pre_dropout': 0.1,
                     'hidden_size': 64,
                     'head_size': 6,
                     'num_features': 20,
                     'num_embeddings': 4096,
                     'embedding_size': 32,
                     'learningrate': [0.0014910336927843352, 28, 0.6],
                     'adam_beta1': 0.9,
                     'adam_beta2': 0.999,
                     'adam_weight_decay': 0.1,
                     'use_gpu': True,
                     'log_freq': 10,
                     'epochs': 36,
                     'output_path': 'modelsave/'}

    try:
        os.mkdir(train_setting['output_path'])
    except:
        print('output_path had been created')

    train(train_setting)
