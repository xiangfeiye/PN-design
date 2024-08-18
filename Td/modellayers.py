import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from flash_pytorch import FLASH


class Net(nn.Module):
    def __init__(self,
                 input_size=1024,
                 hidden_size=None,
                 dropout_rate=0.1,
                 target_num=1):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 128, 8]
        num_layer = len(hidden_size)
        self.input_layer = nn.Linear(input_size, hidden_size[0])

        hidden_layers = [nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(num_layer - 1)]
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.pre_layer = nn.Linear(hidden_size[-1], target_num)
        self.activate = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m)
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        x = inputs
        # x = self.drop(x)
        x_ = self.input_layer(x)
        x_ = self.activate(x_)

        for layer in self.hidden_layers:
            x_ = layer(x_)
            x_ = self.activate(x_)
            x_ = self.drop(x_)

        out = self.pre_layer(x_)
        return out


def get_pad_mask(seq_k, pad_idx):
    return (seq_k == pad_idx).unsqueeze(-2)
    # 添加一个维度，和attn进行mask操作的时候可以进行broadcast


# 生成一个上三角矩阵
def get_subsequent_mask(len_s):
    subsequent_mask = (torch.triu(
        torch.ones((1, 1, len_s, len_s)), diagonal=1)).bool()
    return subsequent_mask


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_ln = nn.LayerNorm(hidden)
        self.output_ln = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        _x = self.input_ln(x)
        x = x + self.attention(_x, _x, _x, mask=mask)
        x = x + self.feed_forward(self.output_ln(x))
        return x


class LikeGPT2(nn.Module):
    def __init__(self, n_layers=8, seq_len=512, vocab_size=42, d_model=512, attn_heads=4, dropout=0.1, device=None):
        super().__init__()
        self.nlayers = n_layers
        self.attn_heads = attn_heads
        self.device = device if device is not None else torch.device('cpu')

        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_model * 4, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.token_embedding.weight

        self.mask = get_subsequent_mask(seq_len).to(self.device)

        self.pos_ids = torch.arange(0, seq_len).unsqueeze(0).to(self.device)

    def forward(self, src, pos_ids=None):
        if pos_ids is None:
            pos_ids = self.pos_ids

        inp = self.drop(self.position_embedding(pos_ids) + self.token_embedding(src))

        mask = torch.tile(self.mask, (src.size(0), self.attn_heads, 1, 1))
        for transformer in self.transformer_blocks:
            inp = transformer(inp, mask)

        inp = self.ln(inp)
        logits = self.out(inp)

        return inp, logits


class TransformerPre(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden=768,
                 n_layers=12,
                 attn_heads=12,
                 padding_idx=0,
                 max_len=512,
                 dropout=0.1,
                 decoder_size=[256],
                 pre_dropout=0.1,
                 target_num=1,
                 device=None):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.padding_idx = padding_idx

        self.pos_ids = torch.arange(0, max_len).unsqueeze(0).to(self.device)

        self.position_embedding = nn.Embedding(max_len, hidden)
        self.token_embedding = nn.Embedding(vocab_size, hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # self.tran = nn.Linear(hidden, 1)
        self.pre = Net(input_size=hidden,
                       hidden_size=decoder_size,
                       dropout_rate=pre_dropout,
                       target_num=target_num)

        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pos_ids=None):
        if pos_ids is None:
            pos_ids = self.pos_ids

        inp = self.drop(self.position_embedding(pos_ids) + self.token_embedding(x))

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            inp = transformer(inp)

        inp = self.ln(inp)
        # inp = self.tran(inp).squeeze()
        out = self.pre(inp[:, 0, :])

        return out


class LSTMModelPre(nn.Module):
    def __init__(self,
                 ntoken,
                 max_len,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 proj_size=0,
                 mlp_dropout=0.2,
                 target_num=1,
                 decoder_size=None,
                 mask=False,
                 device=None
                 ):
        super().__init__()
        if decoder_size is None:
            decoder_size = [64, 8]
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.mask = mask
        self.device = device
        self.model_type = 'LSTMModelPre'
        self.encoder = nn.Embedding(ntoken, input_size, padding_idx=0)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional,
                            proj_size=proj_size)

        self.bidirectional = 2 if bidirectional else 1

        self.pre = Net(input_size=self.num_layers * self.hidden_size * self.bidirectional,
                       hidden_size=decoder_size,
                       dropout_rate=mlp_dropout,
                       target_num=target_num)

        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        x = input[0]
        x_len = input[1]

        x_lengths, idx = x_len.sort(0, descending=True)
        x_lengths = x_lengths.to('cpu')
        _, un_idx = torch.sort(idx, dim=0)
        x_ = x[idx]

        x_input = self.drop(self.encoder(x_))
        x_packed_input = pack_padded_sequence(input=x_input, lengths=x_lengths, batch_first=self.batch_first)

        _, (ht, _) = self.lstm(x_packed_input)
        #
        # out = out.reshape((-1, self.max_len * self.hidden_size * self.bidirectional))
        # # ht = ht.permute(1,0,2).reshape((-1, self.num_layers*self.hidden_size*self.bidirectional))
        # # hout = torch.squeeze(ht[-1, :, :])
        # # out = torch.index_select(ht , 0, un_idx)
        # pred = self.pre(out)

        ht = ht.permute(1, 0, 2).reshape((-1, self.num_layers * self.hidden_size * self.bidirectional))
        out = torch.index_select(ht, 0, un_idx)
        pred = self.pre(out)

        return pred


class LSTMModelNodePre(nn.Module):
    def __init__(self,
                 ntoken,
                 max_len,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 proj_size=0,
                 mlp_dropout=0.2,
                 target_num=1,
                 decoder_size=None,
                 mask=False,
                 device=None
                 ):
        super().__init__()
        if decoder_size is None:
            decoder_size = [64, 8]
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.mask = mask
        self.device = device
        self.model_type = 'LSTMModelPre'
        self.encoder = nn.Embedding(ntoken, input_size, padding_idx=0)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional,
                            proj_size=proj_size)

        self.bidirectional = 2 if bidirectional else 1

        self.pre = Net(input_size=self.num_layers * self.hidden_size * self.bidirectional,
                       hidden_size=decoder_size,
                       dropout_rate=mlp_dropout,
                       target_num=target_num)

        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        x = input
        # masks = input[1].unsqueeze(dim=-1)

        # x_lengths, idx = x_len.sort(0, descending=True)
        # x_lengths = x_lengths.to('cpu')
        # masks = torch.tensor()
        # _, un_idx = torch.sort(idx, dim=0)
        # x_ = x[idx]

        x_input = self.drop(self.encoder(x))
        # x_packed_input = pack_padded_sequence(input=x_input, lengths=x_lengths, batch_first=self.batch_first)
        _, (ht, _) = self.lstm(x_input)
        # output, (ht, _) = self.lstm(x_packed_input)
        # output, _ = pad_packed_sequence(output, batch_first=self.batch_first, total_length=self.max_len)
        # out = out.reshape((-1, self.max_len * self.hidden_size * self.bidirectional))
        # # ht = ht.permute(1,0,2).reshape((-1, self.num_layers*self.hidden_size*self.bidirectional))
        # # hout = torch.squeeze(ht[-1, :, :])
        # # out = torch.index_select(ht , 0, un_idx)
        # pred = self.pre(out)

        # ht = ht.permute(1, 0, 2).reshape((-1, self.num_layers * self.hidden_size * self.bidirectional))
        # output = torch.index_select(output, 0, un_idx)
        ht = ht.permute(1, 0, 2).reshape((-1, self.num_layers * self.hidden_size * self.bidirectional))
        pred = self.pre(ht)

        return pred

