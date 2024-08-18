import torch
import torch.nn as nn


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
        self.activate = nn.LeakyReLU()
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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        # self.activate = nn.Tanh()
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        if attn_bias is not None:
            attn_bias[attn_bias == 0] = -1000
            attn_bias[attn_bias == 1] = -10
            attn_bias[attn_bias == 2] = 0.0001
            attn_bias = torch.unsqueeze(attn_bias, 1)
            attn_bias = torch.tile(attn_bias, (1, self.head_size, 1, 1))
            x = x + attn_bias
        x = torch.softmax(x, dim=3)

        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        # x = self.activate(x)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 attention_dropout_rate=0.,
                 dropout_rate=0.1,
                 ffn_size=512,
                 head_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_dropout_rate = attention_dropout_rate
        self.head_size = head_size

        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.MulAtt = MultiHeadAttention(self.hidden_size, self.attention_dropout_rate, self.head_size)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = inputs[0]
        A = inputs[1]
        y = self.attention_norm(x)
        _y = self.MulAtt(y, y, y, attn_bias=A)
        _y = self.attention_dropout(_y)

        y = x + _y

        y = self.ffn_norm(y)
        out = self.ffn(y)
        out = self.ffn_dropout(out)
        y = out + y

        return y


class GENEncoder(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_size=100,
                 attention_dropout_rate=0.1,
                 dropout_rate=0.1,
                 ffn_size=256,
                 latten_size=None,
                 pre_dropout=0.1,
                 head_size=4,
                 num_features=13,
                 num_embeddings=4096,
                 embedding_size=32,
                 target_num=3,
                 max_node=120):
        super().__init__()
        if latten_size is None:
            latten_size = [256, 128, 64]
        self.max_node = max_node
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.target_num = target_num
        gnn_layers = [EncoderLayer(hidden_size=hidden_size,
                                   attention_dropout_rate=attention_dropout_rate,
                                   dropout_rate=dropout_rate,
                                   ffn_size=ffn_size,
                                   head_size=head_size)
                      for _ in range(num_layers)]
        # 20由输入数据的结构决定，即存在20个原子特征
        # self.input_layer = nn.Linear(num_features, hidden_size)
        self.input_embedding = nn.Embedding(num_embeddings=num_embeddings,  embedding_dim=int(embedding_size))
        self.input_layer = nn.Linear(embedding_size * num_features, hidden_size)
        self.layers = nn.ModuleList(gnn_layers)

        # self.readout = RXout(hidden_size=hidden_size)
        # self.pre1 = nn.Linear(hidden_size, latten_size2)
        # self.pre2 = nn.Linear(latten_size2, latten_size3)
        # self.predict = nn.Linear(latten_size3, target_num)
        # self.activate = nn.LeakyReLU()

        self.pre = Net(input_size=hidden_size,
                       hidden_size=latten_size,
                       dropout_rate=pre_dropout,
                       target_num=target_num)

        self.ln = nn.LayerNorm([max_node, hidden_size])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m)
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        x = inputs[0]
        A = inputs[1]
        mask = inputs[2]
        masks = torch.tile(mask, (1, 1, self.hidden_size))

        # x_ = x.int()
        # batch_size = x_.size(0)
        # num_atoms = x_.size(1)
        x = self.input_layer(self.input_embedding(x).view(x.size(0), x.size(1), -1))
        assert x.size(2) == self.hidden_size

        for layer in self.layers:
            x = layer([x, A])

        x = self.ln(x)

        x = torch.mul(x, masks)
        out = torch.sum(x, dim=1)

        # out = self.pre1(out)
        # out = self.activate(out)
        # # print(out)
        #
        # out = self.pre2(out)
        # out = self.activate(out)

        output = self.pre(out)

        return output

