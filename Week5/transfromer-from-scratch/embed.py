from math import sin, cos, sqrt, log
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
        Lop Embedding de chuyen mot tu sang khong gian embedding (dai dien so)
        :param vocab_size: kich thuoc tu vung
        :param embed_dim: kich thuoc vector embedding

        vi du: neu co 1000 tu va kich thuoc embedding la 512,
        thi lop embedding se la ma tran 1000x512

        gia su co batch size la 64 va do dai cau la 15 tu,
        thi dau ra se la 64x15x512
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        lan truyen tien
        :param x: tu hoac chuoi cac tu
        :return: dai dien so cua dau vao
        """
        output = self.embed(x) * sqrt(self.embed_dim)
        # print(f"Kich thuoc embedding: {output.shape}")
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=5000, dropout=0.1):
        """
        Embedding vi tri hoac ma hoa vi tri
        Y tuong la cong ma hoa vi tri vao embedding dau vao
        truoc khi dua vao encoder/decoder dau tien
        Ma hoa vi tri phai co cung kich thuoc voi vector embedding
        Su dung sin va cos de tao ma hoa vi tri
        Xem them phan "Positional Encoding" trong bai bao "Attention Is All You Need"

        :param embed_dim: kich thuoc embedding, phai giong voi vector embedding
        :param max_seq_len: do dai chuoi toi da (so tu toi da)
        :param dropout: xac suat dropout
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        pe = positional_encoding.unsqueeze(0)

        # su dung register_buffer de luu tham so "pe" vao state_dict
        self.register_buffer('pe', pe)

    def pe_sin(self, position, i):
        return sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        # print(x.shape)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)