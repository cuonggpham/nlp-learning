from math import sin, cos, sqrt, log
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
<<<<<<< HEAD
        Lop Embedding de chuyen mot tu sang khong gian embedding (dai dien so)
        :param vocab_size: kich thuoc tu vung
        :param embed_dim: kich thuoc vector embedding

        vi du: neu co 1000 tu va kich thuoc embedding la 512,
        thi lop embedding se la ma tran 1000x512

        gia su co batch size la 64 va do dai cau la 15 tu,
        thi dau ra se la 64x15x512
=======
        Embedding class to convert a word into embedding space (numerical representation)
        :param vocab_size: the vocabulary size
        :param embed_dim: the embedding dimension

        example: if we have 1000 vocabulary size and our embedding is 512,
        then the embedding layer will be 1000x512

        suppose we have a batch size of 64 and sequence of 15 words,
        then the output will be 64x15x512
>>>>>>> 5952ff2ff999616bdb960be3cff142a54d631a52
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
<<<<<<< HEAD
        lan truyen tien
        :param x: tu hoac chuoi cac tu
        :return: dai dien so cua dau vao
        """
        output = self.embed(x) * sqrt(self.embed_dim)
        # print(f"Kich thuoc embedding: {output.shape}")
=======
        forward pass
        :param x: the word or sequence of words
        :return: the numerical representation of the input
        """
        output = self.embed(x) * sqrt(self.embed_dim)
        # print(f"Embedding shape: {output.shape}")
>>>>>>> 5952ff2ff999616bdb960be3cff142a54d631a52
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=5000, dropout=0.1):
        """
<<<<<<< HEAD
        Embedding vi tri hoac ma hoa vi tri
        Y tuong la cong ma hoa vi tri vao embedding dau vao
        truoc khi dua vao encoder/decoder dau tien
        Ma hoa vi tri phai co cung kich thuoc voi vector embedding
        Su dung sin va cos de tao ma hoa vi tri
        Xem them phan "Positional Encoding" trong bai bao "Attention Is All You Need"

        :param embed_dim: kich thuoc embedding, phai giong voi vector embedding
        :param max_seq_len: do dai chuoi toi da (so tu toi da)
        :param dropout: xac suat dropout
=======
        Positional Embedding or Positional Encoding
        The general idea here is to add positional encoding to the input embedding
        before feeding the input vectors to the first encoder/decoder
        The positional embedding must have the same embedding dimension as in the embedding vectors
        For the positional encoding we use sin and cos
        For more details, check "Positional Encoding" section in the "Attention Is All You Need" paper

        :param embed_dim: the size of the embedding, this must be the same as in embedding vector
        :param max_seq_len: the maximum sequence length (max sequence of words)
        :param dropout: the dropout probability
>>>>>>> 5952ff2ff999616bdb960be3cff142a54d631a52
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

<<<<<<< HEAD
        # su dung register_buffer de luu tham so "pe" vao state_dict
=======
        # we use register_buffer to save the "pe" parameter to the state_dict
>>>>>>> 5952ff2ff999616bdb960be3cff142a54d631a52
        self.register_buffer('pe', pe)

    def pe_sin(self, position, i):
        return sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        # print(x.shape)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)