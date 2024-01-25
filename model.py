import torch
import torch.nn as nn
import math


class LayerNorm(nn.Module):  # 归一化 Xi = (Xi-μ)/σ
    def __init__(self, emb_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_size))
        self.beta = nn.Parameter(torch.zeros(emb_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Add_position(nn.Module):
    def __init__(self, emb_size):
        super(Add_position, self).__init__()
        self.emb_size = emb_size
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_dp):
        seq_length = input_dp.size(1)
        batch_size = input_dp.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_dp.device)
        position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length)).unsqueeze(2)

        pe = torch.zeros(input_dp.shape).cuda()
        div = torch.exp(torch.arange(0., self.emb_size, 2) * -(math.log(10000.0) / self.emb_size)).double().cuda()
        pe[..., 0::2] = torch.sin(position_ids * div)
        pe[..., 1::2] = torch.cos(position_ids * div)

        embeddings = input_dp + pe
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Embeddings_no_position(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embeddings_no_position, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_dp):
        words_embeddings = self.word_embeddings(input_dp)
        embeddings = self.LayerNorm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.bmm(adj, support)

        return output


class CNN_Add_Self(nn.Module):
    def __init__(self, dim, kernel):
        super(CNN_Add_Self, self).__init__()
        self.cnn = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel, padding='same')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.cnn(x)

        x = x.permute(0, 2, 1)
        return x


class CNN_No_Self(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(CNN_No_Self, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel, padding='same')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class HyperAttention(nn.Module):
    def __init__(self, dim, drug_max_len, protein_max_len):
        super(HyperAttention, self).__init__()
        self.attention_layer = nn.Linear(dim, dim)
        self.protein_attention_layer = nn.Linear(dim, dim)
        self.drug_attention_layer = nn.Linear(dim, dim)
        self.Drug_max_pool = nn.MaxPool1d(drug_max_len)
        self.Protein_max_pool = nn.MaxPool1d(protein_max_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

    def forward(self, drug_conv, protein_conv):
        drug_conv = drug_conv.permute(0, 2, 1)
        protein_conv = protein_conv.permute(0, 2, 1)

        drug_att = self.drug_attention_layer(drug_conv.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(protein_conv.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, protein_conv.shape[-1], 1)  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drug_conv.shape[-1], 1, 1)  # repeat along drug size
        att_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        compound_att = torch.mean(att_matrix, 2)
        protein_att = torch.mean(att_matrix, 1)
        compound_att = self.sigmoid(compound_att.permute(0, 2, 1))
        protein_att = self.sigmoid(protein_att.permute(0, 2, 1))

        drug_conv = drug_conv * 0.5 + drug_conv * compound_att
        protein_conv = protein_conv * 0.5 + protein_conv * protein_att

        drug_conv = self.Drug_max_pool(drug_conv).squeeze(2)
        protein_conv = self.Protein_max_pool(protein_conv).squeeze(2)

        drug_conv = self.ln_1(drug_conv)
        protein_conv = self.ln_2(protein_conv)

        return drug_conv, protein_conv


class HyperCNN(nn.Module):
    def __init__(self, dim, out_dim, kernels):
        super(HyperCNN, self).__init__()
        self.cnn_1 = nn.Sequential(
            CNN_Add_Self(dim, kernels[0]),
            nn.LeakyReLU(0.1),
            CNN_Add_Self(dim, kernels[0]),
            nn.LeakyReLU(0.1),
        )
        self.cnn_2 = nn.Sequential(
            CNN_Add_Self(dim, kernels[1]),
            nn.LeakyReLU(0.1),
            CNN_Add_Self(dim, kernels[1]),
            nn.LeakyReLU(0.1),
        )
        self.cnn_3 = nn.Sequential(
            CNN_Add_Self(dim, kernels[2]),
            nn.LeakyReLU(0.1),
            CNN_Add_Self(dim, kernels[2]),
            nn.LeakyReLU(0.1),
        )
        self.transformer_layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, 2)
        self.out = nn.Sequential(
            CNN_Add_Self(dim * 4, kernels[0]),
            nn.LeakyReLU(0.1),
            CNN_No_Self(dim * 4, out_dim, kernels[1]),
            nn.LeakyReLU(0.1),
        )

    def forward(self, embed_1, embed_2):
        cnn_1 = self.cnn_1(embed_1)
        cnn_2 = self.cnn_2(embed_1)
        cnn_3 = self.cnn_3(embed_1)
        encoded = self.encoder(embed_2)
        fusion = torch.concat([cnn_1, cnn_2, cnn_3, encoded], dim=2)
        fusion = self.out(fusion)
        return fusion


class Predictor(nn.Module):
    def __init__(self, hp, protein_max_len=1000, drug_max_len=100):
        super(Predictor, self).__init__()
        self.dim = hp.char_dim
        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel

        self.protein_embed = nn.Embedding(23, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        self.add_position = Add_position(self.dim)

        self.Drug_CNNs = HyperCNN(self.dim, self.dim, self.drug_kernel)
        self.Protein_CNNs = HyperCNN(self.dim, self.dim, self.protein_kernel)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(self.dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

        self.contactGCN = GCN(self.dim, self.dim)
        self.DrugGCN = GCN(34, self.dim)

        self.hyper_att = HyperAttention(self.dim * 2, drug_max_len, protein_max_len)

    def forward(self, contacts, node_feat, adj, drug, protein):
        drug_embed = self.drug_embed(drug)
        protein_embed = self.protein_embed(protein)
        drug_position_embed = self.add_position(drug_embed)
        protein_position_embed = self.add_position(protein_embed)

        drug_cnn = self.Drug_CNNs(drug_embed, drug_position_embed)
        protein_cnn = self.Protein_CNNs(protein_embed, protein_position_embed)

        drug_gcn = self.DrugGCN(node_feat, adj)
        protein_gcn = self.contactGCN(protein_embed, contacts)

        drug_feat = torch.concat([drug_cnn, drug_gcn], dim=2)
        protein_feat = torch.concat([protein_cnn, protein_gcn], dim=2)

        drug_vec, protein_vec = self.hyper_att(drug_feat, protein_feat)

        pair = torch.cat([drug_vec, protein_vec], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict
