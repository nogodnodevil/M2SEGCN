import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import math
from entmax import entmax_bisect

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class MLPs(nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLPs, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        return emb_trans

class ItemConv(nn.Module):
    def __init__(self, layers, emb_size):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size, bias=False) for i in range(self.layers)])

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item[i])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = torch.stack(final, dim=0)
        item_embeddings = torch.mean(item_embeddings, dim=0)
        return item_embeddings

class M2SEGCN(nn.Module):
    def __init__(self, opt, n_node, train_data):
        super(M2SEGCN, self).__init__()
        self.n_node = n_node
        self.dataset = opt.dataset
        self.emb_size = opt.emb_size
        self.img_emb_size = opt.img_emb_size
        self.text_emb_size = opt.text_emb_size
        self.batch_size = opt.batch_size
        self.l2 = opt.l2
        self.lr = opt.lr
        self.dropout = opt.dropout
        self.layers = opt.layers
        self.t_1 = opt.t_1
        self.t_2 = opt.t_2

        self.w_k = 10
        self.adjacency = train_data.adjacency
        self.img_adjacency = train_data.img_adj
        self.text_adjacency = train_data.text_adj

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(2000, self.emb_size)

        self.ItemGraph = ItemConv(self.layers, self.emb_size)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.mlp_mix_trans = MLPs(self.emb_size * 3, self.emb_size, dropout=self.dropout)

        self.init_parameters()

        # load image&text embeddings
        self.image_embedding = nn.Embedding.from_pretrained(torch.from_numpy(train_data.img_weights_pca).float(), freeze=True)
        self.image_embedding_ori = nn.Embedding.from_pretrained(torch.from_numpy(train_data.img_weights).float(), freeze=True)

        self.text_embedding = nn.Embedding.from_pretrained(torch.from_numpy(train_data.text_weights_pca).float(), freeze=True)
        self.text_embedding_ori = nn.Embedding.from_pretrained(torch.from_numpy(train_data.text_weights).float(), freeze=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)

        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)

        # 自适应权重聚合
        order = torch.zeros((self.batch_size, len), dtype=torch.int32).cuda()
        for i in range(self.batch_size):
            length = session_len[i, 0]
            order[i, :length] = torch.arange(length, 0, -1)
        t_1 = self.t_1
        t_2 = self.t_2
        new_order = torch.exp(order / t_2)
        last_item = seq_h[:,0,:].unsqueeze(1).repeat(1, len, 1)
        cs = torch.cosine_similarity(seq_h, last_item, dim=2)
        torch.exp(cs * t_1)

        weights = torch.mul(new_order, cs)
        final_weights = F.softmax(torch.where(weights!=0, weights, -9e10*torch.ones_like(weights)),dim=1)
        session_aw = torch.sum(torch.mul(final_weights.unsqueeze(2).repeat(1, 1, self.emb_size), seq_h), dim=1)

        sess_emb = select + session_aw
        k_v = 3
        cos_sim = self.compute_sim(sess_emb)
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice]
        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.emb_size)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = F.dropout(neighbor_sess, self.dropout)

        return select + session_aw + F.normalize(neighbor_sess)

    def compute_sim(self, sess_emb):
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    def forward(self, session_item, session_len, reversed_sess_item, mask):
        item_emb = self.embedding.weight
        image_emb = self.image_embedding.weight
        text_emb = self.text_embedding.weight

        item_emb = self.ItemGraph(self.adjacency, item_emb)

        image_emb_i = self.ItemGraph(self.adjacency, image_emb)
        text_emb_i = self.ItemGraph(self.adjacency, text_emb)
        image_emb_m = self.ItemGraph(self.img_adjacency, image_emb)
        text_emb_m = self.ItemGraph(self.text_adjacency, text_emb)
        image_emb = image_emb_i + image_emb_m
        text_emb = text_emb_i + text_emb_m

        mix_emb = torch.cat((item_emb, image_emb, text_emb), dim=-1)
        mix_emb = self.mlp_mix_trans(mix_emb)

        sess_emb_i = self.generate_sess_emb(mix_emb, session_item, session_len, reversed_sess_item, mask)
        sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
        item_emb = F.normalize(item_emb, dim=-1, p=2)
        scores_item = torch.mm(sess_emb_i, torch.transpose(item_emb, 1, 0))

        return scores_item

def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    scores_item = model(session_item, session_len, reversed_sess_item, mask)
    return tar, scores_item

def train_test(model, train_data, test_data, f):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    model.train()
    for i in tqdm(slices):
        model.zero_grad()
        targets, scores = forward(model, i, train_data)
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tLoss:\t%.3f' % total_loss, file=f, flush=True)

    print('start predicting: ', datetime.datetime.now())
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    slices = test_data.generate_batch(model.batch_size)
    model.eval()
    for i in tqdm(slices):
        tar, scores = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        tar = trans_to_cpu(tar).detach().numpy()
        index = np.argsort(-scores, 1)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))

    return metrics, total_loss