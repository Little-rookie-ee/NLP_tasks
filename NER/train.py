import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from dataset_processing import normal
from word2vec_train_load import train_word2vec, load_word2vec

#os.environ['CUDA_VI SIBLE_DEVICES'] = '0'


def arg_max(vec):
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()
def log_sum_exp(vec):
	max_score = vec[0][arg_max(vec)]  ## max_score的维度为1
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
	#等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_idx):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_idx = tagset_idx
        self.tagset_len = len(tagset_idx)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_len)

        ## 转移矩阵，transitions[i][j]表示从label_j转移到label_i的概率,虽然是随机生成的但是后面会迭代更新
        self.transitions = torch.randn(self.tagset_len, self.tagset_len).cuda()
        self.transitions[tagset_idx['start'], :] = -10000  ## 不可能从任何标签转移到'start'
        self.transitions[:, tagset_idx['end']] = -10000  ## 不可能从'end'转移到任何标签

    def get_lstm_features(self, x):
        lstm_out, self.hidden = self.lstm(x)
        #lstm_out = lstm_out.permute(1, 0, 2)  ## 将输出seq_len * batch_size * hidden_dim 装换成batch_size * seq_len * hidden_dim
        #print(lstm_out.shape)
        lstm_feats = self.hidden2tag(lstm_out)       ## 全连接是否能进行三维数据的输出，输出为batch_size * seq_len * hidden_dim
        return lstm_feats

    def forward_alg(self, feats, seq_lens):
        '''
        输入：发射矩阵(emission score)，实际上就是LSTM的输出——sentence的每个word经BiLSTM后，对应于每个label的得分
        输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        '''
        forward_score = torch.zeros(1).cuda()
        for i in range(len(feats)):
            init_alphas = torch.full((1, self.tagset_len), -10000.).cuda()
            init_alphas[0][self.tagset_idx['start']] = 0.

            # 包装到一个变量里面以便自动反向传播
            forward_var = init_alphas
            for j in range(seq_lens[i]):
                alphas_t = []
                for tag_id in range(self.tagset_len):
                    emiss_score = feats[i, j, tag_id].view(1, -1).expand(1, self.tagset_len)
                    trans_score = self.transitions[tag_id].view(1, -1)
                    next_tag_var = forward_var + emiss_score + trans_score
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            # 最后将最后一个单词的forward var与转移 stop tag的概率相加
            terminal_var = forward_var + self.transitions[self.tagset_idx['end']]
            alpha = log_sum_exp(terminal_var)
            forward_score += alpha
        return forward_score

    def right_path_score(self, feats, labels, seq_len):
        scores = torch.zeros(1).cuda()
        for i in range(len(seq_len)):
            s = torch.zeros(1).cuda()
            ## 将labels截取它句子的长度，将‘start’标签对应的索引添加到labels最前面
            labs = labels[i, :seq_len[i]]
            labs = torch.cat([torch.tensor([self.tagset_idx['start']], dtype=torch.long).cuda(), labs])
            for j in range(seq_len[i]):
                s += self.transitions[labs[j+1], labs[j]] + feats[i, j, labs[j]]
            s += self.transitions[self.tagset_idx['end'], labs[-1]]
            scores += s
        return scores

    def viterbi_decode(self, feats):
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []

        init_vvars = torch.full((1, self.tagset_len), -10000.).cuda()
        init_vvars[0][self.tagset_idx['start']] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_len):
                next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
                best_tag_id = arg_max(next_tag_var)  # 返回最大值对应的那个tag
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tagset_idx['end']]
        best_tag_id = arg_max(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 无需返回最开始的START位
        start = best_path.pop()
        assert start == self.tagset_idx['start']
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path


    def get_loss(self, texts, labels, seq_len):
        feats = self.get_lstm_features(texts)  ## 输出的是获得的各字的预测标签概率，即feats = emissions scores，大小batch_size * seq_len * hidden_dim
        forward_score = self.forward_alg(feats, seq_len)  ## 是整个batch的所有路径score
        rightpath_score = self.right_path_score(feats, labels, seq_len)
        return abs(forward_score - rightpath_score)  ## forward_score的计算方法可能到后边会小于rightpath_score,因为他不是一条一条路径计算的score的，而是计算整体的score，而有些路径是负值，导致出问题，
                                                     ## 一条一条计算的话，即使是负值，由于做了一个exp(k),会使得结果是正的，forward_score = log(exp(path1) + exp(path2) + ... + exp(pathN))


    def forward(self, texts, seq_len):
        feats = self.get_lstm_features(texts)
        scores = []
        paths = []
        for i in range(len(seq_len)):
            feat = feats[i, :seq_len[i]]
            score, pre_path = self.viterbi_decode(feat)
            scores.append(score)
            paths.append(pre_path)
        return scores, paths


Hidden_dim = 64  ## LSTM的隐藏层的维数
Embedding_dim = 100  ## 输入LSTM中词向量的维度
max_len = 50  ## 文本字最大数量
batch_size = 64  ## 一个批次32个数据输入
Epoch = 30  ## 训练迭代次数

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('正在计算的是：%s'%device)

tagset_idx = {'O': 0, 'start': 1, 'end': 2,
              'address': 3, 'address-I': 4, 'book': 5, 'book-I': 6, 'company': 7, 'company-I': 8,
              'game': 9, 'game-I': 10, 'government': 11, 'government-I': 12, 'movie': 13, 'movie-I': 14,
              'name': 15, 'name-I': 16, 'organization': 17, 'organization-I': 18, 'position': 19, 'position-I': 20,
              'scene': 21, 'scene-I': 22}

if __name__ == '__main__':
    train_texts, train_labels = normal(r'F:\python_data\practice\NLP tasks\NER\dataset\train.json')
    test_texts, test_labels = normal(r'F:\python_data\practice\NLP tasks\NER\dataset\dev.json')

    ## 训练字向量模型
    if not os.path.exists('word2vec.model'):
        total_texts = train_texts + test_texts
        train_word2vec(total_texts)

    train_labels = torch.tensor(train_labels, dtype=torch.long)  ## labels的最后一位为句子长度，即[num_text, max_len(50) + real_seq_len(1)]
    test_labels = torch.tensor(test_labels, dtype=torch.long)  ## 标签用dtype=torch.long
    ## 获取输入文本的向量矩阵表示
    train_texts = torch.tensor(load_word2vec(train_texts), dtype=torch.float32)  ## 训练数据用torch.float32
    test_texts = torch.tensor(load_word2vec(test_texts), dtype=torch.float32)
    ## 创建Tensor Datases
    train_data = TensorDataset(train_texts, train_labels)
    test_data = TensorDataset(test_texts, test_labels)
    ## 打乱数据，需要注意里面的长度信息
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    model = BiLSTM_CRF(Embedding_dim, Hidden_dim, tagset_idx)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1):
        correct = 0
        total = 0
        epoch_loss = 0
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            seq_lens = labels[:, max_len].cuda()  ## labels的最后一位是句子长度
            labels = labels[:, :max_len].cuda()

            optimizer.zero_grad()
            loss = model.get_loss(data, labels, seq_lens)

            loss.backward()
            optimizer.step()
            print('epoch:%s' % epoch, 'batch_idx:%s' % batch_idx, 'loss = %s' % loss.item())

    for epoch in range(1):
        correct = 0
        total = 0
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.cuda()
            seq_lens = labels[:, max_len].cuda()  ## labels的最后一位是句子长度
            labels = labels[:,:max_len]

            scores, paths = model(data, seq_lens)
            print(paths[0])
            print(labels[0, :seq_lens[0]])

            for i in range(len(labels)):
                correct += int(torch.sum(torch.tensor(paths[i]) == labels[i, :seq_lens[i]]))
                total += seq_lens[i]

        print('test_data accuracy：%.3f%%' % (correct * 100 / total))


    '''
    for epoch in range(Epoch):
        correct = 0
        total = 0
        epoch_loss = 0
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            seq_lens = labels[:, max_len].cuda()  ## labels的最后一位是句子长度
            labels = labels[:,:max_len].cuda()


            optimizer.zero_grad()
            predict = model.get_lstm_features(data)

            for i in range(len(labels)):
                if i == 0:
                    loss = F.cross_entropy(predict[i], labels[i])
                else:
                    loss += F.cross_entropy(predict[i], labels[i])

            correct += int(torch.sum(torch.argmax(predict, dim=2) == labels))
            total += len(labels) * max_len
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
        loss = epoch_loss / (batch_idx + 1)
        print('epoch:%s' % epoch, 'accuracy：%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)

    for epoch in range(1):
        correct = 0
        total = 0
        epoch_loss = 0
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.cuda()
            seq_lens = labels[:, max_len].cuda()  ## labels的最后一位是句子长度
            labels = labels[:,:max_len].cuda()

            predict = model.get_lstm_features(data)

            for i in range(len(labels)):
                if i == 0:
                    loss = F.cross_entropy(predict[i], labels[i])
                else:
                    loss += F.cross_entropy(predict[i], labels[i])

            correct += int(torch.sum(torch.argmax(predict, dim=2) == labels))
            total += len(labels) * max_len
            epoch_loss += loss.item()

        loss = epoch_loss / (batch_idx + 1)
        print('test_data accuracy：%.3f%%' % (correct * 100 / total), 'loss = %s' % loss)
        '''





