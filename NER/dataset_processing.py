import json
from gensim.models import Word2Vec
from word2vec_train_load import train_word2vec, load_word2vec
import numpy as np
## 将数据与标签，标准化

def label_normal(labels, texts_len, max_len):
    ## 将标签转换成BIO形式
    tagset_dic = {'O': 0, 'start': 1, 'end': 2,
           'address': 3, 'address-I': 4, 'book': 5, 'book-I': 6, 'company': 7, 'company-I': 8,
           'game': 9, 'game-I': 10, 'government': 11, 'government-I': 12, 'movie': 13, 'movie-I': 14,
           'name': 15, 'name-I': 16, 'organization': 17, 'organization-I': 18, 'position': 19, 'position-I': 20,
           'scene': 21, 'scene-I':22}
    ## labels:[{'address': {'台湾': [[15, 16]]}, 'name': {'彭小军': [[0, 2]]}},.....]
    labs = []
    le = len(texts_len)
    for i in range(le):
        lab = [0] * texts_len[i]
        for bq, d in labels[i].items():
            for posi in d.values():
                lab[posi[0][0]] = tagset_dic[bq]
                for po in range(posi[0][0]+1, posi[0][1]+1):
                    lab[po] = tagset_dic[bq]+1
        labs.append(lab + [0] * (max_len - len(lab)) + [texts_len[i]])  ## 将句子长度放在labels最后一位
    return labs

def normal(path):
    ## 读取json数据，并标准化标签
    r = open(path, encoding = 'utf-8')
    datas = []
    for line in r.readlines():
        datas.append(json.loads(line))
    texts = []
    texts_len = []
    labs = []
    for data in datas:
        texts.append(list(data['text']))
        texts_len.append(len(data['text']))
        labs.append(data['label'])
    labels = label_normal(labs, texts_len, max(texts_len)) ## 大小为 (数据条数， 最大文本长度--50)
    return texts, labels

if __name__ == '__main__':
    train_texts, train_labels = normal(r'F:\python_data\practice\NLP tasks\NER\dataset\train.json')
    test_texts, test_labels = normal(r'F:\python_data\practice\NLP tasks\NER\dataset\dev.json')

    ## 训练字向量模型
    #total_texts = train_texts + test_texts
    #train_word2vec(total_texts)
    #m = Word2Vec.load('word2vec.model')

    ## 获取输入文本的向量矩阵表示
    train_texts = load_word2vec(train_texts)
    test_texts = load_word2vec(test_texts)
    print(test_labels[0])
    print(np.array(test_labels).shape)



