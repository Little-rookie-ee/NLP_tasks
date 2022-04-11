from gensim.models import Word2Vec
import numpy as np

max_len = 50
embedding_dim = 100
def train_word2vec(texts):
    ## 训练语料的词向量模型，用单个字训练
    ## 读取已经保存在矩阵中的语料，语料保存形式为 [['彭', '小', '军', '认', '为', '，', '国', '内', ...，'],.......]

    #texts = [list(t) for t in texts]
    model = Word2Vec(texts, size=100, window=5, min_count=1)
    model.save('word2vec.model')
    #k = Word2Vec.load('word2vec.model')

def load_word2vec(texts):
    ## 将文本装换成向量矩阵

    model = Word2Vec.load('word2vec.model')
    texts_arr = []
    for text in texts:
        text_vec = []
        for word in text:
            text_vec.append(list(model[word]))
        texts_arr.append(text_vec + [[0] * embedding_dim] * (max_len - len(text_vec)))

    return texts_arr



if __name__ == '__main__':
    train_word2vec(['彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，'])