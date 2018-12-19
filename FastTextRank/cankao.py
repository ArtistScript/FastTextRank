import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
from gensim.models import Word2Vec
import numpy as np



class W2V_TextRank:
    def __init__(self):
        self.__model= Word2Vec.load(r'E:\BaiduNetdiskDownload\10G训练好的词向量\60维\Word60.model')
        np.seterr(all='warn')

    #根据yield把文章分成多个句子
    def cut_sentences(self,sentence):
        puns = frozenset(u'。！？……')   #不可变得集合
        tmp = []
        for ch in sentence:#遍历文章中的每一个字
            tmp.append(ch)
            if puns.__contains__(ch):
                yield ''.join(tmp)
                tmp = []
        yield ''.join(tmp)


    # 句子中的stopwords
    def create_stopwords(self):
        stop_list = [line.strip() for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]
        return stop_list


    def two_sentences_similarity(self,sents_1, sents_2):
        '''
        计算两个句子的相似性
        :param sents_1:
        :param sents_2:
        :return:
        '''
        counter = 0
        for sent in sents_1:
            if sent in sents_2:
                counter += 1
        return counter / (math.log(len(sents_1) + len(sents_2)))

    def cosine_similarity(self,vec1, vec2):
        '''
        计算两个向量之间的余弦相似度
        :param vec1:
        :param vec2:
        :return:
        '''
        tx = np.array(vec1)
        ty = np.array(vec2)
        cos1 = np.sum(tx * ty)
        cos21 = np.sqrt(sum(tx ** 2))
        cos22 = np.sqrt(sum(ty ** 2))
        cosine_value = cos1 / float(cos21 * cos22)
        return cosine_value

    def compute_similarity_by_avg(self,sents_1, sents_2):
        '''
        对两个句子求平均词向量
        :param sents_1:
        :param sents_2:
        :return:
        '''
        if len(sents_1) == 0 or len(sents_2) == 0:
            return 0.0
        #把一个句子中的所有词向量相加
        vec1 = self.__model[sents_1[0]]
        for word1 in sents_1[1:]:
            vec1 = vec1 + self.__model[word1]

        vec2 = self.__model[sents_2[0]]
        for word2 in sents_2[1:]:
            vec2 = vec2 + self.__model[word2]

        similarity = self.cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
        return similarity

    def create_graph(self,word_sent):
        """
        传入句子链表  返回句子之间相似度的图
        :param word_sent:
        :return:
        """
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                board[i][j] = self.compute_similarity_by_avg(word_sent[i], word_sent[j])
        return board





    def calculate_score(self,weight_graph,denominator, i):#i表示第i个句子
        """
        计算句子在图中的分数
        :param weight_graph:
        :param scores:
        :param i:
        :return:
        """
        length = len(weight_graph)
        d = 0.85
        added_score = 0.0

        for j in range(length):
            fraction = 0.0
            # 计算分子
            #[j,i]是指句子j指向句子i
            fraction = weight_graph[j][i] * 1.0
            #除以j的出度
            added_score += fraction / denominator[j]
        #算出最终的分数
        weighted_score = (1 - d) + d * added_score
        return weighted_score


    def weight_sentences_rank(self,weight_graph):
        '''
        输入相似度的图（矩阵)
        返回各个句子的分数
        :param weight_graph:
        :return:
        '''
        # 初始分数设置为0.5
        #初始化每个句子的分子和老分数
        scores = [0.5 for _ in range(len(weight_graph))]
        old_scores = [0.0 for _ in range(len(weight_graph))]

        length = len(weight_graph)
        denominator = [0.0 for _ in range(len(weight_graph))]
        for j in range(length):
            for k in range(length):
                denominator[j] += weight_graph[j][k]
            if denominator[j] == 0:
                denominator[j] = 1.0

        # 开始迭代
        while self.different(scores, old_scores):
            for i in range(len(weight_graph)):
                old_scores[i] = scores[i]
            #计算每个句子的分数
            for i in range(len(weight_graph)):
                scores[i] = self.calculate_score(weight_graph,denominator, i)
        return scores


    def different(self,scores, old_scores):
        '''
        判断前后分数有无变化
        :param scores:
        :param old_scores:
        :return:
        '''
        flag = False
        for i in range(len(scores)):
            if math.fabs(scores[i] - old_scores[i]) >= 0.0001:#原始是0.0001
                flag = True
                break
        return flag


    def filter_symbols(self,sents):
        stopwords = self.create_stopwords() + ['。', ' ', '.']
        _sents = []
        for sentence in sents:
            for word in sentence:
                if word in stopwords:
                    sentence.remove(word)
            if sentence:
                _sents.append(sentence)
        return _sents


    def filter_model(self,sents):
        _sents = []
        dele=set()
        for sentence in sents:
            for word in sentence:
                if word not in self.__model:
                    dele.add(word)
            if sentence:
                _sents.append([word for word in sentence if word not in dele])
        return _sents


    def summarize(self,text, n):
        tokens = self.cut_sentences(text) #把文章分成多个句子
        sentences = []
        sents = []
        for sent in tokens:
            sentences.append(sent)
            sents.append([word for word in jieba.cut(sent) if word]) #把句子分成词语

        # sents = filter_symbols(sents)
        #提取在词典里不存在的单词
        sents = self.filter_model(sents)
        #根据句子与句子的相似性构建一幅图
        graph = self.create_graph(sents)

        #计算每个句子的重要程度
        scores = self.weight_sentences_rank(graph)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])#添加入关键词在原来文章中的下标
        return [sentences[i] for i in sent_index]

import codecs
import datetime
if __name__ == '__main__':
    mod = W2V_TextRank()
    # with open("text1.txt", "r", encoding='utf-8') as myfile:
    old_time = datetime.datetime.now()
    for i in range(7):
        text = codecs.open('text' + str(i + 1) + '.txt', 'r', 'utf-8').read()
        print('摘要')
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        old_time = datetime.datetime.now()
        print(mod.summarize(text, 1)[0])
        po=mod.summarize(text, 1)
        import types
        print(type(po[0]))
        print(datetime.datetime.now() - old_time)

