#-*- encoding:utf-8 -*-
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
from gensim.models import Word2Vec
from FastTextRank import util
import numpy as np
import os
import codecs
from itertools import count

class FastTextRank4Sentence(object):
    def __init__(self, use_stopword = False, stop_words_file=None, use_w2v=False, dict_path=None,max_iter=100,tol=0.0001):
        """

        :param use_stopword: 是否使用停用词
        :param stop_words_file: 停用词文件路径
        :param use_w2v: 是否使用词向量计算句子相似性
        :param dict_path: 词向量字典文件路径
        :param max_iter: 最大迭代伦茨
        :param tol: 最大容忍误差
        """
        if use_w2v==False and dict_path!=None:
            raise RuntimeError("再使用词向量之前必须令参数use_w2v=True")
        self.__use_stopword = use_stopword
        self.__use_w2v=use_w2v
        self.__dict_path = dict_path
        self.__max_iter = max_iter
        self.__tol = tol
        if self.__use_w2v:
            self.__word2vec = Word2Vec.load(self.__dict_path)
        self.__stop_words = set()
        self.__stop_words_file = self.get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.__stop_words_file = stop_words_file
        for word in codecs.open(self.__stop_words_file, 'r', 'utf-8', 'ignore'):
            self.__stop_words.add(word.strip())
        np.seterr(all='warn')#Print a RuntimeWarning for all types of floating-point errors

    def get_default_stop_words_file(self):
        d = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(d, 'stopwords.txt')

    #可以改进为删除停用词，词性不需要的词
    def filter_dictword(self,sents):
        """
        删除词向量字典里不存的词
        :param sents:
        :return:
        """
        _sents = []
        dele=set()
        for sentence in sents:
            for word in sentence:
                if word not in self.__word2vec:
                    dele.add(word)
            if sentence:
                _sents.append([word for word in sentence if word not in dele])
        return _sents

    def summarize(self,text,n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = util.as_text(text)#处理编码问题
        tokens=util.cut_sentences(text)
        #sentences用于记录文章最原本的句子，sents用于各种计算操作
        sentences, sents=util.cut_filter_words(tokens,self.__stop_words,self.__use_stopword)
        if self.__use_w2v:
            sents = self.filter_dictword(sents)
        graph = self.create_graph_sentence(sents,self.__use_w2v)
        scores = util.weight_map_rank(graph,self.__max_iter,self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])  # 添加入关键词在原来文章中的下标
        return [sentences[i] for i in sent_index]

    def create_graph_sentence(self,word_sent, use_w2v):
        """
        传入句子链表  返回句子之间相似度的图
        :param word_sent:
        :return:
        """
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                if use_w2v:
                    board[i][j] = self.compute_similarity_by_avg(word_sent[i], word_sent[j])
                else:
                    board[i][j]=util.two_sentences_similarity(word_sent[i], word_sent[j])
        return board

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
        vec1 = self.__word2vec[sents_1[0]]
        for word1 in sents_1[1:]:
            vec1 = vec1 + self.__word2vec[word1]

        vec2 = self.__word2vec[sents_2[0]]
        for word2 in sents_2[1:]:
            vec2 = vec2 + self.__word2vec[word2]

        similarity = util.cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
        return similarity


