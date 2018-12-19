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
from itertools import count
import codecs

class FastTextRank4Word(object):
    def __init__(self,use_stopword=True,stop_words_file=None,max_iter=100,tol=0.0001,window=2):
        """
        :param max_iter: 最大的迭代轮次
        :param tol: 最大的容忍误差
        :param window: 词语窗口
        :return:
        """
        self.__use_stopword = use_stopword
        self.__max_iter = max_iter
        self.__tol = tol
        self.__window = window
        self.__stop_words = set()
        self.__stop_words_file = self.get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.__stop_words_file = stop_words_file
        for word in codecs.open(self.__stop_words_file, 'r', 'utf-8', 'ignore'):
            self.__stop_words.add(word.strip())
        # Print a RuntimeWarning for all types of floating-point errors
        np.seterr(all='warn')

    def get_default_stop_words_file(self):
        d = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(d, 'stopwords.txt')

    def build_worddict(self,sents):
        """
        构建字典，是词语和下标之间生成一对一的联系，为之后的词图构建做准备
        :param sents:
        :return:
        """
        word_index = {}
        index_word = {}
        words_number = 0
        for word_list in sents:
            for word in word_list:
                if not word in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1
        return word_index,index_word,words_number

    def build_word_grah(self,sents,words_number,word_index,window=2):
        graph = [[0.0 for _ in range(words_number)] for _ in range(words_number)]
        for word_list in sents:
            for w1, w2 in util.combine(word_list, window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] += 1.0
                    graph[index2][index1] += 1.0
        return graph

    def summarize(self,text,n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = util.as_text(text)#处理编码问题
        tokens=util.cut_sentences(text)
        #sentences用于记录文章最原本的句子，sents用于各种计算操作
        sentences,sents=util.psegcut_filter_words(tokens,self.__stop_words,self.__use_stopword)

        word_index, index_word, words_number=self.build_worddict(sents)
        graph=self.build_word_grah(sents,words_number,word_index,window=self.__window)
        scores = util.weight_map_rank(graph,max_iter=self.__max_iter,tol=self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])  # 添加入关键词在原来文章中的下标
        return [index_word[i] for i in sent_index]


