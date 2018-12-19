#-*- encoding:utf-8 -*-
import sys
import jieba
import math
import numpy as np
import jieba.posseg as pseg

sentence_delimiters=frozenset(u'。！？……')
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

PY2 = sys.version_info[0] == 2
if not PY2:
    # Python 3.x and up
    text_type    = str
    string_types = (str,)
    xrange       = range

    def as_text(v):  ## 生成unicode字符串
        if v is None:
            return None
        elif isinstance(v, bytes):
            return v.decode('utf-8', errors='ignore')
        elif isinstance(v, str):
            return v
        else:
            raise ValueError('Unknown type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

else:
    # Python 2.x
    text_type    = unicode
    string_types = (str, unicode)
    xrange       = xrange

    def as_text(v):
        if v is None:
            return None
        elif isinstance(v, unicode):
            return v
        elif isinstance(v, str):
            return v.decode('utf-8', errors='ignore')
        else:
            raise ValueError('Invalid type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

def cut_sentences(sentence):
    tmp = []
    for ch in sentence:  # 遍历字符串中的每一个字
        tmp.append(ch)
        if sentence_delimiters.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)

def cut_filter_words(cutted_sentences,stopwords,use_stopwords=False):
    sentences = []
    sents = []
    for sent in cutted_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in jieba.cut(sent) if word and word not in stopwords])  # 把句子分成词语
        else:
            sents.append([word for word in jieba.cut(sent) if word])
    return sentences,sents

def psegcut_filter_words(cutted_sentences,stopwords,use_stopwords=True,use_speech_tags_filter=True):
    sents = []
    sentences = []
    for sent in cutted_sentences:
        sentences.append(sent)
        jieba_result = pseg.cut(sent)
        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result if w.flag in allow_speech_tags]
        else:
            jieba_result = [w for w in jieba_result]
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]
        if use_stopwords:
            word_list = [word.strip() for word in word_list if word.strip() not in stopwords]
        sents.append(word_list)
    return  sentences,sents

def weight_map_rank(weight_graph,max_iter,tol):
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
    denominator = caculate_degree(weight_graph)

    # 开始迭代
    count=0
    while different(scores, old_scores,tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        #计算每个句子的分数
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph,denominator, i)
        count+=1
        if count>max_iter:
            break
    return scores

def caculate_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def calculate_score(weight_graph,denominator, i):#i表示第i个句子
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

def different(scores, old_scores,tol=0.0001):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:#原始是0.0001
            flag = True
            break
    return flag

def cosine_similarity(vec1, vec2):
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


def combine(word_list, window=2):
    """构造在window下的单词组合，用来构造单词之间的边。

    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2: window = 2
    for x in xrange(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def two_sentences_similarity(sents_1, sents_2):
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
    if counter==0:
        return 0
    return counter / (math.log(len(sents_1) + len(sents_2)))

