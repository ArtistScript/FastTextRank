from W2VTextRank.W2VTextRank4Sentence import W2VTextRank4Sentence
import codecs
import datetime
mod = W2VTextRank4Sentence(use_w2v=False,tol=0.0001)
# with open("text1.txt", "r", encoding='utf-8') as myfile:
old_time = datetime.datetime.now()
for i in range(10):
    text = codecs.open('text' + str(i + 1) + '.txt', 'r', 'utf-8').read()
    print('摘要'+str(i+1)+':')
    old_time = datetime.datetime.now()
    print(mod.summarize(text, 1))
    po=mod.summarize(text, 1)
    print(datetime.datetime.now() - old_time)