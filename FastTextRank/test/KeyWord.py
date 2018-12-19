from FastTextRank.FastTextRank4Word import FastTextRank4Word
import codecs
import datetime
mod = FastTextRank4Word(tol=0.0001,window=2)
# with open("text1.txt", "r", encoding='utf-8') as myfile:
old_time = datetime.datetime.now()
for i in range(1):
    text = codecs.open('text' + str(i + 1) + '.txt', 'r', 'utf-8').read()
    print('摘要')
    old_time = datetime.datetime.now()
    print(mod.summarize(text, 3))
    po=mod.summarize(text, 3)
    import types
    print(type(po[0]))
    print(datetime.datetime.now() - old_time)