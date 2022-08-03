#terminal에서 pip install treform 

import pandas as pd
import treform as ptm

df = pd.read_csv('C:/Users/yejin/PycharmProjects/lec-text-mining/sample_data/papers_new.csv').fillna("")
df['text'] = df['text'].str.lower()
corpus = df.iloc[:,1]


pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file='../stopwords/Stopword_Eng_Blockchain.txt'))
result = pipeline.processCorpus(corpus)

df['text'] = result


import pickle
with open('C:/Users/yejin/PycharmProjects/lec-text-mining/sample_data/pre_papers.pkl', "wb") as file:
    pickle.dump(df, file)

#어떻게 생겼나 확인 
df1 = pd.read_pickle('C:/Users/yejin/PycharmProjects/lec-text-mining/sample_data/pre_papers.pkl')
#print(df1)
print(df1.head(40))


# import re
# documents = []
# for doc in result:
#     for sent in doc:
#         sentence = ' '.join(sent)
#         sentence = re.sub('[^A-Za-z0-9가-힣_ ]+', '', sentence)
#         sentence = sentence.strip()
#         print(sentence)
#         if len(sentence) > 0:
#             documents.append(sentence)
#
# print(len(documents))
# co = ptm.cooccurrence.CooccurrenceWorker()
# co_result, vocab = co(documents)
