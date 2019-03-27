
# coding: utf-8

# In[1]:

import json
path = "final.json"

train=[]

with open(path, 'r') as f:
    for line in f:
        temp = json.loads(line.strip())
        train.append(temp)


# In[2]:

import json
path = "final_test.json"

test=[]

with open(path, 'r') as f:
    for line in f:
        temp = json.loads(line.strip())
        test.append(temp)


# In[3]:

import nltk
nltk.download('opinion_lexicon')
nltk.download('punkt')


# In[4]:

vector_size=100
ratio_=0.3


# In[5]:

###extra top5000 words, top 10000 bigram, top 10000 trigram
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
top_words={}
top_bigrams={}
top_trigrams={}
for each in train:
    sent_tokenize_list = sent_tokenize(each["text"])
    for sen in sent_tokenize_list:
        words = word_tokenize(sen)
        words = list(map(lambda x:x.lower(),words))
        words_len=len(words)        
        for i,word in enumerate(words):
            if word in top_words:
                top_words[word]+=1
            else:
                top_words[word]=1
            if i+1<words_len:
                if words[i]+" "+words[i+1] in top_bigrams:
                    top_bigrams[words[i]+" "+words[i+1]]+=1
                else:
                    top_bigrams[words[i]+" "+words[i+1]]=1
            if i+2<words_len:
                if words[i]+" "+words[i+1]+" "+words[i+2] in top_trigrams:
                    top_trigrams[words[i]+" "+words[i+1]+" "+words[i+2]]+=1
                else:
                    top_trigrams[words[i]+" "+words[i+1]+" "+words[i+2]]=1

words_ = list(map(lambda x:x[0],sorted(top_words.items(),key=lambda x:x[1],reverse = True)))
birgrams_ = list(map(lambda x:x[0],sorted(top_bigrams.items(),key=lambda x:x[1],reverse = True)))
trigrams_ = list(map(lambda x:x[0],sorted(top_trigrams.items(),key=lambda x:x[1],reverse = True)))
top_dict={}
count=1
for each in words_:
    top_dict[each]=count
    count+=1
    if count>5000:
        break
for each in birgrams_:
    top_dict[each]=count
    count+=1
    if count>15000:
        break
for each in trigrams_:
    top_dict[each]=count
    count+=1
    if count>20000:
        break


# In[6]:

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon

positive_set=set(opinion_lexicon.positive())
negative_set=set(opinion_lexicon.negative())

import numpy as np
def avg_vec(sen_list,lexicon,vector):
    count=0
    sec_vec = np.array([0.0 for i in range(vector_size)])
    for sen in sen_list:
        words = word_tokenize(sen)
        for word in sen:
            count+=1
            if word in lexicon:
                vec = np.array(vector[data[word]])
                sec_vec+=vec
            else:
                count-=1
    return sec_vec/count
    
def count_vec(sen_list):
    vec1=[]
    vec2=[]
    vec3=[]      
    vec=[]
    for sen in sen_list:
        words = word_tokenize(sen)
        words = list(map(lambda x:x.lower(),words))
        words_len = len(words)
        for i,word in enumerate(words):
            bigram=""  
            trigram=""
            if i+1<words_len:
                bigram=words[i]+" "+words[i+1]
            if i+2<words_len:
                trigram=words[i]+" "+words[i+1]+" "+words[i+2]
            if word in top_dict:
                vec1.append(top_dict[word])
            if bigram in top_dict:
                vec2.append(top_dict[bigram])
            if trigram in top_dict:
                vec3.append(top_dict[trigram])
    return vec2+vec1+vec3
                

def text_to_vec(sentenses,lexicon,vector):
    sent_tokenize_list = sent_tokenize(sentenses)
    return avg_vec(sent_tokenize_list,lexicon,vector)

def text_to_vec_count(sentenses):
    sent_tokenize_list = sent_tokenize(sentenses)
    return count_vec(sent_tokenize_list)

def text_to_pos(sentenses):
    sen_list = sent_tokenize(sentenses)
    res = []
    for sen in sen_list:
        words = words = word_tokenize(sen)
        res.append(nltk.pos_tag(words))
    return res

##simple count
def text_to_sentiment(sentenses):
    sen_list = sent_tokenize(sentenses)
    pos_words = 0
    neg_words = 0
    for sen in sen_list:
        words = words = word_tokenize(sen)
        for word in words:
            if word in positive_set:
                pos_words+=1
            if word in negative_set:
                neg_words+=1
    if pos_words>neg_words:
        return 1
    elif pos_words<neg_words:
        return -1
    else:
        return 0

##count and ratio
def text_to_sentiment_ratio(sentenses):
    sen_list = sent_tokenize(sentenses)
    pos_words = 0
    neg_words = 0
    count = 0
    for sen in sen_list:
        words = word_tokenize(sen)        
        for word in words:
            count+=1
            if word in positive_set:
                pos_words+=1
                #print(word,"pos")
            if word in negative_set:
                neg_words+=1
                #print(word,"neg")
    if pos_words+neg_words==0:
        return 0
    score = (pos_words-neg_words)*1.0/(pos_words+neg_words)
    if score>ratio_:
        return 1
    elif score<ratio_:
        return -1
    else:
        return 0


# In[7]:

y_train = list(map(lambda x:x["stars"],train))
def mapper(x):
    if x>=5:
        return 1
    elif x<=2:
        return -1
    else: 
        return 0
y_train = list(map(lambda x:mapper(x),y_train))
y_train = np.array(y_train)
np.save("y_train",y_train)


# In[8]:

y_test = list(map(lambda x:x["stars"],test))
def mapper(x):
    if x>=5:
        return 1
    elif x<=2:
        return -1
    else: 
        return 0
y_test = list(map(lambda x:mapper(x),y_test))
y_test = np.array(y_test)
np.save("y_test",y_test)


# In[9]:

###X_train_count
X_train = list(map(lambda x:text_to_vec_count(x["text"]),train))
X_train = np.array(X_train)
np.save('X_train_count',X_train)


# In[10]:

###X_test_count
X_test = list(map(lambda x:text_to_vec_count(x["text"]),test))
X_test = np.array(X_test)
np.save('X_test_count',X_test)


# In[11]:

X_train = list(map(lambda x:text_to_sentiment_ratio(x["text"]),train))
X_train = np.array(X_train)
np.save('X_train_sentiment_ratio',X_train)


# In[12]:

X_test = list(map(lambda x:text_to_sentiment_ratio(x["text"]),test))
X_test = np.array(X_test)
np.save('X_test_sentiment_ratio',X_test)


# In[13]:

X_train = list(map(lambda x:text_to_sentiment(x["text"]),train))
X_train = np.array(X_train)
np.save('X_train_sentiment',X_train)


# In[14]:

X_test = list(map(lambda x:text_to_sentiment(x["text"]),test))
X_test = np.array(X_test)
np.save('X_test_sentiment',X_test)


# In[15]:

###word2vec
def load_dict(path):
    data = {}
    vectors=[]
    with open(path,"r") as f:
        count=0
        for line in f.readlines():
            l = line.split(" ")
            word = l[0]
            vector = list(map(lambda x:float(x),l[1:]))
            
            data[word]=count
            vectors.append(vector)
            count+=1
    return data, vectors
path="glove.twitter.27B.100d.txt"
data,vectors = load_dict(path)

X_train = list(map(lambda x:text_to_vec(x["text"],data,vectors),train))
X_train=np.array(X_train)
np.save('X_train_100d',X_train)
X_test = list(map(lambda x:text_to_vec(x["text"],data,vectors),test))
X_test=np.array(X_test)
np.save('X_test_100d',X_test)


# In[ ]:




# In[ ]:



