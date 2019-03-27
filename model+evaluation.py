
# coding: utf-8

# In[2]:

# baseline estimate with ratio 
import numpy as np
from prettytable import PrettyTable
same = [0,0,0]
truth = [0,0,0]
predict = [0,0,0]
X_train=np.load('X_test_sentiment_ratio.npy')
y_train=np.load('y_test.npy')
for i,each in enumerate(X_train):
    if (each==1 and y_train[i]==1):
        same[0]+=1
    elif (each==-1 and y_train[i]==-1):
        same[1]+=1
    elif (each==0 and y_train[i]==0):
        same[2]+=1
    if y_train[i]==1:
        truth[0] += 1
    elif y_train[i]==-1:
        truth[1]+=1
    elif y_train[i]==0:
        truth[2]+=1
    if each==1:
        predict[0]+=1
    elif each==-1:
        predict[1]+=1
    elif each==0:
        predict[2]+=1
t = PrettyTable(['class', 'precision','recall','F1 score'])
recall_sum = 0
precision_sum = 0
f1_sum = 0
recall = 1.0*same[0]/truth[0]
precision = 1.0*same[0]/predict[0]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['positive', precision,recall,f1])
recall = 1.0*same[1]/truth[1]
precision = 1.0*same[1]/predict[1]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['negative', precision,recall,f1])
recall = 1.0*same[2]/truth[2]
precision = 1.0*same[2]/predict[2]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['neutral', precision,recall,f1])
t.add_row(['average', precision_sum/3, recall_sum/3,f1_sum/3 ])
print t


# In[3]:

# baseline estimate without ratio 
import numpy as np
from prettytable import PrettyTable
same = [0,0,0]
truth = [0,0,0]
predict = [0,0,0]
X_train=np.load('X_test_sentiment.npy')
y_train=np.load('y_test.npy')
for i,each in enumerate(X_train):
    if (each==1 and y_train[i]==1):
        same[0]+=1
    elif (each==-1 and y_train[i]==-1):
        same[1]+=1
    elif (each==0 and y_train[i]==0):
        same[2]+=1
    if y_train[i]==1:
        truth[0] += 1
    elif y_train[i]==-1:
        truth[1]+=1
    elif y_train[i]==0:
        truth[2]+=1
    if each==1:
        predict[0]+=1
    elif each==-1:
        predict[1]+=1
    elif each==0:
        predict[2]+=1
t = PrettyTable(['class', 'precision','recall','F1 score'])
recall_sum = 0
precision_sum = 0
f1_sum = 0
recall = 1.0*same[0]/truth[0]
precision = 1.0*same[0]/predict[0]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['positive', precision,recall,f1])
recall = 1.0*same[1]/truth[1]
precision = 1.0*same[1]/predict[1]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['negative', precision,recall,f1])
recall = 1.0*same[2]/truth[2]
precision = 1.0*same[2]/predict[2]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['neutral', precision,recall,f1])
t.add_row(['average', precision_sum/3, recall_sum/3,f1_sum/3 ])
print t


# In[11]:

# LSTM model result evaluation
import numpy as np
from prettytable import PrettyTable
same = [0,0,0]
truth = [0,0,0]
predict = [0,0,0]
X_train=np.load('LSTM.npy')
def mapper(x):
    res=np.argmax(x)
    if res>=2:
        return -1
    else:
        return res
X_train=list(map(lambda x:mapper(x),X_train))
y_train=np.load('y_test.npy')
for i,each in enumerate(X_train):
    if (each==1 and y_train[i]==1):
        same[0]+=1
    elif (each==-1 and y_train[i]==-1):
        same[1]+=1
    elif (each==0 and y_train[i]==0):
        same[2]+=1
    if y_train[i]==1:
        truth[0] += 1
    elif y_train[i]==-1:
        truth[1]+=1
    elif y_train[i]==0:
        truth[2]+=1
    if each==1:
        predict[0]+=1
    elif each==-1:
        predict[1]+=1
    elif each==0:
        predict[2]+=1
t = PrettyTable(['class', 'precision','recall','F1 score'])
recall_sum = 0
precision_sum = 0
f1_sum = 0
recall = 1.0*same[0]/truth[0]
precision = 1.0*same[0]/predict[0]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['positive', precision,recall,f1])
recall = 1.0*same[1]/truth[1]
precision = 1.0*same[1]/predict[1]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['negative', precision,recall,f1])
recall = 1.0*same[2]/truth[2]
precision = 1.0*same[2]/predict[2]
f1 = 2*precision*recall/(precision+recall)
recall_sum += recall
precision_sum += precision
f1_sum += f1
t.add_row(['neutral', precision,recall,f1])
t.add_row(['average', precision_sum/3, recall_sum/3,f1_sum/3 ])
print t


# In[ ]:



