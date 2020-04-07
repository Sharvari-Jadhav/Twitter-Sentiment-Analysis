#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train = pd.read_csv('C:/Users/SHARVARI JADHAV/Documents/Twitter sentiment dataset/train_E6oV3lV.csv')
test = pd.read_csv('C:/Users/SHARVARI JADHAV/Documents/Twitter sentiment dataset/test_tweets_anuFYb8.csv')


# In[6]:


train.head()


# In[7]:


combi = train.append(test,ignore_index=True)


# In[8]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    
    for i in r:
        
       input_txt = re.sub(i,'',input_txt)
    
    return input_txt


# In[9]:


combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'],"@[\w]*")


# In[10]:


combi.head()


# In[11]:


combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[13]:


combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if(len(w))>3] ))


# In[14]:


combi.head()


# In[15]:


tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[16]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet.head()


# In[17]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet
combi.head()


# In[25]:


all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size =110).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[26]:


normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size =110).generate(normal_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[27]:


negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

wordcloud = WordCloud(width = 800, height = 500, random_state = 21, max_font_size =110).generate(negative_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[28]:


def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hashtags.append(ht)
    
    return hashtags


# In[30]:


HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])


# In[34]:


HT_regular


# In[33]:


HT_regular=sum(HT_regular,[])
HT_negative=sum(HT_negative,[])


# In[35]:


a = nltk.FreqDist(HT_regular)
a


# In[38]:


d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})


# In[40]:


d = d.nlargest(columns = "Count", n = 10)
plt.figure(figsize =(16,5))


# In[42]:


ax = sns.barplot(data = d, x = "Hashtag", y = "Count")
ax.set(ylabel = 'count')
plt.show()


# In[43]:


n = nltk.FreqDist(HT_negative)
df = pd.DataFrame({'Hashtag': list(n.keys()),'Count': list(n.values())})
l = df.nlargest(columns = "Count", n =10)
plt.figure(figsize = (16,5))
bx = sns.barplot(data = l, x = "Hashtag", y = "Count")
bx.set(ylabel = "Count")
plt.show()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df = 0.90, min_df=2, max_features =1000, stop_words = 'english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, min_df=2, max_features =1000, stop_words = 'english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid =  train_test_split(train_bow, train['label'], random_state = 42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow,ytrain)

prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid,prediction_int)


# In[46]:


test_pred=lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1]>=0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('C:/Users/SHARVARI JADHAV/Documents/Twitter sentiment dataset/sub_lreg_bow.csv',index = False)


# In[47]:


train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf,ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)


# In[ ]:




