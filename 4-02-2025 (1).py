#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


import gensim


# In[3]:


import pandas as pd
from gensim.models import word2vec


# In[5]:


corpus=[
    'I love reading .',
    'Modeling is great .',
    'I would do that .'
]
tokenized_corpus=[sentence.lower().split() for sentence in corpus]
print(tokenized_corpus)


# In[8]:


model = word2vec.Word2Vec(tokenized_corpus,vector_size=100,window=5,min_count=1,workers=4)
model.save('word2vec.model')
print(model)


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("https://github.com/suhasmaddali/Twitter-Sentiment-Analysis/raw/refs/heads/main/train.csv")


# In[4]:


df=df[["text","sentiment"]]


# In[5]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters & numbers
    words = nltk.word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

tokens = []
for i in range(len(df)):
    tokens.append(preprocess_text(df['text'][i]))


# In[6]:


tokens


# In[ ]:




