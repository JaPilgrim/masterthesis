#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel


# In[9]:


cuda_available = torch.cuda.is_available()

# define hyperparameter
train_args ={"reprocess_input_data": True,
             "fp16":False,
             "num_train_epochs": 4,
             "overwrite_output_dir" : True
             }

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "distilbert-base-german-cased",
    num_labels=2,
    args=train_args,
    use_cuda=cuda_available
)


# In[4]:


#Initialize Classes
def fetch_rawtext_from_wiki(subject='Maschinelles Lernen') :
     
    url = 'https://de.wikipedia.org/w/api.php'
    params = {
                'action': 'parse',
                'page': subject,
                'format': 'json',
                'prop':'text',
                'redirects':''
            }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    raw_html = data['parse']['text']['*']
    soup = BeautifulSoup(raw_html,'html.parser')
    soup.find_all('p')
    text = ''
    
    for p in soup.find_all('p'):
        text += p.text
    return text

def preprocess_text(text):
    newtext=re.sub("\[.*?\]","_",text)
    i=0
    newtext=newtext.replace("._","_.")
    while i<1:
        newtext=newtext.replace("._",".")
        i+=1
    return newtext

def split_classify_text(text):
    sen_text=text.split('.')
    is_claim=[]
    for i in sen_text:
        if "_" in i:
            is_claim.append(True)
        else:
            is_claim.append(False)
    df=pd.DataFrame()
    df["sentence_raw"]=sen_text
    df["is_claim"]=is_claim


    return df

def split_train_test(df):
    train_df, test_df = train_test_split(df,test_size=.10)

    return train_df,test_df


# In[6]:


# Run Classes
text=fetch_rawtext_from_wiki()
text_pre=preprocess_text(text)
df=split_classify_text(text_pre)
train_df,test_df = split_train_test(df)


# In[ ]:


model.train_model(train_df)


# In[24]:


# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

# model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")


# In[4]:





# In[5]:





# In[15]:


def tokenize_text(text):
    nlp = spacy.load("de_core_news_md")
    doc = nlp(sentences[0])
    print(doc.text)
    for token in doc:
        print(token.text, token.pos_, token.dep_)

