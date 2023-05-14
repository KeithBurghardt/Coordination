#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
import os
import numpy as np
import re
from collections import Counter
import sage
import spacy
from tqdm import tqdm


# In[5]:


baseline=''
filespath='/data/Coronavirus-Tweets/DARPA_INCAS/Wiki-Scrape/Robust_Scrape'
folders=os.listdir(filespath)
for folder in folders:
    files=os.listdir(os.path.join(filespath,folder))
    print(folder,files)
    files.sort()
    for file in files:
        if file!='.DS_Store':
            with open(os.path.join(filespath,folder,file),'r') as myfile:
                baseline=baseline+' '+myfile.read()


# In[6]:


baseline_arr=baseline.split('\n')
baseline_arr[0]


# In[7]:


nlp = spacy.load('fr_core_news_md')


# In[8]:


def lemmatize(sent):
    s=[token.lemma_ for token in nlp(sent)]
    s=' '.join(s)
    return s


# In[9]:


baseline_arr=baseline.split('\n')
baseline_arr=[lemmatize(b) for b in tqdm(baseline_arr)]
baseline_arr=[b.strip() for b in baseline_arr]


# In[10]:


irrelevant_chars="~?!./\:;+=&^%$#@(,)-[]-_*"
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


# In[11]:


from string import digits
def deep_clean(x):
    x=x.lower()
    x=re.sub(r'http\S+', '', x)
    remove_digits = str.maketrans('', '', digits)
    remove_chars = str.maketrans('', '', irrelevant_chars)
    x = x.translate(remove_digits)
    x = x.translate(remove_chars)
    x = emoji_pattern.sub(r'', x)
    x=x.replace('!','')
    x=x.replace('?','')
    x=x.replace('@','')
    x=x.replace('&','')
    x=x.replace('$','')
    x=[t for t in x.split() if len(t)>3]
    x=' '.join(x)
    return x


# In[12]:


baseline_arr=[deep_clean(b) for b in baseline_arr]


# In[13]:


from nltk.corpus import stopwords
stp=stopwords.words('french')
stopwords=pd.read_csv('stopwords_fr.csv')
stopwords=stopwords['stopwords'].tolist()
stopwords=[s.strip() for s in stopwords]
stp.extend(['macron','emmanuel','french','pen','marine','france','lepen','cette','faire',"c'est",'via'])
stp.extend(stopwords)


# In[14]:


base_words=[]
for b in tqdm(baseline_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    base_words.extend(b_arr)
base_count=Counter(base_words)


# In[15]:


word_arrs={}


# In[16]:


def ret_scores(eta,K=100):
    scores=eta[(-eta).argsort()[:K]]
    return scores


# ## Terrorism

# In[17]:


terror=[]
for file in os.listdir(os.path.join(filespath,'Terrorism')):
    terror.extend(open(os.path.join(filespath,'Terrorism',file),'r').readlines())
terror_arr=[lemmatize(t) for t in terror]
terror_arr=[deep_clean(t.strip()) for t in terror_arr]


# In[18]:


terror_words=[]
for b in tqdm(terror_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    terror_words.extend(b_arr)
terror_count=Counter(terror_words)


# In[19]:


vocab = [word for word,count in Counter(terror_count).most_common(5000)]
x_terr = np.array([terror_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.


# In[20]:


mu = np.log(x_base) - np.log(x_base.sum())


# In[21]:


eta = sage.estimate(x_terr,mu)


# In[22]:


terror=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
terror_dict={}
for i in range(len(terror)):
    terror_dict[terror[i]]=scores[i]
word_arrs['terrorism']=terror_dict
terror.extend(['isis','daesh','isil'])


# In[ ]:





# ## Economy

# In[23]:


economy=[]
for file in os.listdir(os.path.join(filespath,'Economy')):
    economy.extend(open(os.path.join(filespath,'Economy',file),'r').readlines())
economy_arr=[lemmatize(t) for t in economy]
economy_arr=[deep_clean(t.strip()) for t in economy_arr]


economy_words=[]
for b in tqdm(economy_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    economy_words.extend(b_arr)
economy_count=Counter(economy_words)

vocab = [word for word,count in Counter(economy_count).most_common(5000)]
x_eco = np.array([economy_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_eco,mu)


# In[24]:


eco=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
eco_dict={}
for i in range(len(eco)):
    eco_dict[eco[i]]=scores[i]
word_arrs['economy']=eco_dict


# ## Immigration

# In[25]:


immigration=[]
for file in os.listdir(os.path.join(filespath,'Immigration')):
    immigration.extend(open(os.path.join(filespath,'Immigration',file),'r').readlines())
immigration_arr=[lemmatize(t) for t in immigration]
immigration_arr=[deep_clean(t.strip()) for t in immigration_arr]

immigration_words=[]
for b in tqdm(immigration_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    immigration_words.extend(b_arr)
immigration_count=Counter(immigration_words)

vocab = [word for word,count in Counter(immigration_count).most_common(5000)]
x_immi = np.array([immigration_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_immi,mu)


# In[26]:


immi=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
immi_dict={}
for i in range(len(immi)):
    immi_dict[immi[i]]=scores[i]
word_arrs['immigration']=immi_dict


# ## Religion

# In[27]:


religion=[]
for file in os.listdir(os.path.join(filespath,'Religion')):
    religion.extend(open(os.path.join(filespath,'Religion',file),'r').readlines())
religion_arr=[lemmatize(t) for t in religion]
religion_arr=[deep_clean(t.strip()) for t in religion_arr]

religion_words=[]
for b in tqdm(religion_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    religion_words.extend(b_arr)
religion_count=Counter(religion_words)

vocab = [word for word,count in Counter(religion_count).most_common(5000)]
x_religion = np.array([religion_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_religion,mu)


# In[28]:


rel=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
rel_dict={}
for i in range(len(rel)):
    rel_dict[rel[i]]=scores[i]
word_arrs['religion']=rel_dict


# ## Climate

# In[29]:


climate=[]
for file in os.listdir(os.path.join(filespath,'Climate_Change')):
    climate.extend(open(os.path.join(filespath,'Climate_Change',file),'r').readlines())
climate_arr=[lemmatize(t) for t in climate]
climate_arr=[deep_clean(t.strip()) for t in climate_arr]

climate_words=[]
for b in tqdm(climate_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    climate_words.extend(b_arr)
climate_count=Counter(climate_words)

vocab = [word for word,count in Counter(climate_count).most_common(5000)]
x_climate = np.array([climate_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_climate,mu)


# In[30]:


cli=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
cli_dict={}
for i in range(len(cli)):
    cli_dict[cli[i]]=scores[i]
word_arrs['climate']=cli_dict


# ## Russia

# In[31]:


russia=[]
for file in os.listdir(os.path.join(filespath,'Russia')):
    russia.extend(open(os.path.join(filespath,'Russia',file),'r').readlines())
russia_arr=[lemmatize(t) for t in russia]
russia_arr=[deep_clean(t.strip()) for t in russia_arr]

russia_words=[]
for b in tqdm(russia_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    russia_words.extend(b_arr)
russia_count=Counter(russia_words)

vocab = [word for word,count in Counter(russia_count).most_common(5000)]
x_russia = np.array([russia_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_russia,mu)


# In[32]:


russ=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
russ_dict={}
for i in range(len(russ)):
    russ_dict[russ[i]]=scores[i]
word_arrs['russia']=russ_dict


# ## International Organizations

# In[33]:


intl=[]
for file in os.listdir(os.path.join(filespath,'International_Orgs')):
    intl.extend(open(os.path.join(filespath,'International_Orgs',file),'r').readlines())
intl_arr=[lemmatize(t) for t in intl]
intl_arr=[deep_clean(t.strip()) for t in intl_arr]


intl_words=[]
for b in tqdm(intl_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    intl_words.extend(b_arr)
intl_count=Counter(intl_words)

vocab = [word for word,count in Counter(intl_count).most_common(5000)]
x_intl = np.array([intl_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_intl,mu)


# In[34]:


intl=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
intl_dict={}
for i in range(len(intl)):
    intl_dict[intl[i]]=scores[i]
word_arrs['international organizations']=intl_dict
intl.extend(['nato','g7','g10','eu','Union européenne'])


# ## Fake News

# In[35]:


fake=[]
for file in os.listdir(os.path.join(filespath,'Fake_News')):
    fake.extend(open(os.path.join(filespath,'Fake_News',file),'r').readlines())
fake_arr=[lemmatize(t) for t in fake]
fake_arr=[deep_clean(t.strip()) for t in fake_arr]

fake_words=[]
for b in tqdm(fake_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    fake_words.extend(b_arr)
fake_count=Counter(fake_words)

vocab = [word for word,count in Counter(fake_count).most_common(5000)]
x_fake = np.array([fake_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_fake,mu)


# In[36]:


fake=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
fake_dict={}
for i in range(len(fake)):
    fake_dict[fake[i]]=scores[i]
word_arrs['fake news']=fake_dict


# ## Nationalism

# In[37]:


national=[]
for file in os.listdir(os.path.join(filespath,'Nationalism')):
    national.extend(open(os.path.join(filespath,'Nationalism',file),'r').readlines())
national_arr=[lemmatize(t) for t in national]
national_arr=[deep_clean(t.strip()) for t in national_arr]

national_words=[]
for b in tqdm(national_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    national_words.extend(b_arr)
national_count=Counter(national_words)

vocab = [word for word,count in Counter(national_count).most_common(5000)]
x_national = np.array([national_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_national,mu)


# In[38]:


national=sage.topK(eta,vocab,K=200)
scores=ret_scores(eta,200)
national_dict={}
for i in range(len(national)):
    national_dict[national[i]]=scores[i]
word_arrs['national']=national_dict
national.extend(['culture','fière','fièr'])


# ## Democracy

# In[39]:


democracy=[]
for file in os.listdir(os.path.join(filespath,'Democracy')):
    democracy.extend(open(os.path.join(filespath,'Democracy',file),'r').readlines())
democracy_arr=[lemmatize(t) for t in democracy]
democracy_arr=[deep_clean(t.strip()) for t in democracy_arr]

democracy_words=[]
for b in tqdm(democracy_arr):
    b_arr=b.split()
    b_arr=[b for b in b_arr if b not in stp]
    democracy_words.extend(b_arr)
democracy_count=Counter(democracy_words)

vocab = [word for word,count in Counter(democracy_count).most_common(5000)]
x_democracy = np.array([democracy_count[word] for word in vocab])
x_base = np.array([base_count[word] for word in vocab]) + 1.

mu = np.log(x_base) - np.log(x_base.sum())

eta = sage.estimate(x_democracy,mu)

