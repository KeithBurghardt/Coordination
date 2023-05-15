import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ace_dir = 'new_ace/'
directory = ''
# prolific
#twitter_data = pd.read_csv(directory+'AllCombinedTwitterData+text_new.csv')
twitter_data = pd.read_csv(ace_dir+'twitter_data_updated.csv')
user_twitter_data = twitter_data.groupby('twitterAuthorScreenname')
unique_users = np.unique(twitter_data['twitterAuthorScreenname'].values)




num_bins = (datetime(2018,1,1)-datetime(2017,1,1)).days*24*2
all_bins = [datetime(2017,1,1)+timedelta(minutes=30*i) for i in range(num_bins)]
all_bins = pd.to_datetime(all_bins)

rt_doc = []
pred = []
rt_doc = []
num_times = []


for jj,coord_user in enumerate(unique_users):
    if jj % 10000 == 0:
        print(round(jj/number_coord*100,2),'%')
    coord = user_twitter_data.get_group(coord_user)
    coord_tweet_time = pd.to_datetime(coord['time_dt'])
    hist = np.histogram(coord_tweet_time,all_bins)
    coord_binned_times = hist[1][:-1][hist[0]>0]        
    rt_doc.append(coord_binned_times)
    num_times.append(len(coord))

rt_doc_text = [' '.join(rts.astype(str)) for rts in rt_doc]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(rt_doc_text)

num_times = np.load('num_times_time.npy',allow_pickle=True)
unique_users = np.load('unique_users.npy',allow_pickle=True)
X = np.load('TF-IDF_time.npy',allow_pickle=True)

user_coord= pk.load(open('user_coord_times.pkl','rb'))
user_sim = pk.load(open('user_sim_times.pkl','rb'))
num_not_compared = pk.load(open('num_not_compared_times.pkl','rb'))



retweet_edges = []
for user,time in zip(unique_users,num_times):
    if time > 5:
        if user in user_sim.keys():
             retweet_edges+=[(user,u) for u,s in zip(user_coord[user],user_sim[user]) if s>0.99 and u != user]

    
