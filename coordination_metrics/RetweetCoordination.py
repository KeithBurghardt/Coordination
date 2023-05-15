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


rt_doc = []
pred = []
rt_doc = []
num_times = []


for jj,coord_user in enumerate(unique_users):
    if jj % 10000 == 0:
        print(round(jj/len(unique_users)*100,2),'%')
    coord = user_twitter_data.get_group(coord_user)
    retweets = coord.loc[coord['engagementType']=='retweet','engagementParentId'].astype(str).values        
    rt_doc.append(retweets)
    num_times.append(len(retweets))

rt_doc_text = [' '.join(rts.astype(str)) for rts in rt_doc]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(rt_doc_text)
np.save('num_times_retweets.npy',np.array(num_times))
np.save('unique_users.npy',np.array(unique_users))
np.save('TF-IDF_retweets.npy',X)
pk.dump(X,open('TF-IDF_retweets.pkl','wb'))


num_times = np.array(num_times)

k=5

user_coord = {}
user_sim = {}
num_not_compared = 0
num_edge = 0
for ii,[user,vect] in enumerate(zip(unique_users,X)):
    if ii % 1000 == 0:
        print(round(ii/len(unique_users)*100,2))
        print('Edges ',num_edge)
    if num_times[ii] <= k: continue
    preds = cosine_similarity(X[ii],X)
    u = unique_users[(num_times>k) & (np.array(preds)[0]>0.3)]
    us = np.array(preds)[0][(num_times>k) & (np.array(preds)[0]>0.3)]
    user_coord[user] = u[u!=user]
    user_sim[user] = us[u!=user]
    num_edge+=len(user_coord[user])
    num_not_compared += len(preds) - len(user_sim[user])
#preds = cosine_similarity(X,X)    
pk.dump(user_coord,open('user_coord_retweets.pkl','wb'))
pk.dump(user_sim,open('user_sim_retweets.pkl','wb'))
pk.dump(num_not_compared,open('num_not_compared_retweets.pkl','wb'))


all_weights = []
min_rt = 10
user_time = {user:time for user,time in zip(unique_users,num_times)}
for user,time in zip(unique_users,num_times):
    if user in user_coord.keys() and time > min_rt:
        all_weights+=[s for u,s in zip(user_coord[user],user_sim[user]) if u != user and user_time[u]>min_rt]
print(len(all_weights))   
all_weights = np.array(all_weights)
total_num = len(all_weights)
for t in sorted(list(set(list(all_weights)))):
    if len(all_weights[all_weights>t])/total_num <= 0.005:
        thresh = t
        break


retweet_edges = []
for user,time in zip(unique_users,num_times):
    if user in user_coord.keys():
        retweet_edges+=[(user,u) for u,s in zip(user_coord[user],user_sim[user]) if s>thresh and u != user]

    
