
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

twitter_text = twitter_data['contentText'].values.astype(str)
twitter_data['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in twitter_text]
unique_hash_seq = twitter_data['hashtag_seq'].drop_duplicates()
hashes = twitter_data.groupby('hashtag_seq')
duplicate_hash_users = {}
min_hash=5

for jj,tweet in enumerate(unique_hash_seq):
    # minimum of 5 hashes
    if len(tweet.split('__')) < min_hash: continue
    if jj % 10000 == 0: print(jj)
    all_tweets = hashes.get_group(tweet)
    all_tweets_tweet = all_tweets.loc[all_tweets['engagementType']=='tweet',]
    num_users = len(all_tweets_tweet['twitterAuthorScreenname'].drop_duplicates())
    # if multiple tweets and multiple users 
    if num_users < len(all_tweets_tweet):
        links = all_tweets[['tweetId','engagementParentId']].drop_duplicates()
        users = all_tweets['twitterAuthorScreenname'].drop_duplicates().tolist()
        duplicate_hash_users[tweet] = users
        
        
all_dup_hash_users = []
coord_hash_users = []
for key in duplicate_hash_users.keys():
    if len(duplicate_hash_users[key]) > 1:
        coord_hash_users.append(duplicate_hash_users[key])
    all_dup_hash_users+=duplicate_hash_users[key]
# network
edges = []
for nodes in coord_hash_users:
    unique_edges = list(set([tuple(sorted([n1,n2])) for n1 in nodes for n2 in nodes if n1 != n2]))
    edges+=(unique_edges)
        
