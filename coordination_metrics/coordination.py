#from flask import Flask
#app = Flask(__name__)

#@app.route('/')
#def hello_world():
#    return 'Hello, Docker!'

import pandas as pd
import networkx as nx
import numpy as np
import ast
from datetime import datetime


def hashtag_coord(twitter_data,min_hashes=3):
    # minimum of 5 hashtags; alternatives based on cosine similarity of tweets could work too
    #min_hashes = 5
    
    twitter_text = twitter_data['contentText'].values.astype(str)
    twitter_data['hashtag_seq'] = ['__'.join([tag.strip("#").strip('.').strip(',').strip(';').strip('!').strip(':') for tag in tweet.split() if tag.startswith("#")]) for tweet in twitter_text]
    unique_hash_seq = twitter_data['hashtag_seq'].drop_duplicates()
    hashes = twitter_data.groupby('hashtag_seq')
    duplicate_hash_users = {}

    for jj,tweet in enumerate(unique_hash_seq):
        
        if len(tweet.split('__')) < min_hashes: continue
        all_tweets = hashes.get_group(tweet)
        all_tweets_tweet = all_tweets.loc[(all_tweets['engagementType']=='tweet') | (all_tweets['engagementType']=='reply'),]
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
    # nodes = all user IDs of coorinated users
    # edges = if these pairs of users are coordinated
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def retweet_coord(twitter_data,most_similar_cutoff= 0.995):
    #most_similar_cutoff = 0.995
    # minimum number of retweets
    min_retweets=10
    rt_doc = []
    pred = []
    rt_doc = []
    num_times = []
    unique_users = twitter_data['twitterAuthorScreenname'].drop_duplicates().values
    user_twitter_data = twitter_data.groupby('twitterAuthorScreenname')
    twitter_data = pd.concat([user_twitter_data.get_group(u).loc[user_twitter_data.get_group(u)['engagementType']=='retweet',] for u in unique_users if 
len(user_twitter_data.get_group(u).loc[user_twitter_data.get_group(u)['engagementType']=='retweet',])>=min_retweets])
    unique_users = twitter_data['twitterAuthorScreenname'].drop_duplicates().values
    user_twitter_data = twitter_data.groupby('twitterAuthorScreenname')
    # record all retweet IDs
    for jj,coord_user in enumerate(unique_users):
        if jj % 10000 == 0:
            print(round(jj/len(unique_users)*100,2))
        coord = user_twitter_data.get_group(coord_user)
        tweet_types = coord['engagementType'].drop_duplicates().values
        retweets = np.array([])
        if 'retweet' in tweet_types:
            retweets = coord.loc[coord['engagementType']=='retweet','engagementParentId'].astype(str).values        
        rt_doc.append(retweets)
        num_times.append(len(retweets))
    # save retweet IDs as a long string (so we can use TF-IDF python algorithm)
    rt_doc_text = [' '.join(rts.astype(str)) for rts in rt_doc]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(rt_doc_text)
    num_times = np.array(num_times)
    
    user_coord = {}
    user_sim = {}    
    all_sim = []
    for ii,[user,vect] in enumerate(zip(unique_users,X)):
        if num_times[ii] < min_retweets: continue
        if ii % 1000 == 0:
            print(round(ii/len(unique_users)*100,2))            
        similarity = cosine_similarity(X[ii],X)
        similarity = np.array(similarity)[0]
        # remove user ii (self-loop)
        # removing clearly dissimilar user pairs (threshold of cosine similarity <= 0.4)
        # this action reduces memory load
        coord_users = list(unique_users[:])
        coord_sim = list(similarity[:])        
        coord_num_times = list(num_times[:])
        coord_users.pop(ii)
        coord_sim.pop(ii)
        coord_num_times.pop(ii)
        coord_users = np.array(coord_users)
        coord_sim = np.array(coord_sim)        
        coord_num_times = np.array(coord_num_times) 
        
        user_coord[user] = coord_users[(coord_num_times>=min_retweets)]
        user_sim[user] = coord_sim[(coord_num_times>=min_retweets)]
        all_sim += user_sim[user].tolist()

    min_cosine = np.quantile(all_sim,most_similar_cutoff)
    print('MIN COSINE ',min_cosine)
    all_weights = []
    retweet_edges = []
    #user_time = {user:time for user,time in zip(unique_users,num_times)}
    for user in user_coord.keys():#,time in zip(unique_users,num_times):
        #if user in user_coord.keys() and time > min_retweets:
        user_coord[user] = user_coord[user][user_sim[user] >= min_cosine]
        user_sim[user] = user_sim[user][user_sim[user] >= min_cosine]
        #for user_coord.keys():#user,time in zip(unique_users,num_times):
        #if time >= min_retweets:
        #    if user in user_sim.keys():
        retweet_edges+=[(user,u) for u in user_coord[user]]#[(user,u) for u,s in zip(user_coord[user],user_sim[user]) if s>min_cosine and u != user]

    G2 = nx.Graph()
    G2.add_edges_from(retweet_edges)
    return min_cosine,G2


# bin tweets between min year and max year in 30 minute intervals
def time_coord(twitter_data,min_month,min_year,max_month,max_year,most_similar_cutoff=0.995):
    min_tweets=10
    #most_similar_cutoff = 0.995
    bin_size = 30 # 30 minute intervals
    num_bins = int((datetime(max_year,max_month,1)-datetime(min_year,min_month,1)).days*24*60/bin_size)
    # bins of 30 minute intervals from min_year to max_year
    all_bins = [datetime(min_year,1,1)+timedelta(minutes=bin_size*i) for i in range(num_bins)]
    all_bins = pd.to_datetime(all_bins)
    time_doc = []
    num_times = []
    unique_users = twitter_data['twitterAuthorScreenname'].drop_duplicates().values
    user_twitter_data = twitter_data.groupby('twitterAuthorScreenname')
    twitter_data = pd.concat([user_twitter_data.get_group(u) for u in unique_users if len(user_twitter_data.get_group(u))>=min_tweets])
    unique_users = twitter_data['twitterAuthorScreenname'].drop_duplicates().values
    # saving time tweets are made as strings
    # we record whether tweets are made in 30 minute intervals
    for jj,coord_user in enumerate(unique_users):
        if jj % 50000 == 0:
            print(round(jj/len(unique_users)*100,2))        
        coord = user_twitter_data.get_group(coord_user)
        coord_tweet_time = pd.to_datetime(coord['time_dt'])
        # all times where there is 1+ tweets
        hist = np.histogram(coord_tweet_time,all_bins)
        coord_binned_times = hist[1][:-1][hist[0]>0]        
        time_doc.append(coord_binned_times)
        num_times.append(len(coord))

    time_doc_text = [' '.join(times.astype(str)) for times in time_doc]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(time_doc_text)


    num_times = np.array(num_times)
    
    user_coord = {}
    user_sim = {}

    #total_compared =0
    #num_not_compared = 0
    all_sim = []
    for ii,[user,vect] in enumerate(zip(unique_users,X)):
        if num_times[ii] < min_tweets: continue
        if ii % 100 == 0:
            print(round(ii/len(unique_users)*100,2))            
        similarity = cosine_similarity(X[ii],X)
        similarity = np.array(similarity)[0]

        coord_users = list(unique_users[:])
        coord_sim = list(similarity[:])        
        coord_num_times = list(num_times[:])
        coord_users.pop(ii)
        coord_sim.pop(ii)
        coord_num_times.pop(ii)
        coord_users = np.array(coord_users)
        coord_sim = np.array(coord_sim)        
        coord_num_times = np.array(coord_num_times) 
        
        user_coord[user] = coord_users[(coord_num_times>=min_tweets)]
        user_sim[user] = coord_sim[(coord_num_times>=min_tweets)]
        all_sim += user_sim[user].tolist()

    min_cosine = np.quantile(all_sim,most_similar_cutoff)
    print('MIN COSINE ',min_cosine)
    all_weights = []
    #user_time = {user:time for user,time in zip(unique_users,num_times)}
    for user,time in zip(unique_users,num_times):
        if user in user_coord.keys() and time >= min_tweets:
            user_coord[user] = user_coord[user][user_sim[user] >= min_cosine]
            user_sim[user] = user_sim[user][user_sim[user] >= min_cosine]

    time_edges = []
    for user,time in zip(unique_users,num_times):
        if time >= min_tweets:
            if user in user_sim.keys():
                 time_edges+=[(user,u) for u in user_coord[user]]#[(user,u) for u,s in zip(user_coord[user],user_sim[user]) if s>min_cosine and u != user]

    # nodes = coordinated accounts
    # edges = which accounts are very similar
    G3 = nx.Graph()
    G3.add_edges_from(time_edges)
    print(len(G3))
    G3.remove_edges_from(nx.selfloop_edges(G3))
    print(len(G3))
    return min_cosine,G3

def load_data(file):
    data=pd.read_json(file,lines=True)
    if 'Twitter' in data['mediaType'].drop_duplicates().values:
        authors = []
        engagement_types=[]
        tweet_ids = []
        engagementParentIds = []
        for line in data['mediaTypeAttributes'].values:
            line = ast.literal_eval(line)

            try:
                if 'twitterData' in line.keys():
                    if 'twitterAuthorScreenname' in line['twitterData'].keys():
                        author = line['twitterData']['twitterAuthorScreenname']
                    if 'engagementType' in line['twitterData'].keys():
                        engagement_type = line['engagementType']
                    if 'tweetId' in line['twitterData'].keys():
                        tweet_id = line['twitterData']['tweetId']
                    if 'engagementParentId' in line['twitterData'].keys():
                        if str(line['twitterData']['engagementParentId']) != 'null':
                            engagementParentId = line['twitterData']['engagementParentId']

            except:
                author=np.nan
                engagement_type=np.nan
                tweet_id = np.nan
                engagementParentId = np.nan

            authors.append(author)
            engagement_types.append(engagement_type)
            engagementParentIds.append(engagementParentId)
            tweet_ids.append(tweet_id)
        data['twitterAuthorScreenname'] = authors
        data['engagementType'] = engagement_types
        data['tweetId']=tweet_ids
        data['engagementParentId'] = engagementParentIds
    return data
def main(**args):
    file = args[0]
    print(file)
    data = load_data(file)
    for mediatype in data['mediaType'].drop_duplicates().values:
        outfile = file.replace('.jsonl','')+'_'+mediatype
        data_mt = data.loc[data['mediaType']==mediatype,]
        min_time = data_mt['timePublished'].values.min()
        datetime_obj=datetime.fromtimestamp(min_time)
        min_month = datetime_obj.month
        min_year = datetime_obj.year
        max_time = data_mt['timePublished'].values.max()
        datetime_obj=datetime.fromtimestamp(max_time)
        max_month = datetime_obj.month
        max_year = datetime_obj.year
        
        thresh,time_coord_accounts = time_coord(data_mt,min_month,min_year,max_month,max_year)
        thresh,retweet_coord_accounts = [None,nx.Graph()]
        hashtag_accounts = hashtag_coord(data_mt)
        if mediatype == 'Twitter':
            thresh,retweet_coord_accounts = retweet_coord(data_mt)
        nx.write_edgelist(hashtag_accounts,outfile+'_hashtag'+'.edgelist')
        nx.write_edgelist(time_coord_accounts,outfile+'_time'+'.edgelist')
        nx.write_edgelist(retweet_coord_accounts,outfile+'_retweet'+'.edgelist')     


