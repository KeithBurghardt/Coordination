import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
# os.environ['TOKENIZERS_PARALLELISM'] = '0'


class PStanceCOVID2FrenchElection(Dataset):
    def __init__(self, phase, model='bert-base', wiki_model='', random_seed=0, inference=0, chunk_idx=-1):
        if inference == 0:
            df_fe = pd.read_csv(f'../incas/10k_annotations.csv')
            if random_seed != 2022:
                df_fe_train = df_fe.sample(frac=0.5, random_state=random_seed)
                df_fe_test = df_fe.drop(df_fe_train.index)
                df_fe_val = df_fe_train.sample(frac=0.15, random_state=random_seed)
                df_fe_train = df_fe_train.drop(df_fe_val.index)
            else:
                eval_inds = pickle.load(open('../eval_inds2.pkl', 'rb'))
                df_fe_test = df_fe.iloc[eval_inds]
                df_fe_train = df_fe.drop(eval_inds)
                df_fe_val = df_fe_train.sample(frac=0.15, random_state=random_seed)
                df_fe_train = df_fe_train.drop(df_fe_val.index)

            if phase == 'test':
                df = df_fe_test
            else:
                file_paths = [f'../stance-detection/PStance/processed_{phase}_{t}.csv'
                              for t in ['trump', 'biden', 'bernie']]
                dfs_pstance = [pd.read_csv(file_path) for file_path in file_paths]
                df_pstance = pd.concat(dfs_pstance)
                df_pstance = df_pstance[['text', 'target', 'label']]

                file_paths = [f'../stance-detection/covid19-twitter/{t}_{phase}.csv'
                              for t in ['face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']]
                dfs_covid = [pd.read_csv(file_path) for file_path in file_paths]
                df_covid = pd.concat(dfs_covid)
                df_covid = df_covid[['Tweet', 'Target', 'Stance']]
                df_covid['Stance'] = df_covid['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}).tolist()
                df_covid.columns = ['text', 'target', 'label']

                if phase == 'train':
                    df = pd.concat((df_pstance, df_covid, df_fe_train))
                else:
                    df = pd.concat((df_pstance, df_covid, df_fe_val))

        else:
            df_fe = pd.read_csv(f'../10k_annotations.csv')
            df_fe_train = df_fe.sample(frac=0.8, random_state=random_seed)
            df_fe_val = df_fe.drop(df_fe_train.index)

            df_phase1a = pd.read_csv(f'phase1A_data.csv')

            if chunk_idx != -1:
                dfs_phase1a = np.array_split(df_phase1a, 10)
                del df_phase1a
                df_phase1a = dfs_phase1a[chunk_idx]

            if phase == 'test':
                df = df_phase1a
            else:
                file_paths = [f'../stance-detection/PStance/processed_{phase}_{t}.csv'
                              for t in ['trump', 'biden', 'bernie']]
                dfs_pstance = [pd.read_csv(file_path) for file_path in file_paths]
                df_pstance = pd.concat(dfs_pstance)
                df_pstance = df_pstance[['text', 'target', 'label']]

                file_paths = [f'../stance-detection/covid19-twitter/{t}_{phase}.csv'
                              for t in ['face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']]
                dfs_covid = [pd.read_csv(file_path) for file_path in file_paths]
                df_covid = pd.concat(dfs_covid)
                df_covid = df_covid[['Tweet', 'Target', 'Stance']]
                df_covid['Stance'] = df_covid['Stance'].map({'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}).tolist()
                df_covid.columns = ['text', 'target', 'label']

                if phase == 'train':
                    df = pd.concat((df_pstance, df_covid, df_fe_train))
                else:
                    df = pd.concat((df_pstance, df_covid, df_fe_val))

        print(f'# of {phase} examples: {df.shape[0]}\n')

        tweets = df['text'].tolist()
        targets = df['target'].tolist()
        stances = df['label'].tolist()

        # os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        model_name = {
            'bert-base': 'bert-base-uncased',
            'roberta': 'roberta-base',
            'bertweet': 'vinai/bertweet-base',
            'covid-twitter-bert': 'digitalepidemiologylab/covid-twitter-bert-v2',
            'bertweet-covid': "vinai/bertweet-covid19-base-uncased",
            'xlm-t': 'cardiffnlp/twitter-xlm-roberta-base'
        }[model]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if wiki_model:
            wiki_dict_pstance = pickle.load(open(f'../stance-detection/PStance/wiki_dict.pkl', 'rb'))
            wiki_dict_covid = pickle.load(open(f'../stance-detection/covid19-twitter/wiki_dict.pkl', 'rb'))
            wiki_dict_fe = pickle.load(open(f'../wiki_dict.pkl', 'rb'))

            wiki_dict = {}
            for wiki_dict_each in [wiki_dict_pstance, wiki_dict_covid, wiki_dict_fe]:
                for each in wiki_dict_each:
                    wiki_dict[each] = wiki_dict_each[each]
            wiki_dict['biden'] = wiki_dict['joe biden']
            wiki_dict['bernie'] = wiki_dict['bernie sanders']
            wiki_dict['trump'] = wiki_dict['donald trump']

            wiki_summaries = pd.Series(targets).map(wiki_dict).tolist()

            if wiki_model == model or wiki_model == 'merge':
                tokenizer_wiki = tokenizer
            else:
                if wiki_model == 'bert-base':
                    tokenizer_wiki = AutoTokenizer.from_pretrained('bert-base-uncased')
                elif wiki_model == 'roberta':
                    tokenizer_wiki = AutoTokenizer.from_pretrained('roberta-base')
                elif wiki_model == 'bertweet':
                    tokenizer_wiki = AutoTokenizer.from_pretrained('vinai/bertweet-base')
                elif wiki_model == 'covid-twitter-bert':
                    tokenizer_wiki = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
                elif wiki_model == 'bertweet-covid':
                    tokenizer_wiki = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-uncased",
                                                                   normalization=True)
                else: # bert-small
                    tokenizer_wiki = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

            if wiki_model == 'merge':
                tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, targets)]
                encodings = tokenizer(tweets_targets, wiki_summaries, padding=True, truncation=True)
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            else:
                if phase == 'test' and inference == 1:
                    if not os.path.exists('tokenized_phase1a.pkl'):
                        encodings = tokenizer(tweets, targets, padding=True, truncation=True)
                        # pickle.dump(encodings, open('tokenized_phase1a.pkl', 'wb'))
                        # print('Successfully saved tokenized encodings!')
                    else:
                        encodings = pickle.load(open('tokenized_phase1a.pkl', 'rb'))
                else:
                    encodings = tokenizer(tweets, targets, padding=True, truncation=True)
                if wiki_model != 'bert-small':
                    encodings_wiki = tokenizer_wiki(wiki_summaries, padding=True, truncation=True)
                else:
                    encodings_wiki = tokenizer_wiki(wiki_summaries, padding='max_length', truncation=True, max_length=512)

        else:
            encodings = tokenizer(tweets, targets, padding=True, truncation=True)
            encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}

        # encodings for the texts and tweets
        input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long) \
            if model not in ['roberta', 'bertweetfr', 'xlm-t'] else torch.zeros(df.shape[0])

        # encodings for wiki summaries
        input_ids_wiki = torch.tensor(encodings_wiki['input_ids'], dtype=torch.long)
        attention_mask_wiki = torch.tensor(encodings_wiki['attention_mask'], dtype=torch.long)

        stances = torch.tensor(stances, dtype=torch.long)
        print(f'max len: {input_ids.shape[1]}, max len wiki: {input_ids_wiki.shape[1]}')

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.stances = stances
        self.input_ids_wiki = input_ids_wiki
        self.attention_mask_wiki = attention_mask_wiki

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'stances': self.stances[index],
            'input_ids_wiki': self.input_ids_wiki[index],
            'attention_mask_wiki': self.attention_mask_wiki[index]
        }
        return item

    def __len__(self):
        return self.stances.shape[0]


def data_loader(data, phase, topic, batch_size, model='bert-base', wiki_model='', random_seed=42, inference=0,
                chunk_idx=-1):
    shuffle = True if phase == 'train' else False
    drop_last = False
    dataset = PStanceCOVID2FrenchElection(phase, model=model, wiki_model=wiki_model, random_seed=random_seed,
                                          inference=inference, chunk_idx=chunk_idx)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=4)
    return loader
