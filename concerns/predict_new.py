import pandas as pd
import spacy
import re
from string import digits
import numpy as np
import time
from tqdm import tqdm
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
import os
import math
import logging
import pickle
logging.disable(logging.WARNING)

if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
#device = torch.device(﻿"cuda:0" if torch.cuda.is_available(﻿) else "cpu"﻿)

return_list=[]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Dataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


''' Function to generate annotation object that will be returned '''
def generate_annotation_object(tweetids,text,annot_type,annots):
    #result=manager.dict()
    results=[]
    for i in range(len(tweetids)):
        result={}
        result["id"]=tweetids[i]
        result["type"]=annot_type
        result["text"]=text
        result["confidence"]=annots[i]
        result["providerName"]="ta1-usc-isi"
        results.append(result)
    return results

def terror_annotate(ids,test_dataset):
    global return_list
    terr_model_path = "new_models/concerns_terrorism_counterterrorism/checkpoint-400/"
    terr_model = AutoModelForSequenceClassification.from_pretrained(terr_model_path, num_labels=2).to(device)
    test_trainer = Trainer(terr_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del terr_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Terrorism and Counterterrorism","concern-3.2",raw_confidence))


def econ_annotate(ids,test_dataset):
    global return_list
    eco_model_path = "new_models/concerns_economy/checkpoint-400/"
    eco_model = AutoModelForSequenceClassification.from_pretrained(eco_model_path, num_labels=2).to(device)
    test_trainer = Trainer(eco_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del eco_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Economy","concern-3.1",raw_confidence))


def rel_annotate(ids,test_dataset):
    global return_list
    rel_model_path = "new_models/concerns_religion/checkpoint-400/"
    rel_model = AutoModelForSequenceClassification.from_pretrained(rel_model_path, num_labels=2).to(device)
    test_trainer = Trainer(rel_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del rel_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Religion","concern-3.3",raw_confidence))

def immi_annotate(ids,test_dataset):
    global return_list
    immi_model_path = "new_models/concerns_immigration_refugees/checkpoint-400/"
    immi_model = AutoModelForSequenceClassification.from_pretrained(immi_model_path, num_labels=2).to(device)
    test_trainer = Trainer(immi_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del immi_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Immigration and Refugees","concern-3.4",raw_confidence))

def russ_annotate(ids,test_dataset):
    global return_list
    russ_model_path = "new_models/concerns_relationship_w__russia/checkpoint-400/"
    russ_model = AutoModelForSequenceClassification.from_pretrained(russ_model_path, num_labels=2).to(device)
    test_trainer = Trainer(russ_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del russ_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Relationship with Russia","concern-3.6",raw_confidence))

def cli_annotate(ids,test_dataset):
    global return_list
    cli_model_path = "new_models/concerns_environment/checkpoint-200/"
    cli_model = AutoModelForSequenceClassification.from_pretrained(cli_model_path, num_labels=2).to(device)
    test_trainer = Trainer(cli_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del cli_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Environment and Climate Change","concern-3.8",raw_confidence))


def natl_annotate(ids,test_dataset):
    global return_list
    natl_model_path = "new_models/concerns_national_identity_&_pride/checkpoint-400/"
    natl_model = AutoModelForSequenceClassification.from_pretrained(natl_model_path, num_labels=2).to(device)
    test_trainer = Trainer(natl_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del natl_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"National Identity and National Pride","concern-3.7",raw_confidence))

def intl_annotate(ids,test_dataset):
    global return_list
    intl_model_path = "new_models/concerns_international_alliances/checkpoint-350/"
    intl_model = AutoModelForSequenceClassification.from_pretrained(intl_model_path, num_labels=2).to(device)
    test_trainer = Trainer(intl_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del intl_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"International Alliance Organizations","concern-3.5",raw_confidence))


def fake_annotate(ids,test_dataset):
    global return_list
    fake_model_path = "new_models/concerns_fake_news_misinfo/checkpoint-200/"
    fake_model = AutoModelForSequenceClassification.from_pretrained(fake_model_path, num_labels=2).to(device)
    test_trainer = Trainer(fake_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del fake_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Fake News/Misinformation","concern-3.9",raw_confidence))

def dem_annotate(ids,test_dataset):
    global return_list
    dem_model_path = "new_models/concerns_democracy/checkpoint-400/"
    dem_model = AutoModelForSequenceClassification.from_pretrained(dem_model_path, num_labels=2).to(device)
    test_trainer = Trainer(dem_model)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    raw_pred, _, _=test_trainer.prediction_loop(test_loader, description="prediction")
    raw_confidence = raw_pred[:,1]
    raw_confidence = [sigmoid(x) for x in raw_confidence]
    del dem_model
    torch.cuda.empty_cache()
    return_list.extend(generate_annotation_object(ids,"Democracy","concern-3.11",raw_confidence))


df=pd.read_csv('AllCombinedTwitterData+text_cleaned.csv')
print('file read')
df=df[['id','cleaned_text']]
df=df.dropna()
func_dict=[terror_annotate,econ_annotate,rel_annotate,immi_annotate,russ_annotate,cli_annotate,natl_annotate,intl_annotate,fake_annotate,dem_annotate]
ids=df['id'].tolist()
tweet_text=df['cleaned_text'].tolist()
tokenizer = AutoTokenizer.from_pretrained("./bertweetfr-base")
x_test_tokenized = tokenizer(tweet_text, padding=True, truncation=True, max_length=256)
test_dataset = Dataset(x_test_tokenized)

for f in func_dict:
    f(ids,test_dataset)
with open('icwsm_eval.pkl','wb') as fi:
    pickle.dump(return_list,fi)
