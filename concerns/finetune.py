from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import pandas as pd
import numpy as np
import pickle
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
import os

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')
print(device)

# Create torch dataset
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

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def test_metrics(pred,labels):
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Read data
categories = ['concerns_democracy',
 'concerns_economy',
 'concerns_environment',
 'concerns_fake_news/misinfo',
 'concerns_immigration/refugees',
 'concerns_international_alliances',
 'concerns_national_identity_&_pride',
 'concerns_relationship_w/_russia',
 'concerns_religion',
 'concerns_terrorism/counterterrorism']
for label in categories:
    data = pd.read_csv("/data/eval_train_df.csv")
    tdata = pd.read_csv("/data/eval_test_df.csv")
    tdata=tdata.dropna()
    data=data.dropna()
    model_name = "./bertweetfr-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # ----- 1. Preprocess data -----#
    # Preprocess data
    x = data["cleaned_text"].tolist()
    y = data[label].tolist()
    test_x=tdata['cleaned_text'].tolist()
    test_y=tdata[label].tolist()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1,random_state=0)


    x_train_tokenized = tokenizer(x_train, padding=True, truncation=True, max_length=512)
    x_val_tokenized = tokenizer(x_val, padding=True, truncation=True, max_length=512)
    x_test_tokenized = tokenizer(test_x, padding=True, truncation=True, max_length=512)

    train_dataset = Dataset(x_train_tokenized, y_train)
    val_dataset = Dataset(x_val_tokenized, y_val)
    test_dataset= Dataset(x_test_tokenized,test_y)

    name=label.replace(' ','_')
    name=label.replace('/','_')
    if not os.path.exists('/data/'+name+'/'):
        os.makedirs('/data/'+name+'/')

    # Define Trainer
    args = TrainingArguments(
        output_dir='/data/'+name+'/',
        evaluation_strategy="steps",
        save_steps=50,
        eval_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=7,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    test_dataset = Dataset(x_test_tokenized)
    raw_pred, _, _ = trainer.predict(test_dataset)
    y_pred_proba=raw_pred[:,1]

    res=pd.DataFrame(columns=['prediction'])
    res['prediction']=y_pred_proba
    res.to_csv('/data/'+name+'.csv',index=False)
