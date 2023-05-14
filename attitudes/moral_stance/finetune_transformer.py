import os
import argparse
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import classification_report,roc_auc_score,f1_score

from preprocessing import preprocess_tweet


## Parameters
# LR = 2e-5
# EPOCHS = 5
# BATCH_SIZE = 32
MODEL = "cardiffnlp/twitter-xlm-roberta-base"
# MODEL = "camembert-base"

## Data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# def load_data(train_data_path,test_data_path):
def load_data(data_path,test_path,seed =1):
    """
    train/test_data_path: path to csv files with columns 'text' and 'label'

    Note: sentiment analysis data for multiple languages are available here:
          # https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment
    """
    # loading training and dev dataset
    df_train = pd.read_csv(data_path,lineterminator='\n')
    # df_train = df.sample(frac=0.9,random_state=seed)
    # df_test = df.drop(df_train.index)
    # df_train = pd.read_csv(train_data_path,lineterminator='\n')
    df_test = pd.read_csv(test_path,lineterminator='\n')
    df_val = df_train.sample(frac=0.1,random_state=seed)
    df_train = df_train.drop(df_val.index)

    dataset_dict = {
        'train':df_train,
        'val':df_val,
        'test':df_test
    }

    for i in ['train','val','test']:
        dataset_dict[i] = {
            'text':dataset_dict[i]['text'].apply(preprocess_tweet).tolist(), 
            'labels':dataset_dict[i]['label'].astype(int).values
            }

    return dataset_dict


def softmax(z):
    exp = np.exp(z - np.max(z))
    exp = exp/np.sum(exp,axis=-1)[:,np.newaxis]
    return exp


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = [0,1,2]
  roc_auc_lst = []
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_lst.append(roc_auc)

  return roc_auc_lst

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    ## command args
    parser = argparse.ArgumentParser(description='Moral/Immoral prediction for French election tweets.')

    parser.add_argument('--mode',default='train_and_test',type=str, help='train, test, or train_and_test')
    parser.add_argument('--data_path', type=str, help='path to train/dev/test data, the program will automatically split train/dev')
    parser.add_argument('--test_path', type=str, help='path to test data')
    parser.add_argument('-l','--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-f','--max_seq_len', default=50, type=int, help='max sequence length')
    parser.add_argument('-b','--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('-e','--num_epoch', default = 10, type=int, help='number of epochs to train for')
    parser.add_argument('-o','--output_dir', default = './model_outputs', type=str, help='output dir to be written')
    parser.add_argument('-m','--model_path', default = 'camembert-base', type=str, help='pretrained model to be used')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.mode == 'inference':
        df = pd.read_csv(args.data_path,lineterminator='\n')
        print('len of raw data: ',len(df))
        ids = df['id'].tolist()
        tqdm.pandas()
        tweets = df['contentText'].progress_apply(preprocess_tweet).tolist()
        labels = [0]*len(tweets)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        test_encodings = tokenizer(tweets, truncation=True, max_length=args.max_seq_len, padding="max_length")
        test_dataset = MyDataset(test_encodings, labels)

        training_args = TrainingArguments(
                output_dir=args.output_dir,                        # output directory
                per_device_eval_batch_size=2048
            )

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        trainer = Trainer(
            model=model,                              # the instantiated Transformers model to be trained
            args=training_args                        # training arguments, defined above
            )
                
        test_preds_raw, _ , _ = trainer.predict(test_dataset)
        test_preds_confidence = softmax(test_preds_raw)

        cnt = 0
        with open('incas_1a_all_data_morality.jsonl','w') as f:
            for i,p in tqdm(zip(ids,test_preds_confidence),desc='writing to file'):
                # agenda: moral/beneficial
                f.write(json.dumps({'id':i,
                                'type':'agenda-1.4',
                                'text':'Believe that ENTITY or GROUP is moral/ethical/honest/beneficial',
                                'confidence':p[1].item(),
                                'providerName':'ta1-usc-isi'})+'\n')
                # agenda: immoral/harmful
                f.write(json.dumps({'id':i,
                                'type':'agenda-1.3',
                                'text':'Believe that ENTITY or GROUP is immoral/unethical/dishonest/harmful',
                                'confidence':p[2].item(),
                                'providerName':'ta1-usc-isi'})+'\n')
                cnt += 1
        #np.savetxt(args.output_dir+'/test_preds_confidence.txt', test_preds_confidence, delimiter=",")
        print('len of predictions: ',cnt)
    else:
        res = {'auc_perclass_macro':[],'auc_perclass_weighted':[],'auc_perclass_micro':[],'f1':[]}
        for seed in [1]:
            set_seed(seed)

            ## Process data
            # dataset_dict = load_data(args.train_path,args.test_path)
            dataset_dict = load_data(args.data_path,args.test_path,seed=seed)

            tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,local_files_only=True)
            train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")
            val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")

            train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
            val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])

            ## Args
            training_args = TrainingArguments(
                output_dir=args.output_dir,                        # output directory
                num_train_epochs=args.num_epoch,                  # total number of training epochs
                per_device_train_batch_size=args.batch_size,       # batch size per device during training
                per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
                learning_rate=args.lr,                      # learning rate
                warmup_steps=100,                         # number of warmup steps for learning rate scheduler
                weight_decay=0.01,                        # strength of weight decay
                logging_dir=args.output_dir+'/logs',                     # directory for storing logs
                logging_steps=100,                         # when to print log
                evaluation_strategy='steps',
                eval_steps=100,
                load_best_model_at_end=True,              # load or not best model at the end
                disable_tqdm=True,
                seed=seed
            )

            ## Training
            if 'train' in args.mode:
                assert dataset_dict['train'] is not None, 'training data is missing!'
                
                model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3,local_files_only=True)
                
                trainer = Trainer(
                    model=model,                              # the instantiated Transformers model to be trained
                    args=training_args,                       # training arguments, defined above
                    train_dataset=train_dataset,              # training dataset
                    eval_dataset=val_dataset                  # evaluation dataset
                )

                trainer.train()

                trainer.save_model(f"./{args.output_dir}/best_model")

                val_preds_raw, val_labels , _ = trainer.predict(val_dataset)
                val_preds = np.argmax(val_preds_raw, axis=-1)
                print('validation set ROC-AUC: ',roc_auc_score_multiclass(val_labels.tolist(), val_preds.tolist(), average = "macro"))

            ## Test
            if 'test' in args.mode:
                assert dataset_dict['test'] is not None, 'test data is missing!'

                test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")
                test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

                if args.mode == 'test':
                    assert len(args.model_path) > 0, 'trained model file is missing!'
                    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
                    trainer = Trainer(
                        model=model,                              # the instantiated Transformers model to be trained
                        args=training_args                        # training arguments, defined above
                    )
                
                test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
                test_preds_confidence = softmax(test_preds_raw)
                test_preds = np.argmax(test_preds_raw, axis=-1)

                report = classification_report(test_labels, test_preds, digits=3)
                print(report)
                # res['auc_ovr'].append(roc_auc_score(test_labels.tolist(), test_preds.tolist(), average='weighted', multi_class='ovr'))
                # res['auc_ovo'].append(roc_auc_score(test_labels.tolist(), test_preds.tolist(),average='weighted',multi_class='ovo'))
                res['auc'].append(roc_auc_score_multiclass(test_labels.tolist(), test_preds.tolist()))
                res['f1'].append(f1_score(test_labels, test_preds, average=None).tolist())

                # np.savetxt(args.output_dir+'/test_preds_confidence_'+str(seed)+'.txt', test_preds_confidence, delimiter=",")
                df_eval = pd.read_csv(args.test_path,lineterminator='\n')
                df_eval['non_moral_conf'] = test_preds_confidence[:,0]
                df_eval['moral_conf'] = test_preds_confidence[:,1]
                df_eval['immoral_conf'] = test_preds_confidence[:,2]
                df_eval.to_csv(args.output_dir+'/test_preds_confidence.csv',index=False)

                with open(args.output_dir+'/classification_report_'+str(seed)+'.txt','w+') as f:
                    f.write(report)
        
        print(res)
