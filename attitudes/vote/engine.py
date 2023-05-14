import copy

import torch
import torch.nn as nn
import os
import numpy as np
from datasets import data_loader
from models import BERTSeqClf, ModelWithTemperature, _ECELoss
from scipy.special import softmax


class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        suffix = f'{args.model}_{args.wiki_model}_seed={args.seed}_inference={args.inference}'
        if args.chunk_idx != -1:
            suffix += f'_chunk_idx={args.chunk_idx}'

        print('Preparing data....')
        print('Training data....')
        if not os.path.exists(f'ckp/model_{suffix}.pt'):
            train_loader = data_loader(args.data, 'train', args.topic, args.batch_size, model=args.model,
                                       wiki_model=args.wiki_model, random_seed=args.seed, inference=args.inference)
        else:
            train_loader = None

        if not os.path.exists(f'ckp/model_{suffix}.pt'):
            print('Val data....')
            val_loader = data_loader(args.data, 'val', args.topic, 2*args.batch_size, model=args.model,
                                     wiki_model=args.wiki_model, random_seed=args.seed, inference=args.inference)
        else:
            val_loader = None

        print('Test data....')
        if args.inference == 0:
            test_loader = data_loader(args.data, 'test', args.topic, 2*args.batch_size, model=args.model,
                                  wiki_model=args.wiki_model, random_seed=args.seed, inference=args.inference)
        else:
            test_loader = None
        print('Done\n')

        os.environ['TOKENIZERS_PARALLELISM'] = '0'

        print('Initializing model....')
        # num_labels = 2 if args.data == 'pstance' else 3
        num_labels = 3
        model = BERTSeqClf(num_labels=num_labels, model=args.model, n_layers_freeze=args.n_layers_freeze,
                           wiki_model=args.wiki_model, n_layers_freeze_wiki=args.n_layers_freeze_wiki)
        model = nn.DataParallel(model)
        model.to(device)

        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        criterion = nn.CrossEntropyLoss(ignore_index=3)

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.suffix = suffix
        self.calibration_temperature = 1

    def train(self):
        if os.path.exists(f'ckp/model_{self.suffix}.pt'):
            print('Loading checkpoint.....')
            best_state_dict = torch.load(f'ckp/model_{self.suffix}.pt')
            print('Done\n')
        else:
            import copy
            best_epoch = 0
            best_epoch_f1 = 0
            best_state_dict = copy.deepcopy(self.model.state_dict())
            for epoch in range(self.args.epochs):
                print(f"{'*' * 30}Epoch: {epoch + 1}{'*' * 30}")
                loss = self.train_epoch()
                print('Epoch Training Finished\n')
                print('Evaluating....')
                f1, f1_favor, f1_against, f1_neutral = self.eval('val')
                if f1 > best_epoch_f1:
                    best_epoch = epoch
                    best_epoch_f1 = f1
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                print(f'Epoch: {epoch+1}\tTrain Loss: {loss:.3f}\tVal F1: {f1:.3f}\n'
                      f'Val F1_favor: {f1_favor:.3f}\tVal F1_against: {f1_against:.3f}\tVal F1_Neutral: {f1_neutral:.3f}\n'
                      f'Best Epoch: {best_epoch+1}\tBest Epoch Val F1: {best_epoch_f1:.3f}\n')
                if epoch - best_epoch >= self.args.patience:
                    break

            print('Saving the best checkpoint....')
            torch.save(self.model.state_dict(), f'ckp/model_{self.suffix}.pt')
            print('Done\n')

        self.model.load_state_dict(best_state_dict)
        # torch.save(best_state_dict, f"ckp/model-small.pt")
        # if self.args.inference == 1:
        #     print('Calibrating....')
        #     self.calibrate()

        print('Testing....')
        if self.args.data != 'vast':
            if self.args.inference == 0:
                f1_avg, f1_favor, f1_against, f1_neutral = self.eval('test')
                print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
                      f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}')
            else:
                for i in range(10):
                    self.test_loader = data_loader(self.args.data, 'test', self.args.topic, 2 * self.args.batch_size,
                                                   model=self.args.model, wiki_model=self.args.wiki_model,
                                                   random_seed=self.args.seed, inference=self.args.inference,
                                                   chunk_idx=i)
                    f1_avg, f1_favor, f1_against, f1_neutral = self.eval('test', chunk_idx=i)
                    print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
                          f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}')
        else:
            f1_avg, f1_favor, f1_against, f1_neutral, \
            f1_avg_few, f1_favor_few, f1_against_few, f1_neutral_few, \
            f1_avg_zero, f1_favor_zero, f1_against_zero, f1_neutral_zero, = self.eval('test')
            print(f'Test F1: {f1_avg:.3f}\tTest F1_Favor: {f1_favor:.3f}\t'
                  f'Test F1_Against: {f1_against:.3f}\tTest F1_Neutral: {f1_neutral:.3f}\n'
                  f'Test F1_Few: {f1_avg_few:.3f}\tTest F1_Favor_Few: {f1_favor_few:.3f}\t'
                  f'Test F1_Against_Few: {f1_against_few:.3f}\tTest F1_Neutral_Few: {f1_neutral_few:.3f}\n'
                  f'Test F1_Zero: {f1_avg_zero:.3f}\tTest F1_Favor_Zero: {f1_favor_zero:.3f}\t'
                  f'Test F1_Against_Zero: {f1_against_zero:.3f}\tTest F1_Neutral_Zero: {f1_neutral_zero:.3f}')

        print('Done\n')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        interval = max(len(self.train_loader) // 10, 1)

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            stances = batch['stances'].to(self.device)
            if self.args.wiki_model and self.args.wiki_model != 'merge':
                input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
            else:
                input_ids_wiki = None
                attention_mask_wiki = None

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
            loss = self.criterion(logits, stances)
            loss.backward()
            if self.args.max_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad)
            self.optimizer.step()

            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i+1}/{len(self.train_loader)}\tLoss:{loss.item():.3f}')

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def eval(self, phase='val', chunk_idx=-1):
        self.model.eval()
        y_pred = []
        y_true = []
        softmax_logits = []
        mask_few_shot = []
        softmax = nn.Softmax(dim=-1)
        val_loader = self.val_loader if phase == 'val' else self.test_loader

        interval = max(len(val_loader) // 10, 1)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['stances']
                if self.args.data == 'vast' and phase == 'test':
                    mask_few_shot_ = batch['few_shot']
                else:
                    mask_few_shot_ = torch.tensor([0])
                if self.args.wiki_model and self.args.wiki_model != 'merge':
                    input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                    attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
                else:
                    input_ids_wiki = None
                    attention_mask_wiki = None
                logits = self.model(input_ids, attention_mask, token_type_ids,
                                    input_ids_wiki=input_ids_wiki, attention_mask_wiki=attention_mask_wiki)
                logits = logits / self.calibration_temperature
                softmax_logits.append(softmax(logits).detach().to('cpu').numpy())
                preds = logits.argmax(dim=1)
                y_pred.append(preds.detach().to('cpu').numpy())
                y_true.append(labels.detach().to('cpu').numpy())
                mask_few_shot.append(mask_few_shot_.detach().to('cpu').numpy())

                if i % interval == 0 or i == len(val_loader) - 1:
                    print(f'Batch: {i + 1}/{len(val_loader)}')

        y_pred = np.concatenate(y_pred, axis=0)
        softmax_logits = np.concatenate(softmax_logits, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # //////////////////
        if phase == 'test':
            import pickle
            if chunk_idx == -1:
                pickle.dump(softmax_logits, open(f'preds_{self.suffix}', 'wb'))
            else:
                pickle.dump(softmax_logits, open(f'preds_{self.suffix}_{chunk_idx}', 'wb'))

        from sklearn.metrics import f1_score, roc_auc_score
        f1_against, f1_favor, f1_neutral = f1_score(y_true, y_pred, average=None)
        f1_avg = (f1_favor + f1_against + f1_neutral) / 3


        # against
        scores = softmax_logits[:, 0]
        labels = y_true.copy()
        labels[y_true == 0] = 1
        labels[y_true != 0] = 0
        roc_auc_against = roc_auc_score(labels, scores)

        # favor
        scores = softmax_logits[:, 1]
        labels = y_true.copy()
        labels[y_true == 1] = 1
        labels[y_true != 1] = 0
        roc_auc_favor = roc_auc_score(labels, scores)

        # neutral
        scores = softmax_logits[:, 2]
        labels = y_true.copy()
        labels[y_true == 2] = 1
        labels[y_true != 2] = 0
        roc_auc_neutral = roc_auc_score(labels, scores)

        roc_auc_avg = (roc_auc_favor + roc_auc_against + roc_auc_neutral) / 3

        return roc_auc_avg, roc_auc_favor, roc_auc_against, roc_auc_neutral

    def calibrate(self):
        model_with_temperature = ModelWithTemperature()
        # model_with_temperature = nn.DataParallel(model_with_temperature)
        model_with_temperature.to(self.device)

        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        interval = max(len(self.val_loader) // 10, 1)
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # input = input.cuda()
                # logits = self.model(input)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['stances']
                if self.args.wiki_model and self.args.wiki_model != 'merge':
                    input_ids_wiki = batch['input_ids_wiki'].to(self.device)
                    attention_mask_wiki = batch['attention_mask_wiki'].to(self.device)
                else:
                    input_ids_wiki = None
                    attention_mask_wiki = None
                logits = self.model(input_ids, attention_mask, token_type_ids, input_ids_wiki=input_ids_wiki,
                                    attention_mask_wiki=attention_mask_wiki)
                logits_list.append(logits.detach().to('cpu'))
                labels_list.append(labels.detach().to('cpu'))

                if i % interval == 0 or i == len(self.val_loader) - 1:
                    print(f'Batch: {i + 1}/{len(self.val_loader)}')

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        token_type_ids = token_type_ids.to('cpu')
        torch.cuda.empty_cache()
        del input_ids, attention_mask, token_type_ids

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([model_with_temperature.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(model_with_temperature.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(model_with_temperature.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(model_with_temperature.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % model_with_temperature.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        self.calibration_temperature = model_with_temperature.temperature.item()

        logits = logits.to('cpu')
        labels = labels.to('cpu')
        torch.cuda.empty_cache()
        del logits, labels

        return self

