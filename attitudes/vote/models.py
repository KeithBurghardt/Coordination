import torch.nn as nn
import torch
import os
import torch.optim as optim
from torch.nn import functional as F


class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model='bert-base', n_layers_freeze=0, wiki_model='', n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel
        if model == 'bert-base':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif model == 'roberta':
            self.bert = AutoModel.from_pretrained('roberta-base')
        elif model == 'bertweet':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        elif model == 'covid-twitter-bert':
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        elif model == 'bertweet-covid':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-covid19-base-uncased')
        elif model == 'bertweetfr':
            self.bert = AutoModel.from_pretrained('Yanzhu/bertweetfr-base')
        elif model == 'bert-small':
            self.bert = AutoModel.from_pretrained('prajjwal1/bert-small')
        else: # xlm-t
            self.bert = AutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')

        n_layers = 4 if model == 'bert-small' else 12

        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        if wiki_model:
            if wiki_model == model or wiki_model == 'merge':
                self.bert_wiki = self.bert
            else:
                from transformers import AutoModel
                if wiki_model == 'bert-base':
                    self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')
                elif wiki_model == 'roberta':
                    self.bert_wiki = AutoModel.from_pretrained('roberta-base')
                elif wiki_model == 'bertweet':
                    self.bert_wiki = AutoModel.from_pretrained('vinai/bertweet-base')
                elif wiki_model == 'covid-twitter-bert':
                    self.bert_wiki = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
                elif wiki_model == 'bertweet-covid':
                    self.bert_wiki = AutoModel.from_pretrained("vinai/bertweet-covid19-base-uncased")
                elif wiki_model == 'bertweetfr':
                    self.bert_wiki = AutoModel.from_pretrained('Yanzhu/bertweetfr-base')
                else:  # bert-small
                    self.bert_wiki = AutoModel.from_pretrained('prajjwal1/bert-small')

            n_layers = 4 if wiki_model == 'bert-small' else 12

            if n_layers_freeze_wiki > 0:
                n_layers_ft = n_layers - n_layers_freeze_wiki
                for param in self.bert_wiki.parameters():
                    param.requires_grad = False
                for param in self.bert_wiki.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                    for param in self.bert_wiki.encoder.layer[i].parameters():
                        param.requires_grad = True

        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if wiki_model and wiki_model != 'merge':
            hidden = config.hidden_size + self.bert_wiki.config.hidden_size
        else:
            hidden = config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                input_ids_wiki=None, attention_mask_wiki=None):
        if self.model not in ['roberta', 'bertweetfr', 'xlm-t']:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        if input_ids_wiki is not None:
            outputs_wiki = self.bert_wiki(input_ids_wiki, attention_mask=attention_mask_wiki)
            pooled_output_wiki = outputs_wiki.pooler_output
            pooled_output_wiki = self.dropout(pooled_output_wiki)
            pooled_output = torch.cat((pooled_output, pooled_output_wiki), dim=1)
        logits = self.classifier(pooled_output)
        return logits


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
