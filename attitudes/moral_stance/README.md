
# Morality

## Data preprocessing:

preprocessing.py does the following for each tweet:

- remove URLs
- replace all mentions with "@user"
- remove or split hashtags
- emojis to description
- to lower case
- remove punctuations
- remove non-ascii
- remove emoticons

## Model training and fine-tuning
finetune_transformer.py does the following:

Applies AutoModelForSequenceClassification with a XLM-Roberta model (multi-lingual text embedding) trained on 10K human annotated French tweets. Huggingface's AutoModelForSequenceClassification does the following (text from GPT-4, so task this with a grain of salt):

- Takes the input text and tokenizes it.
- Passes the tokenized input through the base transformer model (e.g., BERT, RoBERTa, etc.), which is a deep neural network that has been pre-trained on a large corpus of text.
- Outputs a vector for each token in the input.
- Passes the output vector of the [CLS] token (the first token for BERT-like models) through a linear layer to obtain the logits (unnormalized scores) for each possible class.
- Optionally applies a softmax function to the logits to convert them into probabilities.

## Execute data cleaning and model training pipeline

run.sh does the follwing:
- Activate Conda instance
- Runs python finetune_transformer.py with "cardiffnlp/twitter-xlm-roberta-base" and several training parameters
