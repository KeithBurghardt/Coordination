# Who to vote for
## What this code does
This code extracts whether the tweet advocates for someone to vote for a candidate or against a candidate (or both).

- Uses XLM-T model (rather than original BERT model) for [WS-BERT](https://aclanthology.org/2022.wassa-1.7/)
- WS-BERT (we can also call this WS-XLM-T) is trained on COVID-19-Stance dataset
- Model is fine-tuned on 10K human annotations of French tweets

## How to run

To run the model, just run "python run.py" on Python3.10

Prerequisite libraries:
- torch==2.0
- pandas==2.0.1
- numpy==1.24.3

The full pipeline can be run with run.py

## Files

- datasets.py: extract data for training
- engine.py: all the training/testing is done here
- models.py: all models are defined here
- train.py: train model
- test.py: apply trained model for testing
- run.py: run full pipeline (training/testing parameters are chosen here)


