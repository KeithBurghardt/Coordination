
# Concerns model

## How to run
- finetune.py: Train model
- predict_new.py: Make predictions on data
## File details
- finetune.py: We apply train BERTweetFR embedding models via AutoModelForSequenceClassification with 10K human annotated concerns. 
- predict_new.py: For each conern, load the appropriate model, annotate new data
