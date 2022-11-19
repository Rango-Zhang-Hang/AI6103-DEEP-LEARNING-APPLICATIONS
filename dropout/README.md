# Effect of Dropout on Sentiment Classification on IMDB reviews dataset using BERT model

In this experiment, the effect of dropout is studied on the BERT model on sentiment classification on IMDB reviews dataset. The dropout probabilty is varied from 0 to 0.9, in steps of 0.1.

## Getting started
Install the dependencies:
```
pip install -r requirements.txt
```

## Training
Run with desired dropout probability:
```
python imdb_dropout.py -dropout 0.5
```

The following files are generated:
- best model (based on best val acc)
- final model
- accuracy plot (train and val)
- loss plot (train and val)
- training log
- validation log
- confusion matrix of validation data