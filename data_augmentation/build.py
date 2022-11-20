import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw


from data import create_data_loader
from model import IMDBClassifier, train_n_epochs


RANDOM_SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PRE_TRAINED_MODEL_NAME = 'bert-base-cased/'
MAX_LEN = 200      #for not consuming much resources
BATCH_SIZE = 16

def get_dataset():
  df = pd.read_csv("imdb-dataset-of-50k-movie-reviews.csv")
  df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
  return df


def augment_data(augmenter, df_train, col='review', frac=0.1):
  if augmenter:
    df_train_aug = df_train.sample(frac=frac, replace=False, random_state=RANDOM_SEED)
    aug_text = []
    for review in df_train_aug[col]:
      aug_text.append(augmenter.augment(review))
    df_train_aug[col] = aug_text

    df_train = pd.concat([df_train, df_train_aug], ignore_index=True)
  return df_train


def experiment_run(df_train, df_val):
  # Load data into data loader
  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

  # Get model
  model = IMDBClassifier(len(df['sentiment'].unique()))
  model = model.to(device)

  EPOCHS = 10

  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  loss_fn = nn.CrossEntropyLoss().to(device)

  train_a, train_l, val_a, val_l, best_accuracy = train_n_epochs(
    model, train_data_loader, val_data_loader, loss_fn, optimizer, device,
    scheduler, df_train, df_val, EPOCHS)
  
  return train_a, train_l, val_a, val_l, best_accuracy


if __name__ == '__main__':
  # Data exploration & visualization

  # We use the IMDB reviews dataset available from kaggle datasets through this link:
  # https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  df = get_dataset()

  # Split dataset into training, validation and testing sets
  df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

  # Choose augmenter
  augmenters = {
    'no_aug': None,
    'keyboard_aug': nac.KeyboardAug(),
    'synonym_aug': naw.SynonymAug(aug_src='wordnet'),
    'wordembs_aug': naw.WordEmbsAug(
      model_type='word2vec', model_path='GoogleNews-vectors-negative300.bin',
      action="insert")
  }

  for aug_name, augmenter in augmenters.items():
    print(f"Experiment run: {aug_name}")
    now = datetime.now()
    df_train_aug = augment_data(augmenter, df_train)

    train_a, train_l, val_a, val_l, best_accuracy = experiment_run(df_train_aug, df_val)
    print("Best accuracy")
    print(best_accuracy)
    df_result = pd.DataFrame(data={
      'train_acc': train_a,
      'train_loss': train_l,
      'val_acc': val_a,
      'val_loss': val_l
    })
    df_result.to_csv(f'{aug_name}_result.csv', index=False)
    print(f"Time taken: {datetime.now() - now}")
    print('-'*20)
