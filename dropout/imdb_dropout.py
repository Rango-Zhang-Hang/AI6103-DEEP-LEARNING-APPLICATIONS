"""
Adapted from https://www.kaggle.com/code/houssemayed/imdb-sentiment-classification-with-bert

Perform sentiment classification on IMDB reviews dataset using BERT model (bert-base-cased)

Generated files:
- best model (based on best val acc)
- final model
- acc plot
- loss plot
- training log
- validation log
- confusion matrix of validation data
"""


import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class IMDBDataset(Dataset):

  def __init__(self, reviews, sentiments, tokenizer, max_len):
    self.reviews = reviews
    self.sentiments = sentiments
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    sentiment = self.sentiments[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      # pad_to_max_length = True,
      truncation = True,
      padding = 'max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'sentiments': torch.tensor(sentiment, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = IMDBDataset(
    reviews=df.review.to_numpy(),
    sentiments=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


class IMDBClassifier(nn.Module):

  def __init__(self, n_classes, dropout_value):
    super(IMDBClassifier, self).__init__()
    self.bert = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=dropout_value)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = output.pooler_output

    output = self.drop(pooled_output)
    return self.out(output)


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
    ):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    sentiments = d["sentiments"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, sentiments)

    correct_predictions += torch.sum(preds == sentiments)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiments"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, sentiments)

      correct_predictions += torch.sum(preds == sentiments)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
  model = model.eval()

  review = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      reviews = d["review"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      sentiments = d["sentiments"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review.extend(reviews)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(sentiments)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix, dropout_value):
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ['{0:0.0f}'.format(value) for value in
                  confusion_matrix.flatten()]
  group_percentages = ['{0:.2%}'.format(value) for value in
                      confusion_matrix.flatten()/np.sum(confusion_matrix)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')
  plt.savefig(f'results/confusion_matrix_{dropout_value}.jpg')
  plt.clf()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Sentiment Classification on IMDB reviews dataset using BERT', add_help=False)
  parser.add_argument('-dropout', required=True, help='dropout value')
  args = parser.parse_args()
  dropout_value = float(args.dropout)

  if not os.path.exists('results'):
    os.makedirs('results')
    print('create directory results')

  start_time = datetime.now()

  # import data
  csv_path = "IMDB Dataset.csv"
  df = pd.read_csv(csv_path)
  df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

  #Selecting the bert-base-cased
  PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

  MAX_LEN = 200      #to not consume too much resources
  RANDOM_SEED = 42
  BATCH_SIZE = 16
  EPOCHS = 10

  device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

  #doing the split of the dataset into training, validation and testing sets
  df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

  train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

  # initialise model
  model = IMDBClassifier(len(df['sentiment'].unique()), dropout_value)
  model = model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  loss_fn = nn.CrossEntropyLoss().to(device)


  # TRAINING
  train_a = []
  train_l = []
  val_a = []
  val_l = []
  best_accuracy = 0

  for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
      device,
      len(df_val)
    )

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

    train_a.append(train_acc.item())
    train_l.append(train_loss.item())
    val_a.append(val_acc.item())
    val_l.append(val_loss.item())

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f'results/best_model_state_{dropout_value}.pt')
      best_accuracy = val_acc


  torch.save(model.state_dict(), f'results/model_final_{dropout_value}.pt')
  print('save model')

  print('training accuracy')
  print(train_a)

  print('val accuracy')
  print(val_a)

  print('training loss')
  print(train_l)

  print('val loss')
  print(val_l)

  # plot accuracy graph
  plt.plot(train_a, label='train accuracy')
  plt.plot(val_a, label='validation accuracy')

  plt.title('Accuracy against epoch')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.savefig(f'results/acc_plot_{dropout_value}.jpg')
  plt.clf()

  # plot loss graph
  plt.plot(train_l, label='train loss')
  plt.plot(val_l, label='validation loss')

  plt.title('Loss against epoch')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend()
  plt.savefig(f'results/loss_plot_{dropout_value}.jpg')
  plt.clf()


  # save training log to csv
  df_train = pd.DataFrame(list(zip([i for i in range(EPOCHS)], train_a, train_l, val_a, val_l)),
                columns =['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss'])
  df_train.to_csv(f'results/training_log_{dropout_value}.csv', index=False)


  # analyse the results from validation dataset
  y_review_texts, y_pred, y_pred_probs, y_val = get_predictions(
    model,
    val_data_loader
  )

  # show confusion matrix of val dataset
  class_names = ['negative', 'positive']
  print(classification_report(y_val, y_pred, target_names=class_names))

  cm = confusion_matrix(y_val, y_pred)
  show_confusion_matrix(cm, dropout_value)

  # save validation log to csv
  confidence_score, _ = torch.max(y_pred_probs, dim=1)
  df_test = pd.DataFrame({'review': y_review_texts, 'ground truth': y_val, 'prediction': y_pred, 
                          'correct prediction': torch.eq(y_val, y_pred), 'confidence score': confidence_score})
  df_test.to_csv(f'results/val_log_{dropout_value}.csv', index=False)


  end_time = datetime.now()
  print(f'Time taken: {end_time-start_time}')
