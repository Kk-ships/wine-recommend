import pandas as pd
import torch
from sklearn import metrics
from sklearn import model_selection
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine


def run():
    df = pd.read_csv( config.TRAINING_FILE ).fillna( "none" )
    df['points'] = df['points'] - 80  # setting points from 0 to 21
    df = df[['description', 'title', 'points']]
    df = df.sample(10)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=1234,
        # stratify=df.points.values
    )

    df_train = df_train.reset_index( drop=True )
    df_valid = df_valid.reset_index( drop=True )

    train_dataset = dataset.BERTDataset( description=df_train.description.values, points=df_train.points.values )

    train_data_loader = DataLoader( train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4 )

    valid_dataset = dataset.BERTDataset( description=df_valid.description.values, points=df_valid.points.values )

    valid_data_loader = DataLoader( valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4 )

    device = torch.device( config.DEVICE )
    model = DistilBertForSequenceClassification.from_pretrained(
        config.BERT_PATH,  # use 6 layer base Distil-BERT with uncased vocab
        num_labels=config.NUM_LABELS,  # Linear regression unique points
        output_attentions=False,  # Do not return attention weights
        output_hidden_states=False )  # do not retun all hidden states
    model.to( device )

    param_optimizer = list( model.named_parameters() )
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any( nd in n for nd in no_decay )
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any( nd in n for nd in no_decay )
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int( len( df_train ) / config.TRAIN_BATCH_SIZE * config.EPOCHS )
    optimizer = AdamW( optimizer_parameters, lr=3e-5 )
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=num_train_steps )

    best_accuracy = 0
    for epoch in range( config.EPOCHS ):
        engine.train_fn( train_data_loader, model, optimizer, device, scheduler )
        outputs, targets = engine.eval_fn( valid_data_loader, model, device )
        accuracy = metrics.accuracy_score( targets, outputs )
        print( f"Accuracy Score = {accuracy}" )
        if accuracy > best_accuracy:
            torch.save( model.state_dict(), config.MODEL_PATH )
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
