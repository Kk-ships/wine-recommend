from functools import wraps

import numpy as np
import pandas as pd
import torch
from telegram import ChatAction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from transformers import DistilBertForSequenceClassification

import config
from secrets import token


def get_embedding(user_input):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    user_input = str( user_input )
    user_input = " ".join( user_input.split() )
    inputs = tokenizer.encode_plus(
        user_input, None, add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
    )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    ids = torch.tensor( ids, dtype=torch.long ).unsqueeze( 0 )
    mask = torch.tensor( mask, dtype=torch.long ).unsqueeze( 0 )

    ids = ids.to( config.DEVICE, dtype=torch.long )
    mask = mask.to( config.DEVICE, dtype=torch.long )
    with torch.no_grad():
        outputs = MODEL( input_ids=ids, attention_mask=mask )

    layer_i = 6  # last layer for DistilBERT
    batch_i = 0  # Only One input in the batch
    token_i = 0  # Get token for the [CLS]
    # print(encoded_layer[layer_i][batch_i][token_i])
    encoded_layer = outputs.hidden_states
    # Get sentece emedding
    vector_rep = encoded_layer[layer_i][batch_i][token_i]
    vector_rep = vector_rep.cpu().detach().numpy() # move to cpu and convert to numpy
    norm = np.linalg.norm( vector_rep )  # for normalizing the vector
    return vector_rep / norm


# Sends typing when bot is replying
def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps( func )
        def command_func(update, context, *args, **kwargs):
            context.bot.send_chat_action( chat_id=update.effective_message.chat_id, action=action )
            return func( update, context, *args, **kwargs )

        return command_func

    return decorator


send_typing_action = send_action( ChatAction.TYPING )


def send_message(context=None, update=None, text=None):
    return context.bot.send_message( chat_id=update.effective_chat.id, text=text )


# start command handler
@send_typing_action  # Return typing when processing a request
def start(update, context):
    send_message( context=context, update=update, text="Welcome to Wine recommendation bot!" )
    send_message( context=context, update=update, text="This bot recommends you wines based on your"
                                                       " likings and taste using machine learning and AI" )
    send_message( context=context, update=update, text="Lets get started then shall we?" )
    send_message( context=context, update=update, text="Enter what type of flavor profile you would like"
                                                       " in your wine? e.g. Vintage, old, fruity, white, red etc " )


# Unknown commands handler (Default reply to unknown command) to be placed at the end
@send_typing_action  # Return typing when processing a request
def unknown(update, context):
    send_message( context=context, update=update,
                  text="Sorry, I didn't understand that command. Send /start to get more options" )


@send_typing_action  # Return typing when processing a request
def recommend(update, context):
    user_input = update.message.text
    processing( update, user_input )


def processing(update, user_input):
    df = pd.read_csv( 'winemag-data-130k-v2.csv' )
    train_df = df[['description', 'title', 'points']]
    embeddings = np.load( 'Text_embeddings.npy' )
    user_embedding = get_embedding( user_input )
    result = np.matmul( embeddings, user_embedding )
    top_3_row_nums = result.argsort()[-3:][::-1]
    update.message.reply_text( "====== Top 3 wines recommendations ======" )
    for i, row in enumerate(top_3_row_nums):
        update.message.reply_text( f'({i+1}) -  {train_df.iloc[row].title}' )
        update.message.reply_text( f'Review - {train_df.iloc[row].description}' )
        update.message.reply_text( f'Points by the reviewer - {train_df.iloc[row].points}' )

    return None


def run():
    updater = Updater( token=token, use_context=True )
    dp = updater.dispatcher
    dp.add_handler( CommandHandler( 'start', start ) )
    dp.add_handler( MessageHandler( Filters.text, recommend ) )
    dp.add_handler( MessageHandler( Filters.command, unknown ) )
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    MODEL = DistilBertForSequenceClassification.from_pretrained( config.MODEL_PATH, output_hidden_states=True )
    # MODEL.load_state_dict( torch.load( config.MODEL_PATH ) )
    MODEL.to( config.DEVICE )
    MODEL.eval()
    run()
