import numpy as np
import pandas as pd
from telegram import ChatAction, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from functools import wraps
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from secrets import token
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case = True)
# Loading model and setting output hidden states flag to true
model = DistilBertForSequenceClassification.from_pretrained( './model/', output_hidden_states=True )


def text_to_embedding(tokenizer, model, input_text, device):
    model.to(device)
    input_text = str(input_text)
    # Step 1 = Tokenization
    MAX_LEN = 128  # Max token length
    input_ids = tokenizer.encode(
        input_text,  # description
        add_special_tokens=True,  # add [CLS] and [SEP]
        max_length=MAX_LEN,
        truncation=True
    )
    result = pad_sequences( [input_ids], maxlen=MAX_LEN, dtype='long',
                            value=tokenizer.pad_token_id, truncating='post', padding='post' )
    input_ids = result[0]
    attn_mask = [int( token_id > 0 ) for token_id in input_ids]

    # to tensor
    input_ids = torch.tensor( input_ids ).to(torch.int64)
    attn_mask = torch.tensor( attn_mask ).to(torch.int64)

    # Adding extra dimension for processing
    input_ids = input_ids.unsqueeze( 0 )
    attn_mask = attn_mask.unsqueeze( 0 )

    # to GPU
    input_ids = input_ids.to( device )
    attn_mask = attn_mask.to( device )
    # step 2 evaluaring text on DistilBERT model
    model.eval()
    # No gradient tracking mode
    with torch.no_grad():
        _, encoded_layer = model( input_ids=input_ids, attention_mask=attn_mask )
    layer_i = 6  # last layer for DistilBERT
    batch_i = 0  # Only One input in the batch
    token_i = 0  # Get token for the [CLS]
    # print(encoded_layer[layer_i][batch_i][token_i])

    # Get sentece emedding
    vector_rep = encoded_layer[layer_i][batch_i][token_i]
    vector_rep = vector_rep.numpy()  # move to cpu and convert to numpy
    norm = np.linalg.norm( vector_rep )  # for normalizing the vector
    return vector_rep / norm
# Sends typing when bot is replying
def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(update, context, *args, **kwargs)

        return command_func

    return decorator

send_typing_action = send_action(ChatAction.TYPING)

def send_message(context = None,update = None,text=None):
    return context.bot.send_message( chat_id=update.effective_chat.id, text=text)
# start command handler
@send_typing_action  # Return typing when processing a request
def start(update, context):
    send_message(context = context,update = update,text= "Welcome to Wine recommendation bot!")
    send_message(context = context,update = update,text= "This bot recommends you wines based on your"
                                                         " likings and taste using machine learning and AI")
    send_message(context = context,update = update,text= "Lets get started then shall we?" )
    send_message(context = context,update = update,text= "Enter what type of flavor profile you would like"
                                                         " in your wine? e.g. Vintage, old, fruity, white, red etc ")
# Unknown commands handler (Default reply to unknown command) to be placed at the end
@send_typing_action  # Return typing when processing a request
def unknown(update, context):
    send_message(context = context,update = update,text= "Sorry, I didn't understand that command. Send /start to get more options")



@send_typing_action  # Return typing when processing a request
def recommend(update, context):
    user_input = update.message.text
    processing(update, user_input)


def processing(update, user_input):
    if torch.cuda.is_available():
        # set pytorch to use gpu
        device = torch.device( "cuda" )
    else:
        device = torch.device( 'cpu' )
    df = pd.read_csv('winemag-data-130k-v2.csv')
    train_df = df[['description', 'title', 'points']]
    embeddings = np.load('Text_embeddings.npy')
    user_embedding = text_to_embedding( tokenizer, model, user_input, device )
    result = np.matmul( embeddings, user_embedding)
    top_5_row_nums = result.argsort()[-5:][::-1]
    update.message.reply_text("====== Top 5 wines recommended for you are ======")
    for row in top_5_row_nums:
        update.message.reply_text('Name - ', train_df.iloc[row].title)
        update.message.reply_text('Review - ' , train_df.iloc[row].description)

    return None
def main():
    updater = Updater( token=token, use_context=True )
    dp = updater.dispatcher
    dp.add_handler( CommandHandler( 'start', start ) )
    dp.add_handler( MessageHandler( Filters.text, recommend) )
    dp.add_handler( MessageHandler( Filters.command, unknown ) )
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()