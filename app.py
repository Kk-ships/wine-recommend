import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification

import config


def run():
    # Title
    st.title( "Get wine recommendations using Machine learning and AI" )
    st.markdown( """
        	#### Description
        	+ This is a Natural Language Processing(NLP) Based wine recommendation app. This model uses a fine tuned DistilBERT Langauge
        	 model to generate the embeddings for the reviews of about 130 thousdand types of wine all around the world. 
        	+ When the user enters a wine whose flavor profile he prefers then model again generates a sentence embedding for that 
        	input.
        	+ Using cosine similarity we recommend top 3 wines that user should taste.
        	
        	""" )

    st.sidebar.subheader( "By" )
    st.sidebar.text( "Kaustubh Shirpurkar" )
    st.sidebar.text( "kaustubh.shirpurkar@gmail.com" )

    st.subheader( "Get wine recommendation" )

    message = st.text_area( "Enter Text", "Type Here .. (E.g. Vintage, woody, fruity, apple etc.)" )
    if st.button( "Get recommendations" ):
        processing( message )



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
    vector_rep = vector_rep.cpu().detach().numpy()  # move to cpu and convert to numpy
    norm = np.linalg.norm( vector_rep )  # for normalizing the vector
    return vector_rep / norm


def processing(user_input):
    df = pd.read_csv( 'winemag-data-130k-v2.csv' )
    train_df = df[['description', 'title', 'points']]
    embeddings = np.load( 'Text_embeddings.npy' )
    user_embedding = get_embedding( user_input )
    result = np.matmul( embeddings, user_embedding )
    top_3_row_nums = result.argsort()[-3:][::-1]

    wines_recommended = dict( dict() )
    try:
        for i, row in enumerate( top_3_row_nums ):
            st.write(f'Name: {train_df.iloc[row].title} \n Review: {train_df.iloc[row].description} \n Review score: {int(train_df.iloc[row].points)} \n Country of origin: {df.iloc[row].country}')
    except AttributeError:
        print(" I think I made a mistake somewhere. ")
        raise

    return None


if __name__ == '__main__':
    MODEL = DistilBertForSequenceClassification.from_pretrained( config.MODEL_PATH, output_hidden_states=True )
    MODEL.to( config.DEVICE )
    MODEL.eval()
    run()
