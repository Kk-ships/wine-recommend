from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


DEVICE = "cpu"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 5
NUM_LABELS = 21
BERT_PATH = "distilbert-base-uncased"
MODEL_PATH = "./model/"
TRAINING_FILE = "winemag-data-130k-v2.csv"
TOKENIZER = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MODEL = DistilBertForSequenceClassification.from_pretrained(
        BERT_PATH,  # use 6 layer base Distil-BERT with uncased vocab
        num_labels=NUM_LABELS,  # Linear regression unique points
        output_attentions=False,  # Do not return attention weights
        output_hidden_states=False )  # do not retun all hidden states