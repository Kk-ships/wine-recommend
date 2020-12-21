from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


DEVICE = "cpu"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 5
NUM_LABELS = 21
BERT_PATH = "distilbert-base-uncased"
MODEL_PATH = "/model/pytorch_model.bin"
TRAINING_FILE = "winemag-data-130k-v2.csv"
TOKENIZER = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
