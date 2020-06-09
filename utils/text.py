import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertModel.from_pretrained('bert-base-multilingual-cased')


def tokenize(sentence, hidden_state=1):
    input_ids = tf.constant(tokenizer.encode(sentence))[None, :]  # Batch size 1
    outputs = model(input_ids)
    return outputs[hidden_state]
