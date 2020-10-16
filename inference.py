import pickle
import tensorflow as tf
from util import create_padding_mask, create_look_ahead_mask
from extract_data import Extract_Data
from transformer import Transformer
from util import create_padding_mask, create_look_ahead_mask
import numpy as np
import time
# -----------------------------------------
encoder_maxlen = 400
decoder_maxlen = 75
# -----------------------------------------
# STEP-1:
# Load tokenizers
doc_pklfile = 'document_tokenizer_pickle'
sum_pklfile = 'summary_tokenizer_pickle'

doc_fpkl = open(doc_pklfile, 'rb')
sum_fpkl = open(sum_pklfile, 'rb')

document_tokenizer = pickle.load(doc_fpkl)
summary_tokenizer = pickle.load(sum_fpkl)

doc_fpkl.close()
sum_fpkl.close()

# -------------------------------------------------------------
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
encoder_vocab_size, decoder_vocab_size = 76362, 29661

transformer = Transformer(
    num_layers, 
    d_model, 
    num_heads, 
    dff,
    encoder_vocab_size, 
    decoder_vocab_size, 
    pe_input=encoder_vocab_size, 
    pe_target=decoder_vocab_size,
)
# -------------------------------------------------------------
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

# -------------------------------------------------------------
def evaluate(input_document):
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

# 
def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document


# 
if __name__ == '__main__':
    x = input('Enter 1 for url, 2 for text: ')

    if(int(x) == 1):
        url = input('Enter url: ')
        data = Extract_Data(url)
    elif(int(x) == 2):
        data = input('Enter data:\n')

    else:
        data = ""

    data = '''Nitish Kumar-led Bihar government has reduced the hike in power tariff to 28% after an earlier 55% hike by Bihar Electricity Regulatory 
    Commission (BERC) sparked protests by the Opposition. The hike will be limited to 28% after the government announced that it will continue the 
    subsidy in the power sector which would go directly into the bank accounts of consumers'''

    # start_time = time.time()
    if len(data) > 1:
        summary = summarize(data)
        # end_time = time.time()
        # print('Document: ', data)
        print('Summary: ', summary)
        # print('Inference time: ', end_time-start_time)
    else:
        print('Wrong Input')