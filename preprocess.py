import pandas as pd
import numpy as np
import tensorflow as tf
import time
import re
import pickle

# -------------------------------------------------------------------
def PreProcess():
# STEP-1:
	# Read data
	news = pd.read_csv(r"news.csv", engine = 'python', nrows=5000)
	news.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
	# news.head()
	# news.info()

# STEP-2:
	# Collect document and summary from data
	document = news['Short']
	summary = news['Headline']

# STEP-3:
	# add tokens <go> and <stop> to target summary
	summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')
	# summary.head()

# STEP-4:
	# don't remove < and > due <go> and <stop>
	filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
	oov_token = '<unk>'

# STEP-5:
	# initialize tokenizer
	document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
	summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)

# STEP-6:
	# tokenize text and convert text to sequences
	document_tokenizer.fit_on_texts(document)
	summary_tokenizer.fit_on_texts(summary)
	inputs = document_tokenizer.texts_to_sequences(document)
	targets = summary_tokenizer.texts_to_sequences(summary)

# STEP-7:
	# vocabulary size
	encoder_vocab_size = len(document_tokenizer.word_index) + 1
	decoder_vocab_size = len(summary_tokenizer.word_index) + 1

# STEP-8:
	# 
	document_lengths = pd.Series([len(x) for x in document])
	summary_lengths = pd.Series([len(x) for x in summary])

# STEP-9:
	# Set encoder decoder size
	encoder_maxlen = 400
	decoder_maxlen = 7

# STEP-10:
	# covert to fix length input by padding/truncating for generialized input to model
	inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
	targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

# STEP-11:
	# cast to tf data (tensors)
	inputs = tf.cast(inputs, dtype=tf.int32)
	targets = tf.cast(targets, dtype=tf.int32)

# STEP-12:
	# 
	BUFFER_SIZE = 20000
	BATCH_SIZE = 64

# STEP-13:
	# 
	dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# STEP-14:
	# save tokenizers
	doc_pklfile = 'document_tokenizer_pickle'
	sum_pklfile = 'summary_tokenizer_pickle'
	doc_fpkl = open(doc_pklfile, 'wb')
	sum_fpkl = open(sum_pklfile, 'wb')
	document_tokenizer = pickle.dump(document_tokenizer ,doc_fpkl)
	summary_tokenizer = pickle.dump(summary_tokenizer ,sum_fpkl)
	doc_fpkl.close()
	sum_fpkl.close()
	
# STEP-15:
	# 
	print('PreProcess Dataset Complete')
	return dataset, encoder_vocab_size, decoder_vocab_size


