
from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
#from attention import AttentionDecoder
import random
import numpy as np
	
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer(num_words=5000)
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

data_folder='data'
#data_folder = '/floyd/input/data'
# load datasets
dataset = load_clean_sentences(data_folder+'/tweet-response-both.pkl')
train = load_clean_sentences(data_folder+'/tweet-response-train.pkl')
test = load_clean_sentences(data_folder+'/tweet-response-test.pkl')






# prepare english tokenizer
tweet_tokenizer = create_tokenizer(dataset[:, 0])
tweet_vocab_size = min(len(tweet_tokenizer.word_index), 5000) + 1
tweet_length = 30
print('Tweet Vocabulary Size: %d' % tweet_vocab_size)
print('Tweet Max Length: %d' % (tweet_length))
# prepare german tokenizer
response_tokenizer = create_tokenizer(dataset[:, 1])
response_vocab_size = min(len(response_tokenizer.word_index), 5000) + 1
response_length = 30
print('Response Vocabulary Size: %d' % response_vocab_size)
print('Response Max Length: %d' % (response_length))
del dataset 

# define model
model = define_model(tweet_vocab_size, response_vocab_size, tweet_length, response_length, 256)
#model = define_non_seq_model(tweet_vocab_size, response_vocab_size, tweet_length, response_length, 256)
adam = Adam(lr=0.0001, decay=0.00001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize defined model
print(model.summary())



print('preparing training data')
# prepare training data
trainX = encode_sequences(tweet_tokenizer, tweet_length, train[:, 0])
trainY = encode_sequences(response_tokenizer, response_length, train[:, 1])
#trainY = encode_output(trainY, response_vocab_size)
# prepare validation data
testX = encode_sequences(tweet_tokenizer, tweet_length, test[:, 0])
testY = encode_sequences(response_tokenizer, response_length, test[:, 1])
#testY = encode_output(testY, response_vocab_size)

np.random.shuffle(trainX)

idx = np.random.choice(np.arange(len(testX)), 1000, replace=False)
batch_testX = testX[idx]
batch_testY = testY[idx]


#batch_testX = testX[:1000]
#batch_testY = testY[:1000]
batch_testY = encode_output(batch_testY, response_vocab_size)


MINI_BATCH_SIZE=1000

for i in range(250):
	batch_start = i * MINI_BATCH_SIZE
	batch_end = batch_start+MINI_BATCH_SIZE
	print('batch_start={} batch_end={}'.format(batch_start, batch_end))
	print('processing batch_train')
	batch_trainX = trainX[batch_start:batch_end]
	batch_trainY = trainY[batch_start:batch_end]
	batch_trainY = encode_output(batch_trainY, response_vocab_size)

	print(batch_trainX.shape)
	print(batch_trainY.shape)
	
	print('starting training')

	#model(model, to_file='model.png', show_shapes=True)
	# fit model
	filename = 'model.h5'
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	model.fit(batch_trainX, batch_trainY, epochs=4, batch_size=64, validation_data=(batch_testX, batch_testY), callbacks=[checkpoint], verbose=2)