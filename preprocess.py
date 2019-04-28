from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('data/tweet-response.pkl')

# reduce dataset size
#n_sentences = 200																																																																																																																																																																																																																																																																																																																																																																																																									000
#dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(raw_dataset)
# split into train/test
test, train = raw_dataset[:2000], raw_dataset[200:]
# save
save_clean_data(raw_dataset, 'data/tweet-response-both.pkl')
save_clean_data(train, 'data/tweet-response-train.pkl')
save_clean_data(test, 'data/tweet-response-test.pkl')
