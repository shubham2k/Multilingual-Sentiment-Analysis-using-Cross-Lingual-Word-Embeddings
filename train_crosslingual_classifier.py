import gensim
import os
import sys
import numpy as np
import pickle
import data_manager
from argparse import ArgumentParser
#from utils import embeddings,neuralnets
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict
### keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Bidirectional, LSTM, Activation, MaxPooling1D, Conv1D, Dropout, Embedding, ActivityRegularization, concatenate
from keras.models import Model, Sequential

def load(data_folder):
    # d = {}
    d = defaultdict(list)
    # your code here
    # the returned dictionary should be of the form:
    # {'pos': [
    #			['word1', 'word2', word3, ... ], # each of these nested lists of strings are imdb reviews
    #			['word1', 'word2', word3, ... ]
    # 			... }
    # {'neg': [
    #			['word1', 'word2', word3, ... ],
    #			['word1', 'word2', word3, ... ]
    # 			... }
    files = os.listdir(data_folder)
    for infile in files:
        f = open(os.path.join(data_folder, infile), 'r', encoding='utf-8')  #  (Windows)
        documents = []
        for line in f:
            words = [word for word in line.split()]
            documents.append(words)
        d[infile] = documents
    # process the review file 'f' and populate the dictionary 'd'
    return d

def get_maxlen(dataset):
	maxlen = 0
	for label in dataset:
		for doc in dataset[label]:
			doclen = len(doc)
			if doclen > maxlen:
				maxlen = doclen
	return maxlen

### HELPERS TO SAVE AND LOAD KERAS TOKENIZER ###
def save_tokenizer(tk, outpath):
	with open(outpath, 'wb') as handle:
	    pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('Tokenizer: ',tk,' | Saved to: ',outpath)

def load_tokenizer(tokenizer_path):
	print('Loading tokenizer from: ',tokenizer_path)
	with open(tokenizer_path, 'rb') as handle:
		tk = pickle.load(handle)
		return tk

def get_labels(data):
	label2id = {}
	for label in data:
		if not label in label2id:
			label2id[label] = len(label2id)
	return label2id, dict([(b,a) for a,b in label2id.items()])

def merge_models(tk,model1,model2):                #
	outm = {}
	for w in tk.word_index:
		if w in model1.vocab and not w in model2.vocab:
			outm[w] = model1[w]
		elif not w in model1.vocab and w in model2.vocab:
			outm[w] = model2[w]
		elif w in model1.vocab and w in model2.vocab:
			outm[w] = model1[w]+model2[w]
		else:
			print('Found OOV (not in src or tgt vocab): ',w)
	return outm      #dictionary which contains the word,corresponding vector

def create_em(tk,vector_size,embedding_matrix,maxlen):
    # build word embedding layer based on dataset vocabulary
    word_index=tk.word_index
    out_embedding_matrix = np.zeros((len(word_index)+1, vector_size))
    for word, index in word_index.items():
    	if word in embedding_matrix:
        	out_embedding_matrix[word_index[word]]=embedding_matrix[word]  #{just vectors where the ith index corresponds to the tokeniser index for that word_vector }
    em_layer = Embedding(len(word_index) + 1,
                                vector_size,
                                weights=[out_embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    return em_layer

class CBLSTM:
	def __init__(self,labels_idx, em_layer):
		# cnn config
		self.FILTERS=10
		self.KERNEL_SIZE=3
		self.POOL_SIZE=4
		self.STRIDES=1
		self.PADDING='valid'
		self.CNN_ACTIVATION='relu'
		# lstm config
		self.LSTM_SIZE=100
		# global
		self.OPTIMIZER='adam'
		self.labels_idx = labels_idx
		self.em_layer = em_layer
	def build_model(self, cblstm = True):
		# last (prediction) layer hyperparams
		softmax_size=len(self.labels_idx)
		nnmodel = Sequential()
		nnmodel.add(self.em_layer)
		nnmodel.add(Dropout(0.25));		if cblstm:
			nnmodel.add(Conv1D(self.FILTERS,
			                 self.KERNEL_SIZE,
			                 padding=self.PADDING,
			                 activation=self.CNN_ACTIVATION,
			                 strides=self.STRIDES))
			nnmodel.add(MaxPooling1D(pool_size=self.POOL_SIZE))
		nnmodel.add(Bidirectional(LSTM(self.LSTM_SIZE)))
		nnmodel.add(Dropout(0.25))
		nnmodel.add(Dense(softmax_size))
		nnmodel.add(Activation('softmax'))
		nnmodel.compile(loss='categorical_crossentropy',
		              optimizer=self.OPTIMIZER,
		              metrics=['accuracy'])
		print('=== MODEL SUMMARY ===')
		print(nnmodel.summary())
		return nnmodel

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-srcv', '--src-word-vectors', help='Source word vectors file', required=True)
	parser.add_argument('-tgtv', '--tgt-word-vectors', help='Target word vectors file', required=True)
	parser.add_argument('-d', '--dataset', help='Tab-separated dictionary (DLE only!) path', required=True)
	parser.add_argument('-om', '--output-model', help='Output file to store the keras classifier', required=True)
	parser.add_argument('-ot', '--output-tokenizer', help='Output file to save keras trained tokenizer', required=True)
	parser.add_argument('-ol', '--output-labelmap', help='Output labelemap file', required=True)

	args = parser.parse_args()
	srcmodel = gensim.models.KeyedVectors.load_word2vec_format(args.src_wv)
	tgtmodel = gensim.models.KeyedVectors.load_word2vec_format(args.tgt_wv)

	print(f'loading {args.dataset}')
	data = load(args.dataset)

	if not args.find_maxlen == None:
		maxlen = get_maxlen(data)
		print(f'dataset maxlen: {maxlen} (from data)')
	else:
		maxlen = 300		
		print(f'maxlen not obtained from data, defaulting to {maxlen}')

	
	label2id,id2label = get_labels(data)
	print(f'label2id and id2label: {label2id,id2label}')

	docs = []
	labels = []
	for label in data:
		for d in data[label]:
			docs.append(d)
			labels.append(label2id[label])

	if not os.path.exists(args.output_tokenizer):
		t = Tokenizer()
		print('No tokenizer found, fitting to train set')
		t.fit_on_texts(docs)
		t.fit_on_texts(list(tgtmodel.vocab))     #tokeniser creates a dict of word, rank  (rank is decided by the frequency of words occuring)
		save_tokenizer(t, args.output_tokenizer)
	else:
		#print('Using tokenizer found at ',tokenizer_path)
		t = load_tokenizer(args.output_tokenizer)

	seqs = t.texts_to_sequences(docs)
	X_train = pad_sequences(seqs, maxlen=maxlen, padding='post')
	y_train = to_categorical(np.asarray(labels))
	print('Shape of train data tensor:', X_train.shape)
	print('Shape of train label tensor:', y_train.shape)

	print('Merging two cross-lingual embeddings')
	mapped = merge_models(t,srcmodel,tgtmodel)

	# create embedding layer
	em_layer=create_em(t,srcmodel.vector_size,mapped,maxlen)

	nnmodel = CBLSTM(label2id, em_layer)
	# build and compile cnn + lstm classifier
	clf = nnmodel.build_model(cblstm=True)

	epochs = 10
	batch_size = 8
	clf.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, shuffle=True)
		
	clf.save(args.output_model)
	print('Classifier saved to: ',args.output_model)

	with open(args.output_labelmap,'w') as outf:
		for k,v in label2id.items():
			outs = f'{k}\t{v}\n'
			outf.write(outs)
