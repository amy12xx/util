'''
Implementation of various term weighting techniques mentioned in these papers:
https://arxiv.org/pdf/1610.03106v1.pdf
http://www.aclweb.org/anthology/P10-1141

Results of Sentiment Analysis on Farmers insurance dataset:
-----------------------------------------------------------
Accuracy with tp-unary : 0.815
Accuracy with tf-idf : 0.817
Accuracy with bm25-bm25 : 0.816
Accuracy with atf-idf : 0.815

2-classes
Accuracy with tp-unary : 0.846
Accuracy with tf-dbidf : 0.826
Accuracy with tf-dsidf : 0.833
Accuracy with bm25-bm25 : 0.846
Accuracy with atf-ne : 0.849
Accuracy with atf-wllr : 0.799

'''
from __future__ import division
from collections import Counter, defaultdict ,OrderedDict
from math import log
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

import text_processor

class TermWeightScorer():
	def __init__(self, tokenize=True, preprocess=True, stopwords=True, 
			tf_type='tp', k=0.5, k1=1.2, b=.95, idf_type='unary', smooth_factor=0):
		'''
		Parameters:
		-------------
		tokenize: True if text must be tokenized 

		preprocess; True if preprocessing must be done

		stopwords: True if stopwords must be removed

		tf_type: type of term frequency from ['tp', 'tf', 'atf', 'log_tf', 'bm25']
		tp - term presence, 1 if tf > 0, else 0
		tf - raw term frequency
		atf - augmented term freq. = k + ((k * tf) / max_tf) where max_tf
		log_tf - 1 + log(tf)
		bm25 - BM25 term freq

		k: parameter for augmented term freq. Ignored otherwise.

		k1: parameter for BM25. Ignored otherwise.

		b: parameter for BM25. Ignored otherwise.

		avg_dl: average number of terms in all documents. this will be set at training time.
		Relevant for bm25, ignored otherwise.
	
		idf_type: type of document/global frequency calculation from 
				['unary', 'idf', 'bm25', 'dsidf', 'dbidf', 'ne', 'wllr']
		unary - 1
		idf - inverse document frequency
		bm25 - BM25 inverse document frequency
		dsidf - delta smoothed idf, for 2-class classification
		dbidf - delta BM25 idf, for 2-class classification
		ne - natural entropy
		wllr - Weighted Log Likelihood Ratio
		
		smooth_factor: smoothing factor to be applied.
		'''
		self.tokenize = tokenize
		self.preprocess = preprocess
		self.stopwords = stopwords
		self.tf_type = tf_type
		self.k = k
		self.k1 = k1
		self.b = b
		self.idf_type = idf_type
		self.smooth_factor = smooth_factor
		self.X = None
		self.y = None
		self.N = 0
		self.vocab = {}
		self.class_vocab = defaultdict(dict)
		self.class_counts = defaultdict(dict)
		self.avg_dl = 0
		self.idf = {}
		self.vec = None

	def _tokenize(self):
		self.X = self.X.apply(lambda x: text_processor.tokenize(x))

	def _preprocess(self):
		if self.stopwords:
			self.X = self.X.apply(lambda x: 
					text_processor.remove_common_words(text_processor.lower(x)))
		else:
			self.X = self.X.apply(lambda x: text_processor.lower(x))

	def _common_doc_frequency(self):
		# vocab: Number of documents containing a term in corpus
		self.vocab = Counter([token for token_list in self.X for token in set(token_list)])

		# N: Number of documents in corpus
		self.N = len(self.X)

		if self.tf_type in ['bm25']:
			nb_terms_doc = self.X.apply(lambda x: len(x))
			# avg_dl: Average number of terms in all documents
			self.avg_dl = int(np.average(nb_terms_doc))

		if self.idf_type in ['dsidf', 'dbidf', 'ne', 'wllr']:
			data = np.column_stack((self.X, self.y))
			for lbl in np.unique(self.y):
				subset = data[data[:,1] == lbl]
				# class_counts: Number of documents present in class
				# class_vocab: Number of documents containing term in class
				self.class_counts[lbl] = subset.shape[0]
				self.class_vocab[lbl] = Counter([token for tokens in subset[:, 0] 
										for token in set(tokens)])

	def _term_frequency(self, token_list):
		tf = {}
		if self.tf_type == 'tf':
			tf = dict(Counter(token_list))
		elif self.tf_type == 'tp':
			tf = dict([(token, 1) for token in token_list])
		elif self.tf_type == 'atf':
			tf = dict(Counter(token_list))
			max_tf = max(tf.values())
			atf = dict(tf)
			for key, val in tf.items():
				atf[key] = self.k + ((self.k * val) / max_tf)
			tf = atf
		elif self.tf_type == 'log_tf':
			tf = dict(Counter(token_list))
			for key, val in tf.items():
				tf[key] = 1 + log(val)
		elif self.tf_type == 'bm25':
			tf = dict(Counter(token_list))
			bmtf = dict(tf)
			for key, val in tf.items():
				bmtf[key] = ((self.k1 + 1) * val) / ((self.k1 * ((1 - self.b) + 
							self.b * (len(token_list) / self.avg_dl))) + val)
			tf = bmtf
		return tf

	def _document_frequency(self):
		for token in self.vocab.keys():
			try:
				if self.idf_type == 'unary':
					self.idf[token] = 1.0
				elif self.idf_type == 'idf':
					self.idf[token] = log(self.smooth_factor + 
								(self.N / (1 + self.vocab[token])))
				elif self.idf_type == 'bm25':
					self.idf[token] = log((self.N - self.vocab[token] + 0.5) / 
											(self.vocab[token] + 0.5))
				elif self.idf_type == 'dsidf':
					labels = list(self.class_vocab.keys())
					assert (len(labels) == 2), "Only two classes allowed for dsidf."
					numerator = self.class_counts[labels[0]] * self.class_vocab[labels[1]][token] + 0.5
					denominator = self.class_counts[labels[1]] * self.class_vocab[labels[0]][token] + 0.5
					self.idf[token] = log(numerator / denominator)
				elif self.idf_type == 'dbidf':
					labels = list(self.class_vocab.keys())
					assert (len(labels) == 2), "Only two classes allowed for dbidf."
					numerator = (self.class_counts[labels[0]] - self.class_vocab[labels[1]][token] + 0.5) \
						* self.class_vocab[labels[0]][token] + 0.5
					denominator = (self.class_counts[labels[1]] - self.class_vocab[labels[0]][token] + 0.5) \
						* self.class_vocab[labels[1]][token] + 0.5
					self.idf[token] = log(numerator / denominator)
				elif self.idf_type == 'ne':
					labels = list(self.class_vocab.keys())
					assert (len(labels) == 2), "Only two classes allowed for ne."
					nb_cl0_tok = self.class_vocab[labels[0]][token]
					nb_cl1_tok = self.class_vocab[labels[1]][token]
					nb_tok = self.vocab[token]
					term0 = (nb_cl0_tok / nb_tok)
					if term0 != 0:
						term0 = term0 * log(term0)
					term1 = (nb_cl1_tok / nb_tok)
					if term1 != 0:
						term1 = term1 * log(term1)
					self.idf[token] = 1 + term0 + term1
				elif self.idf_type == 'wllr':
					labels = list(self.class_vocab.keys())
					assert (len(labels) == 2), "Only two classes allowed for wllr."
					nb_cl0_tok = self.class_vocab[labels[0]][token]
					nb_cl1_tok = self.class_vocab[labels[1]][token]
					nb_tok = self.vocab[token]
					term0 = ((nb_cl0_tok / nb_tok) * self.class_counts[labels[0]]) / nb_tok
					term1 = ((nb_cl1_tok / nb_tok) * self.class_counts[labels[1]]) / nb_tok
					if term0 != 0 and term1 != 0:
						self.idf[token] = term0 * log(term0 / term1)
					else:
						self.idf[token] = 1
			except Exception as e:
				raise Exception("Error while computing IDF scores", e)

	def _tf_idf(self, token_list):
		tf = self._term_frequency(token_list)
		if not self.idf:
			self._document_frequency()
		raw_tf_idfs, tf_idfs = {}, {}
		for token in token_list:
			if token not in self.vocab.keys():
				continue
			raw_tf_idfs[token] = tf[token] * self.idf[token]
		# cosine normalization
		normal = np.sqrt(np.sum(np.power(list(raw_tf_idfs.values()), 2)))
		for token in token_list:
			if token not in self.vocab.keys():
				continue
			tf_idfs[token] = raw_tf_idfs[token] / normal
		return tf_idfs

	def fit_transform(self, X, y):
		print('In fit transform...')
		self.X = pd.Series(X)
		self.y = pd.Series(y)

		if self.tokenize:
			self._tokenize()
		if self.preprocess:
			self._preprocess()
		
		self._common_doc_frequency()
		print('computing tf-idf scores...')
		tfidf_weights = self.X.apply(self._tf_idf)
		tw = tfidf_weights.tolist()
		
		self.vec = DictVectorizer()
		tw_new = self.vec.fit_transform(tw)
		return tw_new

	def transform(self, X):
		print('In transform...')		
		self.X = pd.Series(X)

		if self.tokenize:
			self._tokenize()
		if self.preprocess:
			self._preprocess()
		
		tfidf_weights = self.X.apply(self._tf_idf)
		tw = tfidf_weights.tolist()
		
		tw_new = self.vec.transform(tw)
		return tw_new


if __name__ == '__main__':
	PATH1 = "/home/amandadsouza/decrypt/datasets/train.csv"
	PATH2 = "/home/amandadsouza/decrypt/datasets/test.csv"
	text_field = 'Verbatim'
	label_field = 'NET_Name'
	encoding = 'utf-8' #'cp1252'

	train = pd.read_csv(PATH1, encoding=encoding)
	test = pd.read_csv(PATH2, encoding=encoding)
	print(train.shape, test.shape)

	train = train[train[label_field].isin(['Positive', 'Negative'])]
	test = test[test[label_field].isin(['Positive', 'Negative'])]

	X_train, y_train = train[text_field], train[label_field]
	X_test, y_test = test[text_field], test[label_field]

	tf_type, idf_type = 'atf', 'wllr'
	twm = TermWeightScorer(tf_type=tf_type, idf_type=idf_type)
	X_train = twm.fit_transform(X_train, y_train)
	X_test = twm.transform(X_test)

	print('Training model...')
	from sklearn.svm import LinearSVC
	clf = LinearSVC(multi_class='ovr', C=0.1)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)

	from sklearn.metrics import accuracy_score
	print('Accuracy with {}-{} : {}'.format(tf_type, idf_type, accuracy_score(y_test, preds)))
