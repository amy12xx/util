from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

import text_processor
import term_weighting

class TermWeightMaxtrix():
	def __init__(self, tf_weight_type='tf', idf_weight_type='idf'):
		self.X = None
		self.y = None
		self.Xy = None
		self.tf_weight_type = tf_weight_type
		self.idf_weight_type = idf_weight_type

	def _tokenize(self):
		self.X = self.X.apply(lambda x: text_processor.tokenize(x))

	def _preprocess(self, stopwords=True):
		self.X = self.X.apply(lambda x: 
					text_processor.remove_common_words(text_processor.lower(x)))

	def fit_transform(self, X, y, tokenize=True, preprocess=True, stopwords=True,):
		self.X = pd.Series(X)
		self.y = pd.Series(y)

		if tokenize:
			self._tokenize()
		if preprocess:
			self._preprocess(stopwords=stopwords)
		
		self.Xy = pd.DataFrame(np.column_stack((self.X, self.y)))
		N, vocab, class_vocab, class_counts = term_weighting.common_doc_frequency(self.X, self.y)

		tfidf_weights = self.Xy.apply(term_weighting.tf_idf, axis=1, args=(N, vocab, class_vocab, class_counts))
		tw = tfidf_weights.tolist()
		
		vec = DictVectorizer()
		tw_new = vec.fit_transform(tw)
		print(tw_new)


PATH = "/home/amandadsouza/decrypt/datasets/Verbatims_with_sentiment_fudged.csv"
text_field = 'Verbatim'
label_field = 'NET_Name'
encoding = 'cp1252'
data = pd.read_csv(PATH, encoding=encoding)
X, y = data[text_field], data[label_field]

twm = TermWeightMaxtrix()
Xnew = twm.fit_transform(X, y)
