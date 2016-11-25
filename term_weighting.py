'''
Implementation of various (main) term weighting techniques mentioned in this paper:
https://arxiv.org/abs/1610.03106
'''
from __future__ import division
from collections import Counter, defaultdict ,OrderedDict
from math import log
import numpy as np

def term_frequency(token_list, type="atf", k=0.5):
	'''
	ARGUMENTS:
	
	token_list: list of word tokens
	
	type: type of term frequency from ['tp', 'tf', 'log_tf', 'atf']
	tp - term presence, 1 if tf > 0, else 0
	tf - raw term frequency
	atf - augmented term freq. = k + ((1 - k) * tf / max_tf) where max_tf
	log_tf - log(tf + 1)

	k = parameter for augmented term freq. Ignored otherwise

	RETURNS:
	tf - Dictionary of {term: "frequency"} where freq is calculated by selected type
	'''
	tf = {}
	if type == 'tf':
		tf = dict(Counter(token_list))
	elif type == 'tp':
		for w in token_list:
			tf[w] = 1
	elif type == 'atf':
		tf = dict(Counter(token_list))
		max_tf = max(tf.values())
		atf = dict(tf)
		for key, val in tf.items():
			atf[key] = k + ((1 - k) * val / max_tf)
		tf = atf
	elif type == 'log_tf':
		tf = dict(Counter(token_list))
		for k, v in tf.items():
			tf[k] = log(v + 1)
	return tf

def common_doc_frequency(sentence_token_list, labels):
	'''
	Function to compute vocabulary term frequency and per class term freq.
	
	ARGUMENTS:
	sentence_token_list: list/array of sentence token lists
  
	'''
	# vocab: Number of documents containing a term in corpus
	vocab = Counter([token for token_list in sentence_token_list for token in set(token_list)])

	data = np.column_stack((sentence_token_list, labels))

	class_vocab = defaultdict(dict)
	class_counts = defaultdict(int)
	for lbl in np.unique(labels):
		subset = data[data[:,1] == lbl]
		# class_counts: Number of documents present in class
		# class_vocab: Number of documents containing term in class
		class_counts[lbl] = subset.shape[0]
		class_vocab[lbl] = Counter([token for tokens in subset[:, 0] for token in set(tokens)])

	# N: Number of documents in corpus
	N = len(sentence_token_list)
	return N, vocab, class_vocab, class_counts

def document_frequency(token_list, label, N, vocab, class_vocab, class_counts, type="dsidf", smooth_factor=0):
	'''
	ARGUMENTS:

	sentence_token_list: list of sentence token list e.g. [['I', 'am'], ['Who', 'goes']] 

	labels: list of labels for sentences e.g. ['Positive', 'Negative']
	
	type: type of document/global frequency calculation from ['unary', 'idf', 'dsidf', 'dbidf', 'rf']
	unary - 1
	idf - inverse document frequency
	dsidf - delta smoothed idf
	dbidf - delta BM25 idf
	rf - relevance frequency

	RETURNS:

	idf - Dictionary of document/global frequency of each term in vocab, 
	calculated by chosen type  
	'''
	idf = {}
	for token in token_list:
		try:
			if type == 'unary':
				idf[token] = 1.0
			elif type == 'idf':
				idf[token] = log(smooth_factor + (N / (1 + vocab[token])))
			elif type == 'dsidf':
				Nclass_prime, df_prime = 0, 0
				for key in class_vocab.keys():
					if key != label:
						Nclass_prime += class_counts[key]
						df_prime += class_vocab[key][token]
				numerator = (class_counts[label] * df_prime + 0.5)
				denominator = (Nclass_prime * class_vocab[label][token] + 0.5)
				idf[token] = log(numerator / denominator)
			elif type == 'dbidf':
				Nclass_prime, df_prime = 0, 0
				for key in class_vocab.keys():
					if key != label:
						Nclass_prime += class_counts[key]
						df_prime += class_vocab[key][token]
				numerator = (class_counts[label] - class_vocab[label][token] + 0.5) * df_prime + 0.5
				denominator = (Nclass_prime - df_prime + 0.5) * class_vocab[label][token] + 0.5
				idf[token] = log(numerator / denominator)
			elif type == 'rf':
				df_prime = 0, 0
				for key in class_vocab.keys():
					if key != label:
						df_prime += class_vocab[key][token]
				idf[token] = log(2 + (class_vocab[label][token] / df_prime))
		except Exception as e:
			print("Error processing ", token)
			print(e)
	return idf
						
def tf_idf(x, N, vocab, class_vocab, class_counts, idf_type="dbidf", smooth_factor=0, tf_type="atf", k=0.5):
	token_list, label = x[0], x[1]
	tf = term_frequency(token_list, type=tf_type, k=k)
	idf = document_frequency(token_list, label, N, vocab, class_vocab, class_counts, type=idf_type, smooth_factor=0)
	raw_tf_idfs, tf_idfs = {}, {}
	for token in token_list:
		raw_tf_idfs[token] = tf[token] * idf[token]
	normal = np.sqrt(np.sum(np.power(list(raw_tf_idfs.values()), 2)))
	for token in token_list:
		tf_idfs[token] = raw_tf_idfs[token] / normal
	return tf_idfs

if __name__ == '__main__':
	data = [
	['I', 'love', 'macaroni','and','cheese', 'at', 'this', 'resto'],
	['This', 'restaurant', 'is', 'not', 'the', 'best'],
	['I', 'think', 'you', 'are', 'mistaken', 'when', 'you', 'say', 'this', 'place', 'sucks', '!'],
	['Sure', 'we', 'need', 'more', 'fast', 'food', 'and', 'obesity'],
	]

	labels = ['Positive', 'Negative', 'Positive', 'Negative']

	N, vocab, class_vocab, class_counts = common_doc_frequency(data, labels)

	for row, label in zip(data, labels):
		full_row = [row, label]
		print(tf_idf(full_row, N, vocab, class_vocab, class_counts))
