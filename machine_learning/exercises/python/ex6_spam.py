#!/usr/local/Cellar/python/2.7.6/bin/python
# -*- coding: utf-8 -*-

import sys
import string
import csv
import re
import pickle

from numpy import *
import nltk, nltk.stem.porter
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm, grid_search

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba
from util import Util

def emailFeatures( word_indices ):
	features = zeros((1899, 1))
	for index in word_indices:
		features[index] = 1
	return features

def processEmail( email_contents ):
	vocab_list = getVocabList()
	
	word_indices = []
	
	email_contents = email_contents.lower()
	email_contents = re.sub( '<[^<>]+>', ' ', email_contents )
	email_contents = re.sub( '[0-9]+', 'number', email_contents )
	email_contents = re.sub( '(http|https)://[^\s]*', 'httpaddr', email_contents )
	email_contents = re.sub( '[^\s]+@[^\s]+', 'emailaddr', email_contents )
	email_contents = re.sub( '[$]+', 'dollar', email_contents )
	
	stemmer = nltk.stem.porter.PorterStemmer()
	tokens = re.split( '[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']' , email_contents )
	
	for token in tokens:
		token = re.sub( '[^a-zA-Z0-9]', '', token )
		token = stemmer.stem( token.strip() )

		if len(token) == 0:
			continue

		if token in vocab_list:
			word_indices.append( vocab_list[token] )
			
	return word_indices
	

def getVocabList():
	vocab_list = {}
	with open('/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/vocab.txt', 'r') as file:
		reader = csv.reader( file, delimiter='\t' )
		for row in reader:
			vocab_list[row[1]] = int(row[0])
			
	return  vocab_list

def part2_1():
	email_contents = ''
	with open( '/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/emailSample1.txt', 'r' ) as f:
		email_contents = f.read()
	
	word_indices = processEmail( email_contents )
    
def part2_2():
	email_contents = ''
	with open( '/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/emailSample1.txt', 'r' ) as f:
		email_contents = f.read()
	
	word_indices = processEmail( email_contents )
	features 	 = emailFeatures( word_indices )

def part2_3():
	mat = scipy.io.loadmat( "/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/spamTrain.mat" )
	X, y = mat['X'], mat['y']


	linear_svm = pickle.load( open("linear_svm.svm", "rb") )

	# linear_svm = svm.SVC(C=0.1, kernel='linear')
	# linear_svm.fit( X, y.ravel() )
	# pickle.dump( linear_svm, open("linear_svm.svm", "wb") )

	predictions = linear_svm.predict( X )
	predictions = predictions.reshape( shape(predictions)[0], 1 )
	print ( predictions == y ).mean() * 100.0

	mat = scipy.io.loadmat( "/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/spamTest.mat" )
	X_test, y_test = mat['Xtest'], mat['ytest']

	predictions = linear_svm.predict( X_test )
	predictions = predictions.reshape( shape(predictions)[0], 1 )
	print ( predictions == y_test ).mean() * 100.0

	vocab_list = getVocabList()
	reversed_vocab_list = dict( (v, k) for (k, v) in vocab_list.iteritems() )
	sorted_indices = argsort( linear_svm.coef_, axis=None )

	for i in sorted_indices[0:15]:
		print reversed_vocab_list[i]

def part2_4():
	mat = scipy.io.loadmat( "/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/spamTrain.mat" )
	X, y = mat['X'], mat['y']

	# linear_svm = pickle.load( open("linear_svm.svm", "rb") )
	linear_svm = svm.SVC(C=0.1, kernel='linear')
	linear_svm.fit( X, y.ravel() )
	# pickle.dump( linear_svm, open("linear_svm.svm", "wb") )

	email_contents = ''
	with open( '/Users/saburookita/Downloads/mlclass-ex6-004/mlclass-ex6/spamSample2.txt', 'r' ) as f:
		email_contents = f.read()

	word_indices = processEmail( email_contents )
	features 	 = emailFeatures( word_indices ).transpose()

	print linear_svm.predict( features )



def main():
	part2_1()
	part2_2()
	part2_3()
	part2_4()


if __name__ == '__main__':
	main()