#!/usr/local/Cellar/python/2.7.6/bin/python
# -*- coding: utf-8 -*-

import sys

from numpy import *

import scipy.io, scipy.misc, scipy.optimize
from matplotlib import pyplot, cm, colors, lines
from mpl_toolkits.mplot3d import Axes3D


def loadMovieList():
	movies = {}
	counter = 0
	with open('/Users/saburookita/Downloads/mlclass-ex8-004/mlclass-ex8/movie_ids.txt', 'rb') as f:
		contents = f.readlines()
		for content in contents:
			movies[counter] = content.strip().split(' ', 1)[1]
			counter += 1

	return movies

def normalizeRatings( Y, R ):
	m = shape( Y )[0]
	Y_mean = zeros((m, 1))
	Y_norm = zeros( shape( Y ) )

	for i in range( 0, m ):
		idx 			= where( R[i] == 1 )
		Y_mean[i] 		= mean( Y[i, idx] )
		Y_norm[i, idx] 	= Y[i, idx] - Y_mean[i]

	return Y_norm, Y_mean

def unrollParams( params, num_users, num_movies, num_features ):
	X 		= params[:num_movies * num_features]
	X 		= X.reshape( (num_features, num_movies) ).transpose()
	theta 	= params[num_movies * num_features:]
	theta 	= theta.reshape( num_features, num_users ).transpose()
	return X, theta
	
def cofiGradFunc( params, Y, R, num_users, num_movies, num_features, lamda ):
	X, theta 	= unrollParams( params, num_users, num_movies, num_features )
	inner 		= X.dot( theta.T ) * R - Y
	X_grad 		= inner.dot( theta ) + lamda * X
	theta_grad 	= inner.T.dot( X ) + lamda * theta
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

			
def cofiCostFunc( params, Y, R, num_users, num_movies, num_features, lamda ):
	X, theta 	   = unrollParams( params, num_users, num_movies, num_features  )
	J 			   = 0.5 * sum( (X.dot( theta.T ) * R - Y) ** 2 )
	regularization = 0.5 * lamda * (sum( theta**2 ) + sum(X**2))
	return J + regularization

def part2_1():
	mat = scipy.io.loadmat('/Users/saburookita/Downloads/mlclass-ex8-004/mlclass-ex8/ex8_movies.mat')
	Y, R = mat['Y'], mat['R']

	print mean( extract ( Y[0,:] * R[0,:] > 0, Y[0, :] ) )

	pyplot.imshow( Y )
	pyplot.ylabel( 'Movies' )
	pyplot.xlabel( 'Users')
	pyplot.show()

def part2_2():
	mat = scipy.io.loadmat('/Users/saburookita/Downloads/mlclass-ex8-004/mlclass-ex8/ex8_movies.mat')
	Y, R = mat['Y'], mat['R']

	mat = scipy.io.loadmat('/Users/saburookita/Downloads/mlclass-ex8-004/mlclass-ex8/ex8_movieParams.mat')
	num_features = mat['num_features']
	num_users 	 = mat['num_users']
	num_movies 	 = mat['num_movies']
	X 			 = mat['X']
	theta 		 = mat['Theta']

	num_users    = 4
	num_features = 3
	num_movies 	 = 5

	X 		= X[:num_movies, :num_features]
	theta 	= theta[:num_users, :num_features]
	Y 		= Y[:num_movies, :num_users]
	R 		= R[:num_movies, :num_users]


	params = r_[X.T.flatten(), theta.T.flatten()]
	print cofiCostFunc( params, Y, R, num_users, num_movies, num_features, 0 )
	print cofiGradFunc( params, Y, R, num_users, num_movies, num_features, 0 )
	print cofiCostFunc( params, Y, R, num_users, num_movies, num_features, 1.5 )
	print cofiGradFunc( params, Y, R, num_users, num_movies, num_features, 1.5 )


def part2_3():
	movies = loadMovieList()

	my_ratings = zeros((1682, 1))
	my_ratings[0] = 4
	my_ratings[97] = 2
	my_ratings[6]  = 3
	my_ratings[11] = 5
	my_ratings[53] = 4
	my_ratings[63] = 5
	my_ratings[65] = 3
	my_ratings[68] = 5
	my_ratings[182] = 4
	my_ratings[225] = 5
	my_ratings[354] = 5
	
	# for i in range( 0, 1682 ):
	# 	if my_ratings[i] > 0:
	# 		print "Rated %d for %s" % (my_ratings[i], movies[i])

	mat = scipy.io.loadmat('/Users/saburookita/Downloads/mlclass-ex8-004/mlclass-ex8/ex8_movies.mat')
	Y, R = mat['Y'], mat['R']

	Y = c_[my_ratings, Y]
	R = c_[my_ratings > 0, R]

	Y_norm, Y_mean = normalizeRatings( Y, R )

	num_movies, num_users = shape( Y )
	num_features = 10


	X 		= random.randn( num_movies, num_features )
	theta 	= random.randn( num_users, num_features )
	initial_params = r_[X.T.flatten(), theta.T.flatten()]


	lamda = 10.0


	result = scipy.optimize.fmin_cg( cofiCostFunc, fprime=cofiGradFunc, x0=initial_params, \
									args=( Y, R, num_users, num_movies, num_features, lamda ), \
									maxiter=100, disp=True, full_output=True )
	J, params = result[1], result[0]

	X, theta = unrollParams( params, num_users, num_movies, num_features )
	prediction = X.dot( theta.T )

	my_prediction = prediction[:, 0:1] + Y_mean
	
	idx = my_prediction.argsort(axis=0)[::-1]
	my_prediction = my_prediction[idx]


	for i in range(0, 10):
		j = idx[i, 0]
		print "Predicting rating %.1f for movie %s" % (my_prediction[j], movies[j])



def main():
	# part2_1()
	# part2_2()
	part2_3()


if __name__ == '__main__':
	main()