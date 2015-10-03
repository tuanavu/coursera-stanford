#!/usr/local/Cellar/python/2.7.6/bin/python
# -*- coding: utf-8 -*-

import sys
import scipy.optimize, scipy.special
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

def plot( data ):
	negatives = data[data[:, 2] == 0]
	positives = data[data[:, 2] == 1]

	pyplot.xlabel("Microchip test 1")
	pyplot.ylabel("Microchip test 2")
	pyplot.xlim([-1.0, 1.5])
	pyplot.ylim([-1.0, 1.5])

	pyplot.scatter( negatives[:,0], negatives[:,1], c='y', marker='o', linewidths=1, s=40, label='y=0' )
	pyplot.scatter( positives[:,0], positives[:,1], c='k', marker='+', linewidths=2, s=40, label='y=1' )
	
	pyplot.legend()


def mapFeature( X1, X2 ):
	degrees = 6
	out = ones( (shape(X1)[0], 1) )
	
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1 ** (i-j)
			term2 = X2 ** (j)
			term  = (term1 * term2).reshape( shape(term1)[0], 1 ) 
			out   = hstack(( out, term ))
	return out

def sigmoid( z ):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))

def gradientCost( theta, X, y, lamda ):
	m = shape( X )[0]
	grad = X.T.dot( sigmoid( X.dot( theta ) ) - y ) / m
	grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
	return grad

def computeCost( theta, X, y, lamda ):
	m = shape( X )[0]
	hypo 	   = sigmoid( X.dot( theta ) )
	term1 	   = log( hypo ).dot( -y )
	term2 	   = log( 1.0 - hypo ).dot( 1 - y )
	left_hand  = (term1 - term2) / m
	right_hand = theta.transpose().dot( theta ) * lamda / (2*m)
	return left_hand + right_hand

def costFunction( theta, X, y, lamda ):
	cost 	 = computeCost( theta, X, y, lamda )
	gradient = gradientCost( theta, X, y, lamda )
	return cost

def findMinTheta( theta, X, y, lamda ):
	result = scipy.optimize.minimize( costFunction, theta, args=(X, y, lamda),  method='BFGS', options={"maxiter":500, "disp":True} )
	return result.x, result.fun

def part2_1():
	data  = genfromtxt( "/Users/saburookita/Downloads/mlclass-ex2-004/mlclass-ex2/ex2data2.txt", delimiter = ',' )
	plot( data )
	pyplot.show()

def part2_2():
	data  = genfromtxt( "/Users/saburookita/Downloads/mlclass-ex2-004/mlclass-ex2/ex2data2.txt", delimiter = ',' )
	X 	  = mapFeature( data[:, 0], data[:, 1] )
	print X

def part2_3():
	data  = genfromtxt( "/Users/saburookita/Downloads/mlclass-ex2-004/mlclass-ex2/ex2data2.txt", delimiter = ',' )
	y 	  = data[:,2]
	X 	  = mapFeature( data[:, 0], data[:, 1] )
	theta = zeros( shape(X)[1] )
	lamda = 1.0
	print computeCost( theta, X, y, lamda )

	theta, cost = findMinTheta( theta, X, y, lamda )

def part2_4():
	data  = genfromtxt( "/Users/saburookita/Downloads/mlclass-ex2-004/mlclass-ex2/ex2data2.txt", delimiter = ',' )
	y 	  = data[:,2]
	X 	  = mapFeature( data[:, 0], data[:, 1] )
	theta = zeros( shape(X)[1] )
	lamdas = [0.0, 1.0, 100]

	for lamda in lamdas:
		theta, cost = findMinTheta( theta, X, y, lamda )

		pyplot.text( 0.15, 1.4, 'Lamda %.1f' % lamda )
		plot( data )

		u = linspace( -1, 1.5, 50 )
		v = linspace( -1, 1.5, 50 )
		z = zeros( (len(u), len(v)) )

		for i in range(0, len(u)): 
			for j in range(0, len(v)):
				mapped = mapFeature( array([u[i]]), array([v[j]]) )
				z[i,j] = mapped.dot( theta )
		z = z.transpose()

		u, v = meshgrid( u, v )	
		pyplot.contour( u, v, z, [0.0, 0.0], label='Decision Boundary' )		

		pyplot.show()

def main():
	set_printoptions(precision=6, linewidth=200)
	part2_1()
	part2_2()
	part2_3()
	part2_4()
	

if __name__ == '__main__':
	main()