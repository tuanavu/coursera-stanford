#!/usr/local/Cellar/python/2.7.6/bin/python
# -*- coding: utf-8 -*-
import sys
from numpy import *
import scipy, scipy.special
import handythread

class Util(object):
	def __init__(self):
		pass

	@staticmethod
	def featureNormalize( data ):
		mu 			= mean( data, axis=0 )
		data_norm 	= data - mu
		sigma 		= std( data_norm, axis=0, ddof=1 )
		data_norm 	= data_norm / sigma
		return data_norm, mu, sigma

	@staticmethod
	def sigmoid( z ):
		# return array(handythread.parallel_map( lambda z: 1.0 / (1.0 + exp(-z)), z ))
		return scipy.special.expit(z)
		

	@staticmethod
	def sigmoidGradient( z ):
		sig = Util.sigmoid(z)
		return sig * (1 - sig)

	@staticmethod
	def recodeLabel( y, k ):
		m = shape(y)[0]
		out = zeros( ( k, m ) )
		for i in range(0, m):
			out[y[i]-1, i] = 1
		return out

	@staticmethod
	def mod( length, divisor ):
		dividend = array([x for x in range(1, length+1)])
		divisor  = array([divisor for x in range(1, length+1)])
		return mod( dividend, divisor ).reshape(1, length )

	@staticmethod
	def fmincg( f, x0, fprime, args, maxiter=100 ):
		nargs = (x0,) + args

		realmin = finfo(double).tiny
		RHO = 0.01     # a bunch of constants for line searches
		SIG = 0.5      # rho and sig are the constants in the wolfe-powell conditions
		INT = 0.1      # don't reevaluate within 0.1 of the limit of the current bracket
		EXT = 3.0      # extrapolate maximum 3 times the current bracket
		MAX = 20       # max 20 function evaluations per line search
		RATIO = 100    # maximum allowed slope ratio5
		length = maxiter

		red = 1
		i 	= 0                                 # zero the run length counter
		ls_failed = False                           # no previous line search has failed
		fX 	= array([])
		f1 	= f(*nargs)						# get function value and gradient
		df1 = fprime(*nargs)
		i 	= i + (length<0)                    # count epochs?!
		s 	= -df1                              # search direction is steepest
		d1 	= -s.T.dot(s)                       # this is the slope
		z1 	= red/(1-d1)                        # initial step is red/(|s|+1)


		while ( i < abs( length )):
			i 		= i + (length>0)
			X0 		= copy( x0 )
			f0 		= copy( f1 )
			df0 	= copy( df1 )

			x0 		= x0 + (z1 * s).reshape( shape( x0 )[0], 1 )
			nargs 	= (x0,) + args
			f2 		= f( *nargs )
			df2 	= fprime( *nargs)

			i 		= i + (length<0)
			d2 		= df2.T.dot(s)

			
			f3 = copy(f1)	# initialize point 3 equal to point 1
			d3 = copy(d1) 
			z3 = copy(-z1)

			M = MAX if length > 0 else min( MAX, -length-i )
			success = False
			limit = -1

			while True:
				while ((f2 > f1 + z1 * RHO * d1) or (d2 > -SIG * d1)) and ( M > 0 ):
					limit = z1
					if f2 > f1:
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
					else:
						A = 6*(f2-f3)/z3+3*(d2+d3)                      # make cubic extrapolation
					  	B = 3*(f3-f2)-z3*(d3+2*d2)
					  	z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A
					
					if isnan( z2 ) or isinf( z2 ):
						z2 = z3 / 2

					z2 = max(min( z2, INT*z3 ), (1-INT)* z3)
					z1 = z1 + z2
					x0 = x0 + (z2 * s).reshape( shape( x0 )[0], 1 )
					nargs 	= (x0,) + args
					f2 		= f( *nargs )
					df2 	= fprime( *nargs)
					
					M = M - 1
					i = i + (length<0)
					
					d2 = df2.T.dot( s ) 	# numerically unstable here, but the value still stays as very small decimal number
					z3 = z3 - z2

				if (f2 > f1 + z1 * RHO * d1 ) or (d2 > -SIG * d1 ):
					break
				elif d2 > SIG * d1:
					success = True
					break
				elif M == 0:
					break

			  	A = 6*(f2-f3)/z3+3*(d2+d3)                      # make cubic extrapolation
			  	B = 3*(f3-f2)-z3*(d3+2*d2)
			  	z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3))        # num. error possible - ok!
			  	
			  	if not isreal(z2) or isnan(z2) or isinf(z2) or z2 < 0:
			  		if limit < -0.5:
			  			z2 = z1 * (EXT - 1)
		  			else:
		  				z2 = (limit - z1) / 2
		  		elif (limit > -0.5) and (z2+z1 > limit):
					z2 = (limit-z1)/2
				elif (limit < -0.5) and (z2+z1 > z1 * EXT ):
					z2 = z1 * (EXT - 1.0)
				elif z2 < -z3 * INT:
					z2 = -z3 * INT
				elif (limit > -0.5) and (z2 < (limit-z1) * (1.0-INT)):
					z2 = (limit-z1) *(1.0-INT)
				
				f3 = copy( f2 )
				d3 = copy( d2 )
				z3 = copy( -z2 )
				z1 = z1 + z2
				x0 = x0 + (z2 * s).reshape( shape( x0 )[0], 1 )
				nargs 	= (x0,) + args
				f2 		= f( *nargs )
				df2 	= fprime( *nargs)
				
				M = M - 1
				i = i + (length<0)
				d2 = df2.T.dot( s )
				
			if success is True:
				f1 = copy( f2 )
				
				tmp = []
				tmp[len(tmp):] = fX.tolist()
				tmp[len(tmp):] = [f1.tolist()]
				fX = array(tmp)
				
				s = (df2.T.dot(df2) - df1.T.dot(df2)) / (df1.T.dot(df1)) * s - df2
				
				tmp = copy( df1 )
				df1 = copy( df2 )
				df2 = copy( df1 )
				d2 = df1.T.dot( s )
				
				if d2 > 0:
					s = -df1
					d2 = -s.T.dot( s )

				z1 = z1 * min(RATIO, d1 / (d2-realmin))
				d1 = copy(d2)
				ls_failed = False
			else:
				x0 = copy( X0 )
				f1 = copy( f0 )
				df1 = copy( df0 )

				if ls_failed is True or i > abs(length):
					break

				tmp = copy( df1 )
				df1 = copy( df2 )
				df2 = copy( tmp )

				s = -df1
				d1 = -s.T.dot( s )
				z1 = 1 / (1 - d1)
				ls_failed = True
				
		return x0, fX




import unittest
class TestUtil(unittest.TestCase):
	def setUp( self ):
		pass

	def tearDown( self ) :
		pass

	def test_sigmoid( self ):
		self.assertEqual( Util.sigmoid( 0 ), 0.5 )
		
def main():
	print Util.sigmoid( array([0, 0, 1]) )
	print Util.sigmoidGradient( array([0, 0, 1]) )

	# y = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
	# print Util.recodeLabel( y, 10 )

	pass

	# print mat1
	# print Util.ravelMat( mat1, 3, 3 )

if __name__ == '__main__':
	# unittest.main()
	main()
