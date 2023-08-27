# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:21:08 2023

@author: berti
"""
import numpy as np

class Kernel:
    def __init__(self, f=None, *args):
        """Abstract class for a Kernel, that can be multiplied  and summed. 
        The (elementary) function f defined by calling the different kernels
        with the arguments will be defined in the children"""
        self.kernels=[]
        self.weights=[]
    def f(self,x):
        if len(self.kernels)==len(self.weights):
            np.sum([self.kernels[i](x)*self.weights[i]] 
                        for i in range(len(self.weights)))
        
    def __mul__(self, w):
        """Define multiplication of kernels.
        Let f=K(args) be a function defined by a kernel K and w a constant.
        Then multiplication is defined (w*f)(x)=w*f(x)"""
        self.weights.append(w)
        return self

    def __rmul__(self, other):
        """Define reverse multiplication to make the operation
        conmutative"""
        return self.__mul__(other)

    def __add__(self, other):
        """Define addition of kernels.
        Let f=K(args) be a function defined by a kernel K, c a constant and g
        another function defined by a kernel.
        Then addition is defined as (c+f)(x)=c+f(x) and (f+g)(x)=c+f(x)"""
        
        if isinstance(other, Kernel):
            self.kernels+=other.kernels
            self.weights+=other.weights
            return self
        else:
            print("addition is only defined for kernels")

    def __radd__(self, other):
        """Define reverse adding to make the operation conmutative"""
        return self.__add__(other)
    

class KG(Kernel):
    def __init__(self, lambd, c, null=None):
        """ Class for the Gaussian kernel."""
        self.lambd = lambd
        self.c = c
        self.kernels=[self.f]
    def eval(self, x):
        return np.exp(-self.lambd * (self.c - x) ** 2)


class KP(Kernel):
    def __init__(self, alpha, beta, d):
        """ Class for the Polynomial kernel."""
        self.alpha = alpha
        self.beta = beta
        self.d = d

    def f(self, x):
        return (self.alpha * x + self.beta) ** self.d


class KS(Kernel):
    def __init__(self, delta, theta, null=None):
        """ Class for the Sigmoid kernel."""
        self.delta = delta
        self.theta = theta

    def f(self, x):
        return np.tanh(self.delta * x + self.theta)
