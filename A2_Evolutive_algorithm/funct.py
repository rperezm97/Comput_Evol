# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:21:08 2023

@author: berti
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funct import Function
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:21:08 2023

@author: berti
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the child class "Powell_sum"
class Powell_sum(Function):
    """
    Class representing the Powell sum function. Inherits from the Function 
    class.

    Methods:
        function (x): Specifies the implementation of the Powell sum function.
    """
    def __init__(self):
        """
        Constructor for the Powell_sum class.
        """
        # Call the constructor of the parent class 
        super().__init__("Powell Sum", dom=(-1,1))
    
    def function(self, x):
        """
        Specifies the implementation of the Powell sum function.
        
        Parameters
        ----------
        x : np.array or list with k n-dimensional points where to 
        evaluate the function. Also can be one ndarray representing one point
        
        Returns
        -------
        np.array of dimension k, or float if the input is just one point
        """
        # We need the dimension of the points /the point
        n = x.shape[1] if len(x.shape)==2 else x.shape[0]
        return np.sum(np.abs(x)**(np.arange(1, n+1) + 1), axis=1)

# Define the child class "Xin_she_yang_4"
class Xin_she_yang_4(Function):
    """
    Class to implement Xin-She Yang's Function No.04.
    This class is a child class of the class Function.
    """
    
    def __init__(self):
        """
        Constructor of the class Xin_she_yang_4.
        """
        # Call the constructor of the parent class 
        super().__init__("Xin-She Yang's Function No.04", dom=(-10,10))
    
    def function(self, x):
        """
        Implements Xin-She Yang's Function No.04.
        
        Parameters:
        x : np.array or list with k n-dimensional points where to 
        evaluate teh function
        
        Returns:
        np.array of dimension k (since its a scalar function)
        """
        # Define the dimension of the array the represents the variables
        a=len(x.shape)-1
        return np.sum(np.sin(x)**2 , axis=a)- np.exp(-np.sum(x**2, axis=a))* \
            np.exp(-np.sum(np.sin(np.sqrt(abs(x)))**2, axis=a))


if __name__ == "__main__":
    
    powell = Powell_sum()
    powell.plot()
    
    xin_she_yang = Xin_she_yang_4()
    xin_she_yang.plot()